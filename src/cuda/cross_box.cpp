/*  cross_box.cpp
 *
 *
 */

#include "cross_box.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <algorithm>
#include "cudaErr.h"
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace std;

const double D_PI = 3.14159265358979;

// Just setting things up here
//  configure_cells() decides how the space should be divided into cells
//  and which cells are next to each other
void Cross_Box::configure_cells()
{
  assert(m_dRMax > 0.0);
  assert(m_dAMax >= 0.0);

  // Minimum height & width of cells
  //  Width is set so that it is only possible for particles in 
  //  adjacent cells to interact as long as |gamma| < 0.5
  double dWMin = 2.24 * (m_dRMax + m_dAMax) + m_dEpsilon;
  double dHMin = 2 * (m_dRMax + m_dAMax) + m_dEpsilon;

  m_nCellRows = max(static_cast<int>(m_dL / dHMin), 1);
  m_nCellCols = max(static_cast<int>(m_dL / dWMin), 1);
  m_nCells = m_nCellRows * m_nCellCols;
  cout << "Cells: " << m_nCells << ": " << m_nCellRows << " x " << m_nCellCols << endl;

  m_dCellW = m_dL / m_nCellCols;
  m_dCellH = m_dL / m_nCellRows;
  cout << "Cell dimensions: " << m_dCellW << " x " << m_dCellH << endl;

  h_pnPPC = new int[m_nCells];
  h_pnCellList = new int[m_nCells * m_nMaxPPC];
  h_pnAdjCells = new int[8 * m_nCells];
  cudaMalloc((void **) &d_pnPPC, m_nCells * sizeof(int));
  cudaMalloc((void **) &d_pnCellList, m_nCells*m_nMaxPPC*sizeof(int));
  cudaMalloc((void **) &d_pnAdjCells, 8*m_nCells*sizeof(int));
  m_nDeviceMem += m_nCells*(9+m_nMaxPPC)*sizeof(int);
#if GOLD_FUNCS == 1
  g_pnPPC = new int[m_nCells];
  g_pnCellList = new int[m_nCells * m_nMaxPPC];
  g_pnAdjCells = new int[8 * m_nCells];
#endif
  
  // Make a list of which cells are next to each cell
  // This is done once for convinience since the boundary conditions
  //  make this more than trivial
  for (int c = 0; c < m_nCells; c++)
    {
      int nRow = c / m_nCellCols; 
      int nCol = c % m_nCellCols;

      int nAdjCol1 = (nCol + 1) % m_nCellCols;
      int nAdjCol2 = (m_nCellCols + nCol - 1) % m_nCellCols;
      h_pnAdjCells[8 * c] = nRow * m_nCellCols + nAdjCol1;
      h_pnAdjCells[8 * c + 1] = nRow * m_nCellCols + nAdjCol2;

      int nAdjRow = (nRow + 1) % m_nCellRows;
      h_pnAdjCells[8 * c + 2] = nAdjRow * m_nCellCols + nCol;
      h_pnAdjCells[8 * c + 3] = nAdjRow * m_nCellCols + nAdjCol1;
      h_pnAdjCells[8 * c + 4] = nAdjRow * m_nCellCols + nAdjCol2;
      
      nAdjRow = (m_nCellRows + nRow - 1) % m_nCellRows;
      h_pnAdjCells[8 * c + 5] = nAdjRow * m_nCellCols + nCol;
      h_pnAdjCells[8 * c + 6] = nAdjRow * m_nCellCols + nAdjCol1;
      h_pnAdjCells[8 * c + 7] = nAdjRow * m_nCellCols + nAdjCol2;
    }
  cudaMemcpy(d_pnAdjCells, h_pnAdjCells, 8*m_nCells*sizeof(int), cudaMemcpyHostToDevice);
  checkCudaError("Configuring cells");
}

// Set the thread configuration for kernel launches
void Cross_Box::set_kernel_configs()
{
  switch (m_nCross)
    {
   	   case 256:
   		   m_nGridSize = 2;
   		   m_nBlockSize = 128;
   		   m_nSM_CalcF = 3*128*sizeof(double);
   		   m_nSM_CalcSE = 4*136*sizeof(double);
   		   break;
   	   case 512:
   		   m_nGridSize = 4;
   		   m_nBlockSize = 128;
   		   m_nSM_CalcF = 3*128*sizeof(double);
   		   m_nSM_CalcSE = 4*136*sizeof(double);
   		   break;
   	   case 1024:
   		   m_nGridSize = 4;  // Grid size (# of thread blocks)
   		   m_nBlockSize = 256; // Block size (# of threads per block)
   		   m_nSM_CalcF = 3*256*sizeof(double);
   		   m_nSM_CalcSE = 4*264*sizeof(double); // Size of shared memory per block
   		   break;
   	   default:
   		   m_nGridSize = m_nCross / 512;
   		   m_nBlockSize = 512;
   		   m_nSM_CalcF = 3*512*sizeof(double);
   		   m_nSM_CalcSE = 4*520*sizeof(double);
    };
  cout << "Kernel config (cross):\n";
  cout << m_nGridSize << " x " << m_nBlockSize << endl;
  cout << "Shared memory allocation (calculating forces):\n";
  cout << (float)m_nSM_CalcF / 1024. << "KB" << endl;
  cout << "Shared memory allocation (calculating S-E):\n";
  cout << (float)m_nSM_CalcSE / 1024. << " KB" << endl; 
}

void Cross_Box::construct_defaults()
{
  m_dGamma = 0.0;
  m_dTotalGamma = 0.0;
  m_dStep = 1;
  m_dStrainRate = 1e-3;
  m_szDataDir = "./";
  m_szFileSE = "sd_stress_energy.dat";

  cudaHostAlloc((void**) &h_bNewNbrs, sizeof(int), 0);
  *h_bNewNbrs = 1;
  cudaMalloc((void**) &d_bNewNbrs, sizeof(int));
  cudaMemcpyAsync(d_bNewNbrs, h_bNewNbrs, sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void**) &d_pdTempX, sizeof(double)*m_nCross);
  cudaMalloc((void**) &d_pdTempY, sizeof(double)*m_nCross);
  cudaMalloc((void**) &d_pdTempPhi, sizeof(double)*m_nCross);
  cudaMalloc((void**) &d_pdXMoved, sizeof(double)*m_nCross);
  cudaMalloc((void**) &d_pdYMoved, sizeof(double)*m_nCross);
  cudaMalloc((void**) &d_pdMOI, sizeof(double)*m_nCross);
  cudaMalloc((void**) &d_pdIsoC, sizeof(double)*m_nCross);
  m_bMOI = 0;
  m_nDeviceMem += 6*m_nCross*sizeof(double);
#if GOLD_FUNCS == 1
  *g_bNewNbrs = 1;
  g_pdTempX = new double[m_nCross];
  g_pdTempY = new double[m_nCross];
  g_pdTempPhi = new double[m_nCross];
  g_pdXMoved = new double[m_nCross];
  g_pdYMoved = new double[m_nCross];
#endif
  
  // Stress, energy, & force data
  cudaHostAlloc((void **)&h_pfSE, 4*sizeof(float), 0);
  m_pfEnergy = h_pfSE;
  m_pfPxx = h_pfSE+1;
  m_pfPyy = h_pfSE+2;
  m_pfPxy = h_pfSE+3;
  m_fP = 0;
  h_pdFx = new double[m_nCross];
  h_pdFy = new double[m_nCross];
  h_pdFt = new double[m_nCross];
  // GPU
  cudaMalloc((void**) &d_pfSE, 4*sizeof(float));
  cudaMalloc((void**) &d_pdFx, m_nCross*sizeof(double));
  cudaMalloc((void**) &d_pdFy, m_nCross*sizeof(double));
  cudaMalloc((void**) &d_pdFt, m_nCross*sizeof(double));
  m_nDeviceMem += 4*sizeof(float) + 3*m_nCross*sizeof(double);
 #if GOLD_FUNCS == 1
  g_pfSE = new float[4];
  g_pdFx = new double[m_nCross];
  g_pdFy = new double[m_nCross];
  g_pdFt = new double[m_nCross];
#endif

  // Cell & neighbor data
  h_pnCellID = new int[m_nCross];
  cudaMalloc((void**) &d_pnCellID, sizeof(int)*m_nCross);
  m_nDeviceMem += m_nCross*sizeof(int);
#if GOLD_FUNCS == 1
  g_pnCellID = new int[m_nCross];
#endif
  configure_cells();
  
  h_pnNPP = new int[m_nCross];
  h_pnNbrList = new int[m_nCross*m_nMaxNbrs];
  cudaMalloc((void**) &d_pnNPP, sizeof(int)*m_nCross);
  cudaMalloc((void**) &d_pnNbrList, sizeof(int)*m_nCross*m_nMaxNbrs);
  m_nDeviceMem += m_nCross*(1+m_nMaxNbrs)*sizeof(int);
#if GOLD_FUNCS == 1
  g_pnNPP = new int[m_nCross];
  g_pnNbrList = new int[m_nCross*m_nMaxNbrs];
#endif

  set_kernel_configs();	
}

double Cross_Box::calculate_packing()
{
  double dParticleArea = 0.0;
  for (int p = 0; p < m_nCross; p++)
    {
      dParticleArea += (4*h_pdAx[p] + 4*h_pdAy[p] + (2*D_PI - 4)*h_pdR[p]) * h_pdR[p];
    }
  return dParticleArea / (m_dL * m_dL);
}

// Creates the class
// See cross_box.h for default values of parameters
Cross_Box::Cross_Box(int nCross, double dL, double dR, double dAx, double dAy, double dEpsilon, int nMaxPPC, int nMaxNbrs, Potential ePotential)
{
  assert(nCross > 0);
  m_nCross = nCross;
  assert(dL > 0.0);
  m_dL = dL;
  m_ePotential = ePotential;

  m_dEpsilon = dEpsilon;
  m_nMaxPPC = nMaxPPC;
  m_nMaxNbrs = nMaxNbrs;
  m_dRMax = dR;
  m_dAMax = dAx;
  m_nDeviceMem = 0;

  // This allocates the coordinate data as page-locked memory, which
  //  transfers faster, since they are likely to be transferred often
  cudaHostAlloc((void**)&h_pdX, nCross*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdY, nCross*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdPhi, nCross*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdR, nCross*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdAx, nCross*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdAy, nCross*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pnMemID, nCross*sizeof(int), 0);
  m_dPacking = 0;
  // This initializes the arrays on the GPU
  cudaMalloc((void**) &d_pdX, sizeof(double)*nCross);
  cudaMalloc((void**) &d_pdY, sizeof(double)*nCross);
  cudaMalloc((void**) &d_pdPhi, sizeof(double)*nCross);
  cudaMalloc((void**) &d_pdR, sizeof(double)*nCross);
  cudaMalloc((void**) &d_pdAx, sizeof(double)*nCross);
  cudaMalloc((void**) &d_pdAy, sizeof(double)*nCross);
  cudaMalloc((void**) &d_pnInitID, sizeof(int)*nCross);
  cudaMalloc((void**) &d_pnMemID, sizeof(int)*nCross);
  // Crossinders
  m_nDeviceMem += nCross*(6*sizeof(double) + 2*sizeof(int));
#if GOLD_FUNCS == 1
  g_pdX = new double[nCross];
  g_pdY = new double[nCross];
  g_pdPhi = new double[nCross];
  g_pdR = new double[nCross];
  g_pdAx = new double[nCross];
  g_pdAy = new double[nCross];
  g_pnInitID = new int[nCross];
  g_pnMemID = new int[nCross];
#endif

  construct_defaults();
  cout << "Memory allocated on device (MB): " << (double)m_nDeviceMem / (1024.*1024.) << endl;
  place_random_cross(0,1);
  m_dPacking = calculate_packing();
  cout << "Random crosses placed" << endl;
  //display(0,0,0,0);
  
}
// Create class with coordinate arrays provided
Cross_Box::Cross_Box(int nCross, double dL, double *pdX, double *pdY, double *pdPhi, double *pdR, double *pdAx,
		     double *pdAy, double dEpsilon, int nMaxPPC, int nMaxNbrs, Potential ePotential)
{
  assert(nCross > 0);
  m_nCross = nCross;
  assert(dL > 0);
  m_dL = dL;
  m_ePotential = ePotential;

  m_dEpsilon = dEpsilon;
  m_nMaxPPC = nMaxPPC;
  m_nMaxNbrs = nMaxNbrs;

  // This allocates the coordinate data as page-locked memory, which 
  //  transfers faster, since they are likely to be transferred often
  cudaHostAlloc((void**)&h_pdX, nCross*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdY, nCross*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdPhi, nCross*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdR, nCross*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdAx, nCross*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdAy, nCross*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pnMemID, nCross*sizeof(int), 0);
  m_dRMax = 0.0;
  m_dAMax = 0.0;
  for (int p = 0; p < nCross; p++)
    {
      h_pdX[p] = pdX[p];
      h_pdY[p] = pdY[p];
      h_pdPhi[p] = pdPhi[p];
      h_pdR[p] = pdR[p];
      h_pdAx[p] = pdAx[p];
      h_pdAy[p] = pdAy[p];
      assert(h_pdAx[p] >= h_pdAy[p]);
      h_pnMemID[p] = p;
      if (pdR[p] > m_dRMax)
    	  m_dRMax = pdR[p];
      if (pdAx[p] > m_dAMax)
    	  m_dAMax = pdAx[p];
      while (h_pdX[p] > dL)
    	  h_pdX[p] -= dL;
      while (h_pdX[p] < 0)
    	  h_pdX[p] += dL;
      while (h_pdY[p] > dL)
    	  h_pdY[p] -= dL;
      while (h_pdY[p] < 0)
    	  h_pdY[p] += dL;
    }
  m_dPacking = calculate_packing();

  // This initializes the arrays on the GPU
  cudaMalloc((void**) &d_pdX, sizeof(double)*nCross);
  cudaMalloc((void**) &d_pdY, sizeof(double)*nCross);
  cudaMalloc((void**) &d_pdPhi, sizeof(double)*nCross);
  cudaMalloc((void**) &d_pdR, sizeof(double)*nCross);
  cudaMalloc((void**) &d_pdAx, sizeof(double)*nCross);
  cudaMalloc((void**) &d_pdAy, sizeof(double)*nCross);
  cudaMalloc((void**) &d_pnInitID, sizeof(int)*nCross);
  cudaMalloc((void**) &d_pnMemID, sizeof(int)*nCross);
  // This copies the values to the GPU asynchronously, which allows the
  //  CPU to go on and process further instructions while the GPU copies.
  //  Only workes on page-locked memory (allocated with cudaHostAlloc)
  cudaMemcpyAsync(d_pdX, h_pdX, sizeof(double)*nCross, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdY, h_pdY, sizeof(double)*nCross, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdPhi, h_pdPhi, sizeof(double)*nCross, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdR, h_pdR, sizeof(double)*nCross, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdAx, h_pdAx, sizeof(double)*nCross, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdAy, h_pdAy, sizeof(double)*nCross, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pnMemID, h_pnMemID, sizeof(int)*nCross, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pnInitID, d_pnMemID, sizeof(int)*nCross, cudaMemcpyDeviceToDevice);

  m_nDeviceMem += nCross*(5*sizeof(double)+2*sizeof(int));

#if GOLD_FUNCS == 1
  g_pdX = new double[nCross];
  g_pdY = new double[nCross];
  g_pdPhi = new double[nCross];
  g_pdR = new double[nCross];
  g_pdAx = new double[nCross];
  g_pdAy = new double[nCross];
  g_pnInitID = new int[nCross];
  g_pnMemID = new int[nCross];
  for (int p = 0; p < nCross; p++)
    {
      g_pdX[p] = h_pdX[p];
      g_pdY[p] = h_pdY[p];
      g_pdPhi[p] = h_pdPhi[p];
      g_pdR[p] = h_pdR[p];
      g_pdAx[p] = h_pdAx[p];
      g_pdAy[p] = h_pdAy[p];
      g_pnMemID[p] = h_pnMemID[p];
      g_pnInitID[p] = g_pnMemID[p];
    }
#endif

  construct_defaults();
  cout << "Memory allocated on device (MB): " << (double)m_nDeviceMem / (1024.*1024.) << endl;
  // Get spheocyl coordinates from cross

  cudaThreadSynchronize();
  //display(0,0,0,0);
}

//Cleans up arrays when class is destroyed
Cross_Box::~Cross_Box()
{
  // Host arrays
  cudaFreeHost(h_pdX);
  cudaFreeHost(h_pdY);
  cudaFreeHost(h_pdPhi);
  cudaFreeHost(h_pdR);
  cudaFreeHost(h_pdAx);
  cudaFreeHost(h_pdAy);
  cudaFreeHost(h_pnMemID);
  cudaFreeHost(h_bNewNbrs);
  cudaFreeHost(h_pfSE);
  delete[] h_pdFx;
  delete[] h_pdFy;
  delete[] h_pdFt;
  delete[] h_pnCellID;
  delete[] h_pnPPC;
  delete[] h_pnCellList;
  delete[] h_pnAdjCells;
  delete[] h_pnNPP;
  delete[] h_pnNbrList;
  
  // Device arrays
  cudaFree(d_pdX);
  cudaFree(d_pdY);
  cudaFree(d_pdPhi);
  cudaFree(d_pdR);
  cudaFree(d_pdAx);
  cudaFree(d_pdAy);
  cudaFree(d_pdTempX);
  cudaFree(d_pdTempY);
  cudaFree(d_pdTempPhi);
  cudaFree(d_pnInitID);
  cudaFree(d_pnMemID);
  cudaFree(d_pdMOI);
  cudaFree(d_pdIsoC);
  cudaFree(d_pdXMoved);
  cudaFree(d_pdYMoved);
  cudaFree(d_bNewNbrs);
  cudaFree(d_pfSE);
  cudaFree(d_pdFx);
  cudaFree(d_pdFy);
  cudaFree(d_pdFt);
  cudaFree(d_pnCellID);
  cudaFree(d_pnPPC);
  cudaFree(d_pnCellList);
  cudaFree(d_pnAdjCells);
  cudaFree(d_pnNPP);
  cudaFree(d_pnNbrList);
#if GOLD_FUNCS == 1
  delete[] g_pdX;
  delete[] g_pdY;
  delete[] g_pdPhi;
  delete[] g_pdR;
  delete[] g_pdAx;
  delete[] g_pdAy;
  delete[] g_pdTempX;
  delete[] g_pdTempY;
  delete[] g_pdTempPhi;
  delete[] g_pdXMoved;
  delete[] g_pdYMoved;
  delete[] g_pnInitID;
  delete[] g_pnMemID;
  delete[] g_pfSE;
  delete[] g_pdFx;
  delete[] g_pdFy;
  delete[] g_pdFt;
  delete[] g_pnCellID;
  delete[] g_pnPPC;
  delete[] g_pnCellList;
  delete[] g_pnAdjCells;
  delete[] g_pnNPP;
  delete[] g_pnNbrList;
#endif 
}

// Display various info about the configuration which has been calculated
// Mostly used to make sure things are working right
void Cross_Box::display(bool bParticles, bool bCells, bool bNeighbors, bool bStress)
{
  if (bParticles)
    {
      cudaMemcpyAsync(h_pdX, d_pdX, sizeof(double)*m_nCross, cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdY, d_pdY, sizeof(double)*m_nCross, cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdPhi, d_pdPhi, sizeof(double)*m_nCross, cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdR, d_pdR, sizeof(double)*m_nCross, cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdAx, d_pdAx, sizeof(double)*m_nCross, cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdAy, d_pdAy, sizeof(double)*m_nCross, cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pnMemID, d_pnMemID, sizeof(int)*m_nCross, cudaMemcpyDeviceToHost);
      cudaThreadSynchronize();
      checkCudaError("Display: copying particle data to host");
      
      cout << endl << "Box dimension: " << m_dL << endl;;
      for (int p = 0; p < m_nCross; p++)
	{
	  int m = h_pnMemID[p];
	  cout << "Particle " << p << " (" << m  << "): (" << h_pdX[m]
	       << ", " << h_pdY[m] << ", " << h_pdPhi[m] << ") R = " << h_pdR[m]
	       << " Ax = " << h_pdAx[m] << " Ay = " << h_pdAy[m] << endl;
	}
    }
  if (bCells)
    {
      cudaMemcpy(h_pnPPC, d_pnPPC, sizeof(int)*m_nCells, cudaMemcpyDeviceToHost); 
      cudaMemcpy(h_pnCellList, d_pnCellList, sizeof(int)*m_nCells*m_nMaxPPC, cudaMemcpyDeviceToHost);
      checkCudaError("Display: copying cell data to host");

      cout << endl;
      int nTotal = 0;
      int nMaxPPC = 0;
      for (int c = 0; c < m_nCells; c++)
	{
	  nTotal += h_pnPPC[c];
	  nMaxPPC = max(nMaxPPC, h_pnPPC[c]);
	  cout << "Cell " << c << ": " << h_pnPPC[c] << " particles\n";
	  for (int p = 0; p < h_pnPPC[c]; p++)
	    {
	      cout << h_pnCellList[c*m_nMaxPPC + p] << " ";
	    }
	  cout << endl;
	}
      cout << "Total particles in cells: " << nTotal << endl;
      cout << "Maximum particles in any cell: " << nMaxPPC << endl;
    }
  if (bNeighbors)
    {
      cudaMemcpy(h_pnNPP, d_pnNPP, sizeof(int)*m_nCross, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_pnNbrList, d_pnNbrList, sizeof(int)*m_nCross*m_nMaxNbrs, cudaMemcpyDeviceToHost);
      checkCudaError("Display: copying neighbor data to host");

      cout << endl;
      int nMaxNPP = 0;
      for (int p = 0; p < m_nCross; p++)
	{
	  nMaxNPP = max(nMaxNPP, h_pnNPP[p]);
	  cout << "Particle " << p << ": " << h_pnNPP[p] << " neighbors\n";
	  for (int n = 0; n < h_pnNPP[p]; n++)
	    {
	      cout << h_pnNbrList[n*m_nCross + p] << " ";
	    }
	  cout << endl;
	}
      cout << "Maximum neighbors of any particle: " << nMaxNPP << endl;
    }
  if (bStress)
    {
      cudaMemcpyAsync(h_pfSE, d_pfSE, 4*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_pdFx, d_pdFx, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_pdFy, d_pdFy, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_pdFt, d_pdFt, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
      cudaThreadSynchronize();
      m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
      cout << endl;
      for (int p = 0; p < m_nCross; p++)
	{
	  cout << "Particle " << p << ":  (" << h_pdFx[p] << ", " 
	       << h_pdFy[p] << ", " << h_pdFt[p] << ")\n";
	}
      cout << endl << "Energy: " << *m_pfEnergy << endl;
      cout << "Pxx: " << *m_pfPxx << endl;
      cout << "Pyy: " << *m_pfPyy << endl;
      cout << "Total P: " << m_fP << endl;
      cout << "Pxy: " << *m_pfPxy << endl;
    }
}

bool Cross_Box::check_for_contacts(int nIndex)
{
  double dS = 2 * ( m_dAMax + m_dRMax );

  double dX = h_pdX[nIndex];
  double dY = h_pdY[nIndex];
  for (int p = 0; p < nIndex; p++) {
    //cout << "Checking: " << nIndex << " vs " << p << endl;
    double dXj = h_pdX[p];
    double dYj = h_pdY[p];

    double dDelX = dX - dXj;
    double dDelY = dY - dYj;
    dDelX += m_dL * ((dDelX < -0.5*m_dL) - (dDelX > 0.5*m_dL));
    dDelY += m_dL * ((dDelY < -0.5*m_dL) - (dDelY > 0.5*m_dL));
    dDelX += m_dGamma * dDelY;
    double dDelRSqr = dDelX * dDelX + dDelY * dDelY;
    if (dDelRSqr < dS*dS) {
    	double dDeltaX = dX - dXj;
    	double dDeltaY = dY - dYj;
    	double dSigma = h_pdR[nIndex] + h_pdR[p];

    	// Make sure we take the closest distance considering boundary conditions
    	dDeltaX += m_dL * ((dDeltaX < -0.5*m_dL) - (dDeltaX > 0.5*m_dL));
    	dDeltaY += m_dL * ((dDeltaY < -0.5*m_dL) - (dDeltaY > 0.5*m_dL));
    	// Transform from shear coordinates to lab coordinates
    	dDeltaX += m_dGamma * dDeltaX;

    	double dPhiAs[2] = {h_pdPhi[nIndex], h_pdPhi[nIndex] + 0.5*D_PI};
    	double dAs[2] = {h_pdAx[nIndex], h_pdAy[nIndex]};
    	double dPhiBs[2] = {h_pdPhi[p], h_pdPhi[p] + 0.5*D_PI};
    	double dBs[2] = {h_pdAx[p], h_pdAy[p]};

    	for (int i = 0; i < 2; i++) {
    		for (int j = 0; j < 2; j++) {
    			double nxA = dAs[i] * cos(dPhiAs[i]);
    			double nyA = dAs[i] * sin(dPhiAs[i]);
    			double nxB = dBs[j] * cos(dPhiBs[j]);
    			double nyB = dBs[j] * sin(dPhiBs[j]);
	
    			double a = dAs[i] * dAs[i];
    			double b = -(nxA * nxB + nyA * nyB);
    			double c = dBs[j] * dBs[j];
    			double d = nxA * dDeltaX + nyA * dDeltaY;
    			double e = -nxB * dDeltaX - nyB * dDeltaY;
    			double delta = a * c - b * b;
	
    			double t = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
    			double s = -(b*t+d)/a;
    			double sarg = fabs(s);
    			s = fmin( fmax(s,-1.), 1. );
    			if (sarg > 1)
    				t = fmin( fmax( -(b*s+e)/a, -1.), 1.);
	
    			// Check if they overlap and calculate forces
    			double dDx = dDeltaX + s*nxA - t*nxB;
    			double dDy = dDeltaY + s*nyA - t*nyB;
    			double dDSqr = dDx * dDx + dDy * dDy;
    			if (dDSqr < dSigma*dSigma || dDSqr != dDSqr)
    				return 1;
    		}
    	}
    }
  }

  return 0;
}

bool Cross_Box::check_for_intersection(int nIndex, double dEpsilon)
{
	double dS = 2 * ( m_dAMax + m_dRMax );
	double dX = h_pdX[nIndex];
	double dY = h_pdY[nIndex];
	for (int p = 0; p < nIndex; p++) {
		//cout << "Checking: " << nIndex << " vs " << p << endl;
		double dXj = h_pdX[p];
		double dYj = h_pdY[p];
		double dR = h_pdR[nIndex];
		double dPhiAs[2] = {h_pdPhi[nIndex], h_pdPhi[nIndex] + 0.5*D_PI};
		double dAs[2] = {h_pdAx[nIndex], h_pdAy[nIndex]};
    
		double dDeltaX = dX - dXj;
		double dDeltaY = dY - dYj;
		double dSigma = dR + h_pdR[p];
    	double dPhiBs[2] = {h_pdPhi[p], h_pdPhi[p] + 0.5*D_PI};
    	double dBs[2] = {h_pdAx[p], h_pdAy[p]};

    	// Make sure we take the closest distance considering boundary conditions
    	dDeltaX += m_dL * ((dDeltaX < -0.5*m_dL) - (dDeltaX > 0.5*m_dL));
    	dDeltaY += m_dL * ((dDeltaY < -0.5*m_dL) - (dDeltaY > 0.5*m_dL));
    	// Transform from shear coordinates to lab coordinates
    	dDeltaX += m_dGamma * dDeltaX;

    	if (dDeltaX*dDeltaX + dDeltaY*dDeltaY < dS*dS) {
    		for (int i = 0; i < 2; i++) {
    			for (int j = 0; j < 2; j++) {
    				double nxA = dAs[i] * cos(dPhiAs[i]);
    				double nyA = dAs[i] * sin(dPhiAs[i]);
    				double nxB = dBs[j] * cos(dPhiBs[j]);
    				double nyB = dBs[j] * sin(dPhiBs[j]);
    
    				double a = dAs[i] * dAs[i];
    				double b = -(nxA * nxB + nyA * nyB);
    				double c = dBs[j] * dBs[j];
    				double d = nxA * dDeltaX + nyA * dDeltaY;
    				double e = -nxB * dDeltaX - nyB * dDeltaY;
    				double delta = a * c - b * b;

    				double t = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
    				double s = -(b*t+d)/a;
    				double sarg = fabs(s);
    				s = fmin( fmax(s,-1.), 1. );
    				if (sarg > 1)
    					t = fmin( fmax( -(b*s+e)/a, -1.), 1.);

    				// Check if they overlap and calculate forces
    				double dDx = dDeltaX + s*nxA - t*nxB;
    				double dDy = dDeltaY + s*nyA - t*nyB;
    				double dDSqr = dDx * dDx + dDy * dDy;
    				if (dDSqr < dEpsilon*dSigma*dSigma || dDSqr != dDSqr)
    					return 1;
    			}
    		}
    	}
	}
  return 0;
}

void Cross_Box::place_random_0e_cross(int seed, bool bRandAngle)
{
  srand(time(0) + seed);

  for (int p = 0; p < m_nCross; p++) {
    h_pdR[p] = m_dRMax;
    h_pdAx[p] = m_dAMax;
    h_pdAy[p] = 0.5*m_dAMax;
    h_pnMemID[p] = p;
  }
  cudaMemcpy(d_pdR, h_pdR, sizeof(double)*m_nCross, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdAx, h_pdAx, sizeof(double)*m_nCross, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdAy, h_pdAy, sizeof(double)*m_nCross, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pnInitID, h_pnMemID, sizeof(int)*m_nCross, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pnMemID, h_pnMemID, sizeof(int)*m_nCross, cudaMemcpyHostToDevice);
  cudaThreadSynchronize();

  h_pdX[0] = m_dL * static_cast<double>(rand())/static_cast<double>(RAND_MAX);
  h_pdY[0] = m_dL * static_cast<double>(rand())/static_cast<double>(RAND_MAX);
  if (bRandAngle)
    h_pdPhi[0] = 2*D_PI * static_cast<double>(rand())/static_cast<double>(RAND_MAX);
  else
    h_pdPhi[0] = - D_PI * 0.25;

  for (int p = 1; p < m_nCross; p++) {
    bool bContact = 1;
    int nTries = 0;

    while (bContact) {
      h_pdX[p] = m_dL * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
      h_pdY[p] = m_dL * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
      if (bRandAngle)
    	  h_pdPhi[p] = 2*D_PI * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
      else
    	  h_pdPhi[p] = 0;

      bContact = check_for_contacts(p);
      nTries += 1;
    }
    cout << "Cross " << p << " placed in " << nTries << " attempts." << endl;
  }
  cudaMemcpyAsync(d_pdX, h_pdX, sizeof(double)*m_nCross, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdY, h_pdY, sizeof(double)*m_nCross, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdPhi, h_pdPhi, sizeof(double)*m_nCross, cudaMemcpyHostToDevice);
  cudaThreadSynchronize();
  cout << "Data copied to device" << endl;

}

void Cross_Box::place_random_cross(int seed, bool bRandAngle)
{
  srand(time(0) + seed);

  for (int p = 0; p < m_nCross; p++) {
    h_pdR[p] = m_dRMax;
    h_pdAx[p] = m_dAMax;
    h_pdAy[p] = 0.5*m_dAMax;
    h_pnMemID[p] = p;
  }
  cudaMemcpy(d_pdR, h_pdR, sizeof(double)*m_nCross, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdAx, h_pdAx, sizeof(double)*m_nCross, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdAy, h_pdAy, sizeof(double)*m_nCross, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pnInitID, h_pnMemID, sizeof(int)*m_nCross, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pnMemID, h_pnMemID, sizeof(int)*m_nCross, cudaMemcpyHostToDevice);
  cudaThreadSynchronize();

  h_pdX[0] = m_dL * static_cast<double>(rand())/static_cast<double>(RAND_MAX);
  h_pdY[0] = m_dL * static_cast<double>(rand())/static_cast<double>(RAND_MAX);
  if (bRandAngle)
    h_pdPhi[0] = D_PI * (2 * static_cast<double>(rand())/static_cast<double>(RAND_MAX) - 1);
  else
    h_pdPhi[0] = - D_PI * 0.25;

  for (int p = 1; p < m_nCross; p++) {
    bool bContact = 1;
    int nTries = 0;

    while (bContact) {
      h_pdX[p] = m_dL * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
      h_pdY[p] = m_dL * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
      if (bRandAngle)
    	  h_pdPhi[p] = D_PI * (2 * static_cast<double>(rand()) / static_cast<double>(RAND_MAX) - 1);
      else
    	  h_pdPhi[p] = 0;

      bContact = check_for_intersection(p);
      nTries += 1;
    }
    cout << "Cross " << p << " placed in " << nTries << " attempts." << endl;
  }
  cudaMemcpyAsync(d_pdX, h_pdX, sizeof(double)*m_nCross, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdY, h_pdY, sizeof(double)*m_nCross, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdPhi, h_pdPhi, sizeof(double)*m_nCross, cudaMemcpyHostToDevice);
  cudaThreadSynchronize();
  cout << "Data copied to device" << endl;

}
