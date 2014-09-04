// -*- c++ -*-
/*
* calculate_stress_energy.cu
*
*
*/

#include <cuda.h>
#include "cross_box.h"
#include "cudaErr.h"
#include <math.h>
#include <sm_20_atomic_functions.h>
#include <stdio.h>


using namespace std;


////////////////////////////////////////////////////////
// Calculates energy, stress tensor, forces:
//  Returns returns energy, pxx, pyy and pxy to pn
//  Returns 
//
// The neighbor list pnNbrList has a list of possible contacts
//  for each particle and is found in find_neighbors.cu
//
/////////////////////////////////////////////////////////
template<Potential ePot>
__global__ void calc_se(int nCross, int *pnBlockNNbrs, int *pnBlockList, int *pnNPP, 
			int *pnNbrList, double dL, double dGamma, double *pdGlobX, double *pdGlobY, 
			double *pdGlobR, double *pdFx, double *pdFy, float *pfSE)
{
  // Declare shared memory pointer, the size is determined at the kernel launch
  extern __shared__ double sData[];
  int thid = threadIdx.x;
  int blid = blockIdx.x;
  int blsz = blockDim.x;
  int nPID = thid + blid * blsz;
  int nThreads = blsz * gridDim.x;
  int offset = blsz + 8; // +8 helps to avoid bank conflicts (I think)
  for (int i = 0; i < 4; i++)
    sData[thid + i*offset] = 0.0;
  double *pdR = &sData[4*offset];
  __syncthreads();  // synchronizes every thread in the block before going on

  while (nPID < nCross)
    {
      double dFx = 0.0;
      double dFy = 0.0;
      
      double dX = pdGlobX[nPID];
      double dY = pdGlobY[nPID];
      double dR = pdGlobR[nPID];

      int nBlockNbrs = pnBlockNNbrs[blid];
      double *pdX = &pdR[nBlockNbrs];
      double *pdY = &pdX[nBlockNbrs];
      for (int t = thid; t < nBlockNbrs; t++) {
	int nGlobID = pnBlockList[8*blid*blsz + t];
	pdR[t] = pdGlobR[nGlobID];
	pdX[t] = pdGlobX[nGlobID];
	pdY[t] = pdGlobY[nGlobID];
      }
      __syncthreads();

      int nNbrs = pnNPP[nPID];
      for (int p = 0; p < nNbrs; p++)
	{
	  int nAdjPID = pnNbrList[nPID + p * nCross];
	  
	  double dDeltaX = dX - pdX[nAdjPID];
	  double dDeltaY = dY - pdY[nAdjPID];
	  double dSigma = dR + pdR[nAdjPID];
	  // Make sure we take the closest distance considering boundary conditions
	  dDeltaX += dL * ((dDeltaX < -0.5*dL) - (dDeltaX > 0.5*dL));
	  dDeltaY += dL * ((dDeltaY < -0.5*dL) - (dDeltaY > 0.5*dL));
	  // Transform from shear coordinates to lab coordinates
	  dDeltaX += dGamma * dDeltaY;
	  
	  // Check if they overlap
	  double dRSqr = dDeltaX*dDeltaX + dDeltaY*dDeltaY;
	  if (dRSqr < dSigma*dSigma)
	    {
	      double dDelR = sqrt(dRSqr);
	      double dDVij;
	      double dAlpha;
	      if (ePot == HARMONIC)
		{
		  dDVij = (1.0 - dDelR / dSigma) / dSigma;
		  dAlpha = 2.0;
		}
	      else if (ePot == HERTZIAN)
		{
		  dDVij = (1.0 - dDelR / dSigma) * sqrt(1.0 - dDelR / dSigma) / dSigma;
		  dAlpha = 2.5;
		}
	      double dPfx = dDeltaX * dDVij / dDelR;
	      double dPfy = dDeltaY * dDVij / dDelR;
	      dFx += dPfx;
	      dFy += dPfy;
	      if (nAdjPID > nPID)
		{
		  sData[thid] += dDVij * dSigma * (1.0 - dDelR / dSigma) / (dAlpha * dL * dL);
		  sData[thid + offset] += dPfx * dDeltaX / (dL * dL);
		  sData[thid + 2*offset] += dPfy * dDeltaY / (dL * dL);
		  sData[thid + 3*offset] += dPfx * dDeltaY / (dL * dL);
		} 
	    }
	}
      pdFx[nPID] = dFx;
      pdFy[nPID] = dFy;
      
      nPID += nThreads;
      blid += gridDim.x;
    }
  __syncthreads();
  
  // Now we do a parallel reduction sum to find the total number of contacts
  int stride = blockDim.x / 2;  // stride is 1/2 block size, all threads perform two adds
  int base = thid % stride + offset * (thid / stride);
  sData[base] += sData[base + stride];
  base += 2*offset;
  sData[base] += sData[base + stride];
  stride /= 2; // stride is 1/4 block size, all threads perform 1 add
  __syncthreads();
  base = thid % stride + offset * (thid / stride);
  sData[base] += sData[base + stride];
  stride /= 2;
  __syncthreads();
  while (stride > 8)
    {
      if (thid < 4 * stride)
	{
	  base = thid % stride + offset * (thid / stride);
	  sData[base] += sData[base + stride];
	}
      stride /= 2;  
      __syncthreads();
    }
  if (thid < 32) //unroll end of loop
    {
      base = thid % 8 + offset * (thid / 8);
      sData[base] += sData[base + 8];
      if (thid < 16)
	{
	  base = thid % 4 + offset * (thid / 4);
	  sData[base] += sData[base + 4];
	  if (thid < 8)
	    {
	      base = thid % 2 + offset * (thid / 2);
	      sData[base] += sData[base + 2];
	      if (thid < 4)
		{
		  sData[thid * offset] += sData[thid * offset + 1];
		  float tot = atomicAdd(pfSE+thid, (float)sData[thid*offset]);	    
		}
	    }
	}
    }  
}


void Cross_Box::calculate_stress_energy()
{
  cudaMemset((void*) d_pfSE, 0, 4*sizeof(float));
  
  //dim3 grid(m_nGridSize);
  //dim3 block(m_nBlockSize);
  //size_t smem = m_nSM_CalcSE;
  //printf("Configuration: %d x %d x %d\n", m_nGridSize, m_nBlockSize, m_nSM_CalcSE);

  switch (m_ePotential)
    {
    case HARMONIC:
      calc_se <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
	(m_nCross, d_pnBlockNNbrs, d_pnBlockList, d_pnNPP, d_pnNbrList, 
	 m_dL, m_dGamma, d_pdX, d_pdY, d_pdR, d_pdFx, d_pdFy, d_pfSE);
      break;
    case HERTZIAN:
      calc_se <HERTZIAN> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
	(m_nCross, d_pnBlockNNbrs, d_pnBlockList, d_pnNPP, d_pnNbrList, 
	 m_dL, m_dGamma, d_pdX, d_pdY, d_pdR, d_pdFx, d_pdFy, d_pfSE);
    }
  cudaThreadSynchronize();
  checkCudaError("Calculating stresses and energy");
}















////////////////////////////////////////////////////////////////////////////////////
#if GOLD_FUNCS == 1
void calc_se_gold(Potential ePot, int nGridDim, int nBlockDim, int sMemSize, 
		  int nCross, int *pnNPP, int *pnNbrList, double dL,
		  double dGamma, double *pdX, double *pdY, double *pdR,
		  double *pdFx, double *pdFy, float *pfSE)
{
for (int b = 0; b < nGridDim; b++)
  {
    printf("Entering loop, block %d\n", b);
for (int thid = 0; thid < nBlockDim; thid++)
  {
    printf("Entering loop, thread %d\n", thid);
  // Declare shared memory pointer, the size is determined at the kernel launch
    double *sData = new double[sMemSize / sizeof(double)];
  int nPID = thid + b * nBlockDim;
  int nThreads = nBlockDim * nGridDim;
  int offset = nBlockDim + 8; // +8 helps to avoid bank conflicts (I think)
  for (int i = 0; i < 4; i++)
    sData[thid + i*offset] = 0.0;
  double dFx = 0.0;
  double dFy = 0.0;

  while (nPID < nCross)
    {
      double dX = pdX[nPID];
      double dY = pdY[nPID];
      double dR = pdR[nPID];
      
      int nNbrs = pnNPP[nPID];
      for (int p = 0; p < nNbrs; p++)
	{
	  int nAdjPID = pnNbrList[nPID + p * nCross];
	  
	  double dDeltaX = dX - pdX[nAdjPID];
	  double dDeltaY = dY - pdY[nAdjPID];
	  double dSigma = dR + pdR[nAdjPID];
	  // Make sure we take the closest distance considering boundary conditions
	  dDeltaX += dL * ((dDeltaX < -0.5*dL) - (dDeltaX > 0.5*dL));
	  dDeltaY += dL * ((dDeltaY < -0.5*dL) - (dDeltaY > 0.5*dL));
	  // Transform from shear coordinates to lab coordinates
	  dDeltaX += dGamma * dDeltaY;
	  
	  // Check if they overlap
	  double dRSqr = dDeltaX*dDeltaX + dDeltaY*dDeltaY;
	  if (dRSqr < dSigma*dSigma)
	    {
	      double dDelR = sqrt(dRSqr);
	      double dDVij;
	      double dAlpha;
	      if (ePot == HARMONIC)
		{
		  dDVij = (1.0 - dDelR / dSigma) / dSigma;
		  dAlpha = 2.0;
		}
	      else if (ePot == HERTZIAN)
		{
		  dDVij = (1.0 - dDelR / dSigma) * sqrt(1.0 - dDelR / dSigma) / dSigma;
		  dAlpha = 2.5;
		}
	      double dPfx = dDeltaX * dDVij / dDelR;
	      double dPfy = dDeltaY * dDVij / dDelR;
	      dFx += dPfx;
	      dFy += dPfy;
	      if (nAdjPID > nPID)
		{
		  sData[thid] += dDVij * dSigma * (1.0 - dDelR / dSigma) / (dAlpha * dL * dL);
		  sData[thid + offset] += dPfx * dDeltaX / (dL * dL);
		  sData[thid + 2*offset] += dPfy * dDeltaY / (dL * dL);
		  sData[thid + 3*offset] += dPfx * dDeltaY / (dL * dL);
		} 
	    }
	}
      pdFx[nPID] = dFx;
      pdFy[nPID] = dFy;
      dFx = 0.0;
      dFy = 0.0;
      
      nPID += nThreads;
    }
  
  // Now we do a parallel reduction sum to find the total number of contacts
  for (int s = 0; s < 4; s++)
    pfSE[s] += sData[thid + s*offset];
	 
  }
  }
}

void Cross_Box::calculate_stress_energy_gold()
{
  printf("Calculating streeses and energy");
  cudaMemcpy(g_pnNPP, d_pnNPP, sizeof(int)*m_nCross, cudaMemcpyDeviceToHost);
  cudaMemcpy(g_pnNbrList, d_pnNbrList, sizeof(int)*m_nCross*m_nMaxNbrs, cudaMemcpyDeviceToHost);
  cudaMemcpy(g_pdX, d_pdX, sizeof(double)*m_nCross, cudaMemcpyDeviceToHost);
  cudaMemcpy(g_pdY, d_pdY, sizeof(double)*m_nCross, cudaMemcpyDeviceToHost);
  cudaMemcpy(g_pdR, d_pdR, sizeof(double)*m_nCross, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 4; i++)
    g_pfSE[i] = 0.0;

  switch (m_ePotential)
    {
    case HARMONIC:
      calc_se_gold (HARMONIC, m_nGridSize, m_nBlockSize, m_nSM_CalcSE,
			  m_nCross, g_pnNPP, g_pnNbrList, m_dL, m_dGamma, 
			  g_pdX, g_pdY, g_pdR, g_pdFx, g_pdFy, g_pfSE);
      break;
    case HERTZIAN:
      calc_se_gold (HERTZIAN, m_nGridSize, m_nBlockSize, m_nSM_CalcSE,
			  m_nCross, g_pnNPP, g_pnNbrList, m_dL, m_dGamma, 
			  g_pdX, g_pdY, g_pdR, g_pdFx, g_pdFy, g_pfSE);
    }

  for (int p = 0; p < m_nCross; p++)
    {
      printf("Particle %d:  (%g, %g)\n", p, g_pdFx[p], g_pdFy[p]);
    }
  printf("Energy: %g\n", g_pfSE[0]);
  printf("Pxx: %g\n", g_pfSE[1]);
  printf("Pyy: %g\n", g_pfSE[2]);
  printf("P: %g\n", g_pfSE[1] + g_pfSE[2]);
  printf("Pxy: %g\n", g_pfSE[3]);
}

#endif
