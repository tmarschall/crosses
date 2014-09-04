// -*- c++ -*-
/*
* find_neighbors.cu
*
*  
*
*
*/

#include <cuda.h>
#include "cross_box.h"
#include "cudaErr.h"
#include "data_primitives.h"
#include <sm_20_atomic_functions.h>
#include <math.h>

using namespace std;


///////////////////////////////////////////////////////////////
// Find the Cell ID for each particle:
//  The list of cell IDs for each particle is returned to pnCellID
//  A list of which particles are in each cell is returned to pnCellList
//
// *NOTE* if there are more than nMaxPPC particles in a given cell,
//  not all of these particles will get added to the cell list
///////////////////////////////////////////////////////////////
__global__ void find_cells(int nCross, int nMaxPPC, double dCellW, double dCellH,
			   int nCellCols, double dL, double *pdX, double *pdY, 
			   int *pnCellID, int *pnPPC, int *pnCellList)
{
  // Assign each thread a unique ID accross all thread-blocks, this is its particle ID
  int nPID = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;

  while (nPID < nCross) {
    double dX = pdX[nPID];
    double dY = pdY[nPID];
    
    // I often allow the stored coordinates to drift slightly outside the box limits
    //  until 
    if (dY > dL)
      {
	dY -= dL;
	pdY[nPID] = dY;
      }
    else if (dY < 0)
      {
	dY += dL;
	pdY[nPID] = dY;
      }
    if (dX > dL)
      {
	dX -= dL;
	pdX[nPID] = dX;
      }
    else if (dX < 0)
      {
	dX += dL;
	pdX[nPID] = dX;
      }

    //find the cell ID, add a particle to that cell 
    int nCol = (int)(dX / dCellW);
    int nRow = (int)(dY / dCellH); 
    int nCellID = nCol + nRow * nCellCols;
    pnCellID[nPID] = nCellID;

    // Add 1 particle to a cell safely (only allows one thread to access the memory
    //  address at a time). nPPC is the original value, not the result of addition 
    int nPPC = atomicAdd(pnPPC + nCellID, 1);
    
    // only add particle to cell if there is not already the maximum number in cell
    if (nPPC < nMaxPPC)
      pnCellList[nCellID * nMaxPPC + nPPC] = nPID;
    else
      nPPC = atomicAdd(pnPPC + nCellID, -1);

    nPID += nThreads;
  }
}


////////////////////////////////////////////////////////////////
// Here a list of possible contacts is created for each particle
//  The list of neighbors is returned to pnNbrList
//
// This is one function that I may target for optimization in
//  the future because I know it is slowed down by branch divergence
////////////////////////////////////////////////////////////////
__global__ void find_nbrs(int nCross, int nMaxPPC, int *pnCellID, int *pnPPC, 
			  int *pnCellList, int *pnAdjCells, int nMaxNbrs, int *pnNPP, 
			  int *pnNbrList, double *pdX, double *pdY, double *pdR, 
			  double dEpsilon, double dL, double dGamma)
{
  extern __shared__ int sData[];
  int thid = threadIdx.x;
  int blsz = blockDim.x;
  int blid = blockIdx.x;
  int nPID = thid + blid * blsz;
  int nThreads = gridDim.x * blsz;

  while (nPID < nCross)
    {
      double dX = pdX[nPID];
      double dY = pdY[nPID];
      double dR = pdR[nPID];
      int nNbrs = 0;

      // Particles in adjacent cells are added if they are close enough to 
      //  interact without each moving by more than dEpsilon/2
      int nCellID = pnCellID[nPID];
      int nP = pnPPC[nCellID];
      for (int p = 0; p < nP; p++)
	{
	  int nAdjPID = pnCellList[nCellID*nMaxPPC + p];
	  if (nAdjPID != nPID)
	    {
	      double dSigma = dR + pdR[nAdjPID] + dEpsilon;
	      double dDeltaY = dY - pdY[nAdjPID];
	      dDeltaY += dL * ((dDeltaY < -0.5 * dL) - (dDeltaY > 0.5 * dL));
	      
	      if (fabs(dDeltaY) < dSigma)
		{
		  double dDeltaX = dX - pdX[nAdjPID];
		  dDeltaX += dL * ((dDeltaX < -0.5 * dL) - (dDeltaX > 0.5 * dL));
		  double dDeltaRx = dDeltaX + dGamma * dDeltaY;
		  double dDeltaRx2 = dDeltaX + 0.5 * dDeltaY;
		  if (fabs(dDeltaRx) < dSigma || fabs(dDeltaRx2) < dSigma)
		    {
		      // This indexing makes global memory accesses more coalesced
		      if (nNbrs < nMaxNbrs)
			{
			  //pnNbrList[nCross * nNbrs + nPID] = nAdjPID;
			  sData[blsz * nNbrs + thid] = nAdjPID;
			  nNbrs += 1;
			}
		    }
		}
	    }
	}

      for (int nc = 0; nc < 8; nc++)
	{
	  int nAdjCID = pnAdjCells[8 * nCellID + nc];
	  nP = pnPPC[nAdjCID];
	  for (int p = 0; p < nP; p++)
	    {
	      int nAdjPID = pnCellList[nAdjCID*nMaxPPC + p];
	      
	      // The maximum distance at which two particles could contact
	      //  plus a little bit of moving room - dEpsilon 
	      double dSigma = dR + pdR[nAdjPID] + dEpsilon;
	      double dDeltaY = dY - pdY[nAdjPID];

	      // Make sure were finding the closest separation
	      dDeltaY += dL * ((dDeltaY < -0.5 * dL) - (dDeltaY > 0.5 * dL));

	      if (fabs(dDeltaY) < dSigma)
		{
		  double dDeltaX = dX - pdX[nAdjPID];
		  dDeltaX += dL * ((dDeltaX < -0.5 * dL) - (dDeltaX > 0.5 * dL));

		  // Go to unsheared coordinates
		  double dDeltaRx = dDeltaX + dGamma * dDeltaY;
		  // Also look at distance when the strain parameter is at its max (0.5)
		  double dDeltaRx2 = dDeltaX + 0.5 * dDeltaY;
		  if (fabs(dDeltaRx) < dSigma || fabs(dDeltaRx2) < dSigma)
		    {
		      if (nNbrs < nMaxNbrs)
			{
			  //pnNbrList[nCross * nNbrs + nPID] = nAdjPID;
			  sData[blsz * nNbrs + thid] = nAdjPID;
			  nNbrs += 1;
			}
		    }
		}
	    }
	  
	}
      pnNPP[nPID] = nNbrs;
      for (int n = 0; n < nNbrs; n++) {
	pnNbrList[nCross * n + nPID] = sData[blsz * n + thid];
      }

      nPID += nThreads;
    }
}


__global__ void get_nbr_blocks(int nCross, int *pnNNbrs, int *pnNbrList, int *pnBlockNbrs, int *pnBlockList)
{
  extern __shared__ int sData[];
  int thid = threadIdx.x;
  int blid = blockIdx.x;
  int blsz = blockDim.x;
  int nPID = thid + blid * blsz;
  
  while (nPID < nCross) {
    sData[thid] = pnNNbrs[nPID];
    int nNbrs = sData[thid];
    __syncthreads();

    int offset = 1;
    for (int d = blsz / 2; d > 0; d /= 2) {
      if (thid < d) {
	int ai = offset*(2*thid + 1) - 1;
	int bi = offset*(2*thid + 2) - 1;
	sData[bi] += sData[ai];
      }
      offset *= 2;
      __syncthreads();
    }
    
    int nTotNbrs = sData[blsz - 1];
    int nSortMax = 256;
    int s = nTotNbrs / 256;
    while (s > 0) {
      nSortMax *= 2;
      s /= 2;
    }
    __syncthreads();
    if (thid == 0) {
      sData[blsz - 1] = 0;
      //pnBlockNbrs[blid] = nTotNbrs;
    }
    
    for (int d = 1; d < blsz; d *= 2) {
      __syncthreads();
      offset /= 2;
      if (thid < d) {
	int ai = offset*(2*thid + 1) - 1;
	int bi = offset*(2*thid + 2) - 1;
	int temp = sData[ai];
	sData[ai] = sData[bi];
	sData[bi] += temp;
      }
    }
    
    __syncthreads();
    //int *pnList = &sData[blsz];
    int nID = sData[thid];
    for (int n = 0; n < nNbrs; n++) {
	sData[blsz + nID + n] = pnNbrList[n * nCross + nPID];
    }
    for (int t = nTotNbrs + thid; t < nSortMax; t += blsz) {
      sData[blsz + t] = nCross;
    }
    
    __syncthreads();
    //for (int t = thid; t < 2048; t += blsz){
    //pnBlockList[2048*blid + t] = pnList[t];
    //}
    
    for (int oblock = 1; oblock < nSortMax; oblock *= 2) {
      int t = thid;
      int obid = t / oblock;
      while (obid < nSortMax / (2 * oblock)) {
	for (int iblock = oblock; iblock > 0; iblock /= 2) {
	  int s = (t % oblock);
	  int ibid = s / iblock;
	  while (ibid < oblock / iblock) {
	    int ai, bi;
	    if (obid % 2) {
	      bi = blsz + 2 * (oblock * obid + iblock * ibid) + s % iblock;
	      ai = bi + iblock;
	    }
	    else {
	      ai = blsz + 2 * (oblock * obid + iblock * ibid) + s % iblock;
	      bi = ai + iblock;
	    }
	    if (sData[ai] > sData[bi]) {
	      int temp = sData[ai];
	      sData[ai] = sData[bi];
	      sData[bi] = temp;
	    }
	    s += blsz;
	    ibid = s / iblock;
	  }
	  if (iblock > 32)
	    __syncthreads();
	}
	t += blsz;
	obid = t / oblock;
      }
      __syncthreads();
    }

    
    //int *pnCID = &pnList[nSortMax];
    for (int t = thid; t < nSortMax - 1; t += blsz) {
      if (sData[blsz + t] == sData[blsz + t + 1]) 
	sData[blsz + nSortMax + t] = 0;
      else
	sData[blsz + nSortMax + t] = 1;
    }
    if (thid == 0)
      sData[blsz + 2 * nSortMax - 1] = 0;
    __syncthreads();

    offset = 1;
    for (int d = nSortMax / 2; d > 0; d /= 2) {
      int t = thid;
      while (t < d) {
	int ai = blsz + nSortMax + offset*(2*t + 1) - 1;
	int bi = blsz + nSortMax + offset*(2*t + 2) - 1;
	sData[bi] += sData[ai];
	t += blsz;
      }
      offset *= 2;
      __syncthreads();
    }
    
    nTotNbrs = sData[blsz + 2 * nSortMax - 1];
    __syncthreads();
    if (thid == 0) {
      sData[blsz + 2*nSortMax - 1] = 0;
      pnBlockNbrs[blid] = nTotNbrs;
    }
    
    for (int d = 1; d < nSortMax; d *= 2) {
      __syncthreads();
      int t = thid;
      offset /= 2;
      while (t < d) {
	int ai = blsz + nSortMax + offset*(2*t + 1) - 1;
	int bi = blsz + nSortMax + offset*(2*t + 2) - 1;
	int temp = sData[ai];
	sData[ai] = sData[bi];
	sData[bi] += temp;
	t += blsz;
      }
    }

    __syncthreads();
    for (int t = thid; t < nSortMax - 1; t += blsz) {
      if (sData[blsz + t] != sData[blsz + t + 1]) 
	pnBlockList[blid*8*blsz + sData[blsz + nSortMax + t]] = sData[blsz + t];
    }
    __syncthreads();
    for (int t = thid; t < nTotNbrs; t += blsz) {
      sData[t] = pnBlockList[blid*8*blsz + t];
    }
    __syncthreads();

    for (int n = 0; n < nNbrs; n++) {
      int oldID = pnNbrList[nCross * n + nPID];
      int newID = nTotNbrs / 2;
      int diff = max(-newID, min(oldID - sData[newID], nTotNbrs - 1 - newID));
      int dStep = newID / 2;
      newID += diff;
      while (dStep > 0) {
	diff = max(max(-dStep, -newID), min(oldID - sData[newID], min(dStep, nTotNbrs - 1 - newID)));
	newID += diff;
	dStep /= 2;
      }
      pnNbrList[nCross*n + nPID] = newID;
    }
    
    blid += gridDim.x;
    nPID += gridDim.x * blsz;
  }
    
}

///////////////////////////////////////////////////////////////
// Finds a list of possible contacts for each particle
//
// Usually when things are moving I keep track of an Xmoved and Ymoved
//  and only call this to make a new list of neighbors if some particle
//  has moved more than (dEpsilon / 2) in some direction
///////////////////////////////////////////////////////////////
void Cross_Box::find_neighbors()
{
  // reset each byte to 0
  cudaMemset((void *) d_pnPPC, 0, sizeof(int)*m_nCells);
  cudaMemset((void *) d_pdXMoved, 0, sizeof(double)*m_nCross);
  cudaMemset((void *) d_pdYMoved, 0, sizeof(double)*m_nCross);
  cudaMemset((void *) d_bNewNbrs, 0, sizeof(int));

  find_cells <<<m_nGridSize, m_nBlockSize>>>
    (m_nCross, m_nMaxPPC, m_dCellW, m_dCellH, m_nCellCols, 
     m_dL, d_pdX, d_pdY, d_pnCellID, d_pnPPC, d_pnCellList);
  cudaThreadSynchronize();
  checkCudaError("Finding cells");

  find_nbrs <<<m_nGridSize, m_nBlockSize, m_nSM_FindCells>>>
    (m_nCross, m_nMaxPPC, d_pnCellID, d_pnPPC, d_pnCellList, d_pnAdjCells, 
     m_nMaxNbrs, d_pnNPP, d_pnNbrList, d_pdX, d_pdY, d_pdR, m_dEpsilon, m_dL, m_dGamma);
  cudaThreadSynchronize();
  checkCudaError("Finding neighbors");

  get_nbr_blocks <<<m_nGridSize, m_nBlockSize, m_nSM_GetNbrBlks>>>
    (m_nCross, d_pnNPP, d_pnNbrList, d_pnBlockNNbrs, d_pnBlockList);
  cudaThreadSynchronize();
  checkCudaError("Getting neighbor blocks");

}


////////////////////////////////////////////////////////////////////////////////////
// Sets gamma back by 1 (used when gamma > 0.5)
//  also finds the cells in the process
//
///////////////////////////////////////////////////////////////////////////////////
__global__ void set_back_coords(int nCross, int nMaxPPC, double dCellW, double dCellH,
				int nCellCols, double dL, double *pdX, double *pdY, 
				int *pnCellID, int *pnPPC, int *pnCellList)
{
  // Assign each thread a unique ID accross all thread-blocks, this is its particle ID
  int nPID = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;

  while (nPID < nCross) {
    double dX = pdX[nPID];
    double dY = pdY[nPID];
    
    // I often allow the stored coordinates to drift slightly outside the box limits
    //  until 
    if (dY > dL)
      {
	dY -= dL;
	pdY[nPID] = dY;
      }
    else if (dY < 0)
      {
	dY += dL;
	pdY[nPID] = dY;
      }
    
    // When gamma -> gamma-1, Xi -> Xi + Yi
    dX += dY;
    if (dX < 0)
      {
	dX += dL;
      }
    while (dX > dL)
      {
	dX -= dL;
      }
    pdX[nPID] = dX;


    //find the cell ID, add a particle to that cell 
    int nCol = (int)(dX / dCellW);
    int nRow = (int)(dY / dCellH); 
    int nCellID = nCol + nRow * nCellCols;
    pnCellID[nPID] = nCellID;

    // Add 1 particle to a cell safely (only allows one thread to access the memory
    //  address at a time). nPPC is the original value, not the result of addition 
    int nPPC = atomicAdd(pnPPC + nCellID, 1);
    
    // only add particle to cell if there is not already the maximum number in cell
    if (nPPC < nMaxPPC)
      pnCellList[nCellID * nMaxPPC + nPPC] = nPID;
    else
      nPPC = atomicAdd(pnPPC + nCellID, -1);

    nPID += nThreads;
  }

}

void Cross_Box::set_back_gamma()
{
  cudaMemset((void *) d_pnPPC, 0, sizeof(int)*m_nCells);
  cudaMemset((void *) d_pdXMoved, 0, sizeof(double)*m_nCross);
  cudaMemset((void *) d_pdYMoved, 0, sizeof(double)*m_nCross);
  cudaMemset((void *) d_bNewNbrs, 0, sizeof(int));

  set_back_coords <<<m_nGridSize, m_nBlockSize>>>
    (m_nCross, m_nMaxPPC, m_dCellW, m_dCellH, m_nCellCols, 
     m_dL, d_pdX, d_pdY, d_pnCellID, d_pnPPC, d_pnCellList);
  cudaThreadSynchronize();
  checkCudaError("Finding new coordinates, cells");
  m_dGamma -= 1;

  find_nbrs <<<m_nGridSize, m_nBlockSize, m_nSM_FindCells>>>
    (m_nCross, m_nMaxPPC, d_pnCellID, d_pnPPC, d_pnCellList, d_pnAdjCells, 
     m_nMaxNbrs, d_pnNPP, d_pnNbrList, d_pdX, d_pdY, d_pdR, m_dEpsilon, m_dL, m_dGamma);
  cudaThreadSynchronize();
  checkCudaError("Finding neighbors");
}


////////////////////////////////////////////////////////////////////////////
// Finds cells for all particles regardless of maximum particle per cell
//  used for reordering particles
/////////////////////////////////////////////////////////////////////////
__global__ void find_cells_nomax(int nCross, double dCellW, double dCellH,
				 int nCellCols, double dL, double *pdX, double *pdY, 
				 int *pnCellID, int *pnPPC)
{
  // Assign each thread a unique ID accross all thread-blocks, this is its particle ID
  int nPID = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;

  while (nPID < nCross) {
    double dX = pdX[nPID];
    double dY = pdY[nPID];
    
    // Particles are allowed to drift slightly outside the box limits
    //  until cells are reassigned due to a particle drift of dEpsilon/2 
    if (dY > dL) {
      dY -= dL; 
      pdY[nPID] = dY; }
    else if (dY < 0) {
      dY += dL;
      pdY[nPID] = dY; }
    if (dX > dL) {
      dX -= dL; 
      pdX[nPID] = dX; }
    else if (dX < 0) {
      dX += dL;
      pdX[nPID] = dX; }

    //find the cell ID, add a particle to that cell 
    int nCol = (int)(dX / dCellW);
    int nRow = (int)(dY / dCellH); 
    int nCellID = nCol + nRow * nCellCols;
    
    pnCellID[nPID] = nCellID;
    int nPPC = atomicAdd(pnPPC + nCellID, 1);
    
    nPID += nThreads; }
}

__global__ void reorder_part(int nCross, double *pdTempX, double *pdTempY, 
			     double *pdTempR, int *pnInitID, double *pdX, 
			     double *pdY, double *pdR, int *pnMemID, 
			     int *pnCellID, int *pnCellSID)
{
  int nPID = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;

  while (nPID < nCross) {
    double dX = pdTempX[nPID];
    double dY = pdTempY[nPID];
    double dR = pdTempR[nPID];
    int nInitID = pnInitID[nPID];

    int nCellID = pnCellID[nPID];
    int nNewID = atomicAdd(pnCellSID + nCellID, 1);
    
    pdX[nNewID] = dX;
    pdY[nNewID] = dY;
    pdR[nNewID] = dR;
    pnMemID[nNewID] = nInitID;

    nPID += nThreads; }
}

__global__ void invert_IDs(int nIDs, int *pnIn, int *pnOut)
{
  int thid = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;

  while (thid < nIDs) {
    int i = pnIn[thid];
    pnOut[i] = thid; 
    thid += nThreads; }
    
}

void Cross_Box::reorder_particles()
{
  cudaMemset((void *) d_pnPPC, 0, sizeof(int)*m_nCells);

  //find particle cell IDs and number of particles in each cell
  find_cells_nomax <<<m_nGridSize, m_nBlockSize>>>
    (m_nCross, m_dCellW, m_dCellH, m_nCellCols, 
     m_dL, d_pdX, d_pdY, d_pnCellID, d_pnPPC);
  cudaThreadSynchronize();
  checkCudaError("Reordering particles: Finding cells");

  int *d_pnCellSID;
  double *d_pdTempR;
  cudaMalloc((void **) &d_pnCellSID, sizeof(int) * m_nCells);
  cudaMalloc((void **) &d_pdTempR, sizeof(double) * m_nCross);
  cudaMemcpy(d_pdTempX, d_pdX, sizeof(double) * m_nCross, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_pdTempY, d_pdY, sizeof(double) * m_nCross, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_pdTempR, d_pdR, sizeof(double) * m_nCross, cudaMemcpyDeviceToDevice);

  exclusive_scan(d_pnPPC, d_pnCellSID, m_nCells);

  /*
  int *h_pnCellSID = (int*) malloc(m_nCells * sizeof(int));
  int *h_pnCellNPart = (int*) malloc(m_nCells * sizeof(int));
  cudaMemcpy(h_pnCellNPart, d_pnCellNPart, sizeof(int)*m_nCells, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pnCellSID, d_pnCellSID, sizeof(int)*m_nCells, cudaMemcpyDeviceToHost);
  for (int c = 0; c < m_nCells; c++)
    {
      printf("%d %d\n", h_pnCellNPart[c], h_pnCellSID[c]);
    }
  free(h_pnCellSID);
  free(h_pnCellNPart);
  */

  //reorder particles based on cell ID (first by Y direction)
  reorder_part <<<m_nGridSize, m_nBlockSize>>>
    (m_nCross, d_pdTempX, d_pdTempY, d_pdTempR, d_pnInitID, 
     d_pdX, d_pdY, d_pdR, d_pnMemID, d_pnCellID, d_pnCellSID);
  cudaThreadSynchronize();
  checkCudaError("Reordering particles: changing order");

  invert_IDs <<<m_nGridSize, m_nBlockSize>>> (m_nCross, d_pnMemID, d_pnInitID);
  cudaMemcpyAsync(h_pnMemID, d_pnMemID, m_nCross*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdR, d_pdR, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_pnCellSID);
  cudaFree(d_pdTempR);

  find_neighbors();
}


////////////////////////////////////////////////////////////////////////
// Sets the particle IDs to their order in memory
//  so the current IDs become the initial IDs
/////////////////////////////////////////////////////////////////////
void Cross_Box::reset_IDs()
{
  ordered_array(d_pnInitID, m_nCross, m_nGridSize, m_nBlockSize);
  cudaMemcpy(d_pnMemID, d_pnInitID, sizeof(int)*m_nCross, cudaMemcpyDeviceToDevice);
  cudaMemcpy(h_pnMemID, d_pnInitID, sizeof(int)*m_nCross, cudaMemcpyDeviceToHost);
}
