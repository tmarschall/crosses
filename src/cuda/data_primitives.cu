// -*- c++ -*-
/*
*  data_primitives.cu
*
*  kernels for primitive data operations used in several functions
*
*
*/

#include "data_primitives.h"
#include <cuda.h>
#include <sm_20_atomic_functions.h>
#include "cudaErr.h"
#include <stdio.h>


__global__ void block_scan(int nLength, int nBlockSize, int *pnIn, int *pnOut, int nBlocks, int *pnSums)
{
  extern __shared__ int sMem[];
  int thid = threadIdx.x;
  int blid = blockIdx.x;

  while (blid < nBlocks)
    {
      int offset = 1;

      int globalId = thid + blid * nBlockSize;
      if (globalId < nLength)
	{
	  sMem[thid] = pnIn[globalId];
	  globalId += nBlockSize / 2;
	  if (globalId < nLength)
	    sMem[thid + nBlockSize / 2] = pnIn[globalId];
	  else
	    sMem[thid + nBlockSize / 2] = 0;
	}
      else
	{
	  sMem[thid] = 0;
	  sMem[thid + nBlockSize / 2] = 0;
	}

      for (int d = nBlockSize / 2; d > 0; d /= 2)
	{
	  __syncthreads();
	  if (thid < d)
	    {
	      int ai = offset*(2*thid + 1) - 1;
	      int bi = offset*(2*thid + 2) - 1;
	      sMem[bi] += sMem[ai];
	    }
	  offset *= 2;
	}
      
      __syncthreads();
      int nSID = thid + blid + 1;
      if (nSID < nBlocks)
	int nPartSum = atomicAdd(pnSums+nSID, sMem[nBlockSize - 1]);
      __syncthreads();
      if (thid == 0)
	sMem[nBlockSize - 1] = 0;

      for (int d = 1; d < nBlockSize; d *= 2)
	{
	  offset /= 2;
	  __syncthreads();
	  if (thid < d)
	    {
	      int ai = offset*(2*thid + 1) - 1;
	      int bi = offset*(2*thid + 2) - 1;

	      int temp = sMem[ai];
	      sMem[ai] = sMem[bi];
	      sMem[bi] += temp;
	    }
	}
      __syncthreads();
      
      globalId = blid * nBlockSize + thid;
      pnOut[globalId] = sMem[thid];
      pnOut[globalId + nBlockSize / 2] = sMem[thid + nBlockSize / 2];

      blid += gridDim.x;
    }
}

__global__ void finish_scan(int nLength, int nBlockSize, int *pnOut, int nBlocks, int *pnSums)
{
  int blid = blockIdx.x;
  
  while (blid < nBlocks)
    {
      int globalId = threadIdx.x + blid * nBlockSize;
      if (globalId < nLength)
	{
	  pnOut[globalId] += pnSums[blid];
	  globalId += nBlockSize / 2;
	  if (globalId < nLength)
	    pnOut[globalId] += pnSums[blid];
	}
      
      blid += gridDim.x;
    }
}

void exclusive_scan(int *d_pnIn, int *d_pnOut, int nSize)
{
  int nBlockSize = 512;
  int nBlocks = nSize / nBlockSize + ((nSize % nBlockSize != 0) ? 1 : 0);
  int *d_pnSums;
  cudaMalloc((void **) &d_pnSums, sizeof(int) * nBlocks);
  cudaMemset((void *) d_pnSums, 0, sizeof(int) * nBlocks);
  
  int nGridDim = ((nBlocks <= 16) ? nBlocks : 16);
  int nBlockDim = nBlockSize / 2;
  int sMemSize = nBlockSize * sizeof(int);
  //printf("Scanning %d x %d with %d bytes smem\n", nGridDim, nBlockDim, sMemSize);

  block_scan <<<nGridDim, nBlockDim, sMemSize>>> (nSize, nBlockSize, d_pnIn, d_pnOut, nBlocks, d_pnSums);
  cudaThreadSynchronize();
  checkCudaError("Performing exclusive scan on blocks");

  if (nBlocks > 1)
    {
      //int *h_pnSums = (int*) malloc(sizeof(int) * nBlocks);
      finish_scan <<<nGridDim, nBlockDim>>> (nSize, nBlockSize, d_pnOut, nBlocks, d_pnSums);
      cudaThreadSynchronize();
      checkCudaError("Adding block sums to block scans");

      //cudaMemcpy(h_pnSums, d_pnSums, sizeof(int) * nBlocks)
    }

  cudaFree(d_pnSums);
}


__global__ void ordered_arr(int *pnArr, int nSize)
{
  int thid = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;
  
  while (thid < nSize)
    {
      pnArr[thid] = thid;
      thid += nThreads;
    }
}


void ordered_array(int *pnArray, int nSize, int gridSize, int blockSize)
{
  int nBlockSize;
  int nGridSize;
  if (gridSize == 0 || blockSize == 0)
    {
      if (nSize > 16 * 128)
	nBlockSize = 256;
      else
	nBlockSize = 128;
      nGridSize = nSize / nBlockSize + ((nSize % nBlockSize != 0) ? 1 : 0);
    }
  else
    {
      nGridSize = gridSize;
      nBlockSize = blockSize;
    }

  ordered_arr <<<nGridSize, nBlockSize>>> (pnArray, nSize);
  cudaThreadSynchronize();
  checkCudaError("Creating ordered array");
}
