// -*- c++ -*-
/*
* find_contacts.cu
*
*
*/

#include <cuda.h>
#include "cross_box.h"
#include "cudaErr.h"
#include <math.h>
#include <sm_20_atomic_functions.h>


using namespace std;


////////////////////////////////////////////////////////
// Finds contacts:
//  Returns number of contacts of each particle to pnContacts
//  Returns 2*(the total numer of contacts) to pnTotContacts
//
// The neighbor list pnNbrList has a list of possible contacts
//  for each particle and is found in find_neighbors.cu
//
/////////////////////////////////////////////////////////
__global__ void find_cont(int nCross, int *pnNPP, int *pnNbrList, double dL,
			  double dGamma, double *pdX, double *pdY, double *pdR,
			  double dEpsilon, int *pnContacts, int *pnTotContacts)
{
  // Declare shared memory pointer, the size is determined at the kernel launch
  extern __shared__ int sData[];
  int thid = threadIdx.x;
  int nPID = thid + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;
  sData[thid] = 0;
  __syncthreads();  // synchronizes every thread in the block before going on

  while (nPID < nCross)
    {
      double dX = pdX[nPID];
      double dY = pdY[nPID];
      double dR = pdR[nPID];
      
      int nContacts = 0;
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
	    nContacts += 1;
	}
      pnContacts[nPID] = nContacts;
      sData[thid] += nContacts;

      nPID += nThreads;
    }
  __syncthreads();

  // Now we do a parallel reduction sum to find the total number of contacts
  int offset = blockDim.x / 2;
  while (offset > 32)
    {
      if (thid < offset)
	sData[thid] += sData[thid + offset];
      offset /= 2;  
      __syncthreads();
    }
  if (thid < 32) //unroll end of loop (no need to sync since warp size=32
  {
    sData[thid] += sData[thid + 32];
    if (thid < 16)
    {
      sData[thid] += sData[thid + 16];
      if (thid < 8)
      {
	sData[thid] += sData[thid + 8];
	if (thid < 4)
	{
	  sData[thid] += sData[thid + 4];
	  if (thid < 2)
	  {
	    sData[thid] += sData[thid + 2];
	    if (thid == 0)
	    {
	      sData[0] += sData[1];
	      int tot = atomicAdd(pnTotContacts, sData[0]);
	    }
	  }
	}
      }
    }
  }

}


void Cross_Box::find_contacts()
{
  cudaMemset(d_pnTotContacts, 0, sizeof(int));

  find_cont <<<m_nGS_FindContact, m_nBS_FindContact, m_nSM_FindContact>>>
    (m_nCross, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, d_pdX, d_pdY, 
     d_pdR, m_dEpsilon, d_pnContacts, d_pnTotContacts);
  cudaThreadSynchronize();

}
