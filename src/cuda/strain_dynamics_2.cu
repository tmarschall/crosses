
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
#include <string.h>
#include <stdlib.h>

using namespace std;


///////////////////////////////////////////////////////////////
//
//
///////////////////////////////////////////////////////////
template<Potential ePot, bool bCalcStress>
__global__ void euler_est(int nCross, int *pnBlockNNbrs, int *pnBlockList, int *pnNPP, 
			  int *pnNbrList, double dL, double dGamma, double *pdGlobX, 
			  double *pdGlobY, double *pdGlobR, double *pdFx, double *pdFy, 
			  float *pfSE, double dStep, double *pdTempX, double *pdTempY)
{ 
  int thid = threadIdx.x;
  int blid = blockIdx.x;
  int nPID = thid + blid * blockDim.x;
  int blsz = blockDim.x;
  int nThreads = blsz * gridDim.x;
  // Declare shared memory pointer, the size is passed at the kernel launch
  extern __shared__ double sData[];
  int offset; 
  if (bCalcStress) {
    offset = blsz + 8; // +8 should help to avoid a few bank conflicts 
    for (int i = 0; i < 4; i++)
      sData[thid + i*offset] = 0.0;
    __syncthreads();  // synchronizes every thread in the block before going on
  }
  else {
    offset = 0;
  }
  double *pdR = &sData[4*offset];

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
	      if (bCalcStress)
		{
		  if (nAdjPID > nPID)
		    {
		      sData[thid] += dDVij * dSigma * (1.0 - dDelR / dSigma) / (dAlpha * dL * dL);
		      sData[thid + offset] += dPfx * dDeltaX / (dL * dL);
		      sData[thid + 2*offset] += dPfy * dDeltaY / (dL * dL);
		      sData[thid + 3*offset] += dPfx * dDeltaY / (dL * dL);
		    } 
		}
	    }
	}
      pdFx[nPID] = dFx;
      pdFy[nPID] = dFy;
      pdTempX[nPID] = dX + dStep * (dFx - dGamma * dFy);
      pdTempY[nPID] = dY + dStep * dFy;
      
      blid += gridDim.x;
      nPID += nThreads;
    }
  if (bCalcStress)
    {
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
}

///////////////////////////////////////////////////////////////////
//
//
/////////////////////////////////////////////////////////////////
template<Potential ePot>
__global__ void heun_corr(int nCross, int *pnBlockNNbrs, int *pnBlockList, int *pnNPP, 
			  int *pnNbrList, double dL, double dGamma, double *pdGlobX, double *pdGlobY, 
			  double *pdGlobR, double *pdFx, double *pdFy, double dStep, double dStrRate, 
			  double *pdTempX, double *pdTempY, double *dLabX, double *dLabY, 
			  double *pdXMoved, double *pdYMoved, double dEpsilon, int *bNewNbrs)
{
  extern __shared__ double sData[];
  int thid = threadIdx.x;
  int blid = blockIdx.x;
  int blsz = blockDim.x;
  int nPID = thid + blid * blsz;
  int nThreads = blsz * gridDim.x;
  double *pdR = &sData[0];

  while (nPID < nCross)
    {
      double dFx = 0.0;
      double dFy = 0.0;
      
      double dX = pdTempX[nPID];
      double dY = pdTempY[nPID];
      double dR = pdGlobR[nPID];

      int nBlockNbrs = pnBlockNNbrs[blid];
      double *pdX = &pdR[nBlockNbrs];
      double *pdY = &pdX[nBlockNbrs];
      for (int t = thid; t < nBlockNbrs; t++) {
	int nGlobID = pnBlockList[8*blid*blsz + t];
	pdR[t] = pdGlobR[nGlobID];
	pdX[t] = pdTempX[nGlobID];
	pdY[t] = pdTempY[nGlobID];
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
	  dDeltaX += (dGamma + dStep * dStrRate) * dDeltaY;
	  
	  // Check if they overlap
	  double dRSqr = dDeltaX*dDeltaX + dDeltaY*dDeltaY;
	  if (dRSqr < dSigma*dSigma)
	    {
	      double dDelR = sqrt(dRSqr);
	      double dDVij;
	      if (ePot == HARMONIC)
		{
		  dDVij = (1.0 - dDelR / dSigma) / dSigma;
		}
	      else if (ePot == HERTZIAN)
		{
		  dDVij = (1.0 - dDelR / dSigma) * sqrt(1.0 - dDelR / dSigma) / dSigma;
		}
	      dFx += dDeltaX * dDVij / dDelR;
	      dFy += dDeltaY * dDVij / dDelR;
	    }
	}
      dFx -= (dGamma + dStep * dStrRate) * dFy;
      double dFy0 = pdFy[nPID];
      double dFx0 = pdFx[nPID] - dGamma * dFy0;
      double dDx = 0.5 * dStep * (dFx0 + dFx);
      double dDy = 0.5 * dStep * (dFy0 + dFy);

      pdGlobX[nPID] += dDx;
      pdGlobY[nPID] += dDy;
      pdXMoved[nPID] += dDx;
      pdYMoved[nPID] += dDy;
      if (fabs(pdXMoved[nPID]) > dEpsilon || fabs(pdYMoved[nPID]) > dEpsilon)
	*bNewNbrs = 1;
      
      blid += gridDim.x;
      nPID += nThreads;
    }
}


////////////////////////////////////////////////////////////////////////
//
//
////////////////////////////////////////////////////////////////////
void Cross_Box::strain_step(long unsigned int tTime, bool bSvStress, bool bSvPos)
{
  if (bSvStress)
    {
      cudaMemset((void *) d_pfSE, 0, 4*sizeof(float));

      switch (m_ePotential)
	{
	case HARMONIC:
	  euler_est <HARMONIC, 1> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
	    (m_nCross, d_pnBlockNNbrs, d_pnBlockList, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, 
	     d_pdX, d_pdY, d_pdR, d_pdFx, d_pdFy, d_pfSE, m_dStep, d_pdTempX, d_pdTempY);
	  break;
	case HERTZIAN:
	  euler_est <HERTZIAN, 1> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
	    (m_nCross, d_pnBlockNNbrs, d_pnBlockList, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, 
	     d_pdX, d_pdY, d_pdR, d_pdFx, d_pdFy, d_pfSE, m_dStep, d_pdTempX, d_pdTempY);
	}
      cudaThreadSynchronize();
      checkCudaError("Estimating new particle positions, calculating stresses");

      cudaMemcpyAsync(h_pfSE, d_pfSE, 4*sizeof(float), cudaMemcpyDeviceToHost);
      if (bSvPos)
	{
	  cudaMemcpyAsync(h_pdX, d_pdX, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
	  cudaMemcpyAsync(h_pdY, d_pdY, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
	  cudaMemcpyAsync(h_pdLabX, d_pdLabX, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
	  cudaMemcpyAsync(h_pdLabY, d_pdLabY, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
	}
      cudaThreadSynchronize();
    }
  else
    {
      switch (m_ePotential)
	{
	case HARMONIC:
	  euler_est <HARMONIC, 0> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcF>>>
	    (m_nCross, d_pnBlockNNbrs, d_pnBlockList, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, 
	     d_pdX, d_pdY, d_pdR, d_pdFx, d_pdFy, d_pfSE, m_dStep, d_pdTempX, d_pdTempY);
	  break;
	case HERTZIAN:
	  euler_est <HERTZIAN, 0> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcF>>>
	    (m_nCross, d_pnBlockNNbrs, d_pnBlockList, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, 
	     d_pdX, d_pdY, d_pdR, d_pdFx, d_pdFy, d_pfSE, m_dStep, d_pdTempX, d_pdTempY);
	}
      cudaThreadSynchronize();
      checkCudaError("Estimating new particle positions");
    }

  switch (m_ePotential)
    {
    case HARMONIC:
      heun_corr <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcF>>>
	(m_nCross, d_pnBlockNNbrs, d_pnBlockList, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, 
	 d_pdX, d_pdY, d_pdR, d_pdFx, d_pdFy, m_dStep, m_dStrainRate, d_pdTempX, d_pdTempY, 
	 d_pdLabX, d_pdLabY, d_pdXMoved, d_pdYMoved, m_dEpsilon, d_bNewNbrs);
      break;
    case HERTZIAN:
      heun_corr <HERTZIAN> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcF>>>
	(m_nCross, d_pnBlockNNbrs, d_pnBlockList, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, 
	 d_pdX, d_pdY, d_pdR, d_pdFx, d_pdFy, m_dStep, m_dStrainRate, d_pdTempX, d_pdTempY, 
	 d_pdLabX, d_pdLabY, d_pdXMoved, d_pdYMoved, m_dEpsilon, d_bNewNbrs);
    }

  if (bSvStress)
    {
      m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
      fprintf(m_pOutfSE, "%lu %.7g %.7g %.7g %.7g %.7g\n", 
	      tTime, *m_pfEnergy, *m_pfPxx, *m_pfPyy, m_fP, *m_pfPxy);
      if (bSvPos)
	save_positions(tTime);
    }

  cudaThreadSynchronize();
  checkCudaError("Updating estimates, moving particles");
  
  cudaMemcpyAsync(h_bNewNbrs, d_bNewNbrs, sizeof(int), cudaMemcpyDeviceToHost);

  m_dGamma += m_dStep * m_dStrainRate;
  m_dTotalGamma += m_dStep * m_dStrainRate;
  cudaThreadSynchronize();

  if (m_dGamma > 0.5)
    set_back_gamma();
  else if (*h_bNewNbrs)
    find_neighbors();
}


/////////////////////////////////////////////////////////////////
//
//
//////////////////////////////////////////////////////////////
void Cross_Box::save_positions(long unsigned int nTime)
{
  char szBuf[150];
  sprintf(szBuf, "%s/sd%010lu.dat", m_szDataDir, nTime);
  const char *szFilePos = szBuf;
  FILE *pOutfPos;
  pOutfPos = fopen(szFilePos, "w");
  if (pOutfPos == NULL)
    {
      fprintf(stderr, "Could not open file for writing");
      exit(1);
    }

  int i = h_pnMemID[0];
  fprintf(pOutfPos, "%d %f %.13f %f %.13g %.13g %.13g %.13g %.13g %.13g\n", 
	  0, m_dPacking, m_dL, h_pdR[i], h_pdLabX[i], h_pdLabY[i], h_pdX[i], h_pdY[i], m_dGamma, m_dTotalGamma);
  for (int p = 1; p < m_nCross; p++)
    {
      i = h_pnMemID[p];
      fprintf(pOutfPos, "%d %f %.13f %f %.13g %.13g %.13g %.13g\n",
	      p, m_dPacking, m_dL, h_pdR[i], h_pdLabX[i], h_pdLabY[i], h_pdX[i], h_pdY[i]);
    }

  fclose(pOutfPos); 
}


////////////////////////////////////////////////////////////////////////
//
//
//////////////////////////////////////////////////////////////////////
void Cross_Box::run_strain(double dStartGamma, double dStopGamma, double dSvStressGamma, double dSvPosGamma)
{
  if (m_dStrainRate == 0.0)
    {
      fprintf(stderr, "Cannot strain with zero strain rate\n");
      exit(1);
    }

  printf("Beginnig strain run with strain rate: %g and step %g\n", m_dStrainRate, m_dStep);
  fflush(stdout);

  if (dSvStressGamma < m_dStrainRate * m_dStep)
    dSvStressGamma = m_dStrainRate * m_dStep;
  if (dSvPosGamma < m_dStrainRate)
    dSvPosGamma = m_dStrainRate;

  // +0.5 to cast to nearest integer rather than rounding down
  unsigned long int nTime = (unsigned long)(dStartGamma / m_dStrainRate + 0.5);
  unsigned long int nStop = (unsigned long)(dStopGamma / m_dStrainRate + 0.5);
  unsigned int nIntStep = (unsigned int)(1.0 / m_dStep + 0.5);
  unsigned int nSvStressInterval = (unsigned int)(dSvStressGamma / (m_dStrainRate * m_dStep) + 0.5);
  unsigned int nSvPosInterval = (unsigned int)(dSvPosGamma / m_dStrainRate + 0.5);
  unsigned long int nTotalStep = nTime * nIntStep;
  //unsigned int nReorderInterval = (unsigned int)(1.0 / m_dStrainRate + 0.5);
  
  printf("Strain run configured\n");
  printf("Start: %lu, Stop: %lu, Int step: %lu\n", nTime, nStop, nIntStep);
  printf("Stress save int: %lu, Pos save int: %lu\n", nSvStressInterval, nSvPosInterval);
  fflush(stdout);

  char szBuf[200];
  sprintf(szBuf, "%s/%s", m_szDataDir, m_szFileSE);
  const char *szPathSE = szBuf;
  if (nTime == 0)
    {
      m_pOutfSE = fopen(szPathSE, "w");
      if (m_pOutfSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
      cudaMemcpy(d_pdLabX, d_pdX, m_nCross*sizeof(double), cudaMemcpyDeviceToDevice);
      cudaMemcpy(d_pdLabY, d_pdY, m_nCross*sizeof(double), cudaMemcpyDeviceToDevice);
    }
  else
    {  
      m_pOutfSE = fopen(szPathSE, "r+");
      if (m_pOutfSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
      
      int nTpos = 0;
      while (nTpos != nTime)
	{
	  if (fgets(szBuf, 200, m_pOutfSE) != NULL)
	    {
	      int nPos = strcspn(szBuf, " ");
	      char szTime[20];
	      strncpy(szTime, szBuf, nPos);
	      szTime[nPos] = '\0';
	      nTpos = atoi(szTime);
	    }
	  else
	    {
	      fprintf(stderr, "Reached end of file without finding start position");
	      exit(1);
	    }
	}
    }

  // Run strain for specified number of steps
  while (nTime < nStop)
    {
      bool bSvPos = (nTime % nSvPosInterval == 0);
      if (bSvPos)
	strain_step(nTime, 1, 1);
      else
	{
	  bool bSvStress = (nTotalStep % nSvStressInterval == 0);
	  strain_step(nTime, bSvStress, 0);
	}
      nTotalStep += 1;
      for (unsigned int nI = 1; nI < nIntStep; nI++)
	{
	  bool bSvStress = (nTotalStep % nSvStressInterval == 0); 
	  strain_step(nTime, bSvStress, 0);
	  nTotalStep += 1;
	}
      nTime += 1;
      //if (nTime % nReorderInterval == 0)
      //reorder_particles();
    }
  
  // Save final configuration
  calculate_stress_energy();
  cudaMemcpyAsync(h_pdX, d_pdX, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdY, d_pdY, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdLabX, d_pdLabX, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdLabY, d_pdLabY, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pfSE, d_pfSE, 4*sizeof(float), cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
  m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
  fprintf(m_pOutfSE, "%lu %.7g %.7g %.7g %.7g %.7g\n", 
	  nTime, *m_pfEnergy, *m_pfPxx, *m_pfPyy, m_fP, *m_pfPxy);
  save_positions(nTime);
  
  fclose(m_pOutfSE);
}

void Cross_Box::run_strain(long unsigned int nSteps)
{
  // Run strain for specified number of steps
  long unsigned int nTime = 0;
  while (nTime < nSteps)
    {
      strain_step(nTime, 0, 0);
      nTime += 1;
    }

}

