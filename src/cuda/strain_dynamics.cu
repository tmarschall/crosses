
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
#include <sm_35_atomic_functions.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

using namespace std;

const double D_PI = 3.14159265358979;



template<Potential ePot, int bCalcStress>
__global__ void euler_est(int nCross, int *pnNPP, int *pnNbrList, double dL, double dGamma, double dStrain, 
			  double dStep, double *pdX, double *pdY, double *pdPhi, double *pdR, double *pdAx, 
			  double *pdAy, double dKd, double *pdArea, double *pdMOI, double *pdIsoC, double *pdFx, 
			  double *pdFy, double *pdFt, float *pfSE, double *pdTempX, double *pdTempY, double *pdTempPhi)
{ 
  int thid = threadIdx.x;
  int spi = (thid/2) % 2;
  int spj = thid % 2;
  int nPID = (thid + blockIdx.x * blockDim.x)/4;
  int nThreads = blockDim.x * gridDim.x;
  // Declare shared memory pointer, the size is passed at the kernel launch
  extern __shared__ double sData[];
  int offset = blockDim.x + 8; // +8 should help to avoid a few bank conflict
  if (bCalcStress) {
      for (int i = 0; i < 5; i++)
        sData[3*blockDim.x + i*offset + thid] = 0.0;
    __syncthreads();  // synchronizes every thread in the block before going on
  }
  //if (thid == 0) {
  //  printf("Shared data allocated and zeroed on block %d\n", blockIdx.x);
  //}
    
  double *dFx = sData;
  double *dFy = sData+blockDim.x;
  double *dFt = sData+2*blockDim.x;

  while (nPID < nCross) {
      dFx[thid] = 0.0;
      dFy[thid] = 0.0;
      dFt[thid] = 0.0;
      //if (thid == 0) {
      //	printf("Forces reset on block %d\n", blockIdx.x);
      //}
      
      double dX = pdX[nPID];
      double dY = pdY[nPID];
      double dPhi = pdPhi[nPID] + spi*D_PI/2;
      double dR = pdR[nPID];
      double dA = spi == 0 ? pdAx[nPID] : pdAy[nPID];
      
      int nNbrs = pnNPP[nPID];

      for (int p = 0; p < nNbrs; p++) {
    	  int nAdjPID = pnNbrList[nPID + p * nCross];
	  //if (spi == 0 && spj == 0) {
	  //  printf("nPID: %d nAdjPID: %d\n", nPID, nAdjPID);
	  //}
    	  double dAdjX = pdX[nAdjPID];
    	  double dAdjY = pdY[nAdjPID];
	  
    	  double dDeltaX = dX - dAdjX;
    	  double dDeltaY = dY - dAdjY;
    	  double dPhiB = pdPhi[nAdjPID] + spj*D_PI/2;
    	  double dSigma = dR + pdR[nAdjPID];
    	  double dB = spj == 0 ? pdAx[nPID] : pdAy[nPID];


    	  // Make sure we take the closest distance considering boundary conditions
    	  dDeltaX += dL * ((dDeltaX < -0.5*dL) - (dDeltaX > 0.5*dL));
    	  dDeltaY += dL * ((dDeltaY < -0.5*dL) - (dDeltaY > 0.5*dL));
    	  // Transform from shear coordinates to lab coordinates
    	  dDeltaX += dGamma * dDeltaY;
	  
    	  double nxA = dA * cos(dPhi);
    	  double nyA = dA * sin(dPhi);
    	  double nxB = dB * cos(dPhiB);
    	  double nyB = dB * sin(dPhiB);

    	  double a = dA * dA;
    	  double b = -(nxA * nxB + nyA * nyB);
    	  double c = dB * dB;
    	  double d = nxA * dDeltaX + nyA * dDeltaY;
    	  double e = -nxB * dDeltaX - nyB * dDeltaY;
    	  double delta = a * c - b * b;

    	  double t = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
    	  double s = -(b*t+d)/a;
    	  double sarg = fabs(s);
    	  s = fmin( fmax(s,-1.), 1. );
    	  if (sarg > 1)
    		  t = fmin( fmax( -(b*s+e)/c, -1.), 1.);
	  
    	  // Check if they overlap and calculate forces
    	  double dDx = dDeltaX + s*nxA - t*nxB;
    	  double dDy = dDeltaY + s*nyA - t*nyB;
    	  double dDSqr = dDx * dDx + dDy * dDy;

    	  //printf("nPID: %d, spi: %d, nAdjPID: %d, spj: %d, dPhi: %g, dPhiB: %g, s: %g, t: %g, Dx: %g, Dy: %g, Dt: %g\n",
    	  //    	   nPID, spi, nAdjPID, spj, dPhi, dPhiB, s, t, dDx, dDy, atan(dDy/dDx));
    	  if (dDSqr < dSigma*dSigma) {
    		  double dDij = sqrt(dDSqr);
    		  double dDVij;
    		  double dAlpha;
    		  if (ePot == HARMONIC) {
    			  dDVij = (1.0 - dDij / dSigma) / dSigma;
    			  dAlpha = 2.0;
    		  }
    		  else if (ePot == HERTZIAN) {
    			  dDVij = (1.0 - dDij / dSigma) * sqrt(1.0 - dDij / dSigma) / dSigma;
    			  dAlpha = 2.5;
    		  }
    		  double dPfx = dDx * dDVij / dDij;
    		  double dPfy = dDy * dDVij / dDij;
    		  dFx[thid] += dPfx;
    		  dFy[thid] += dPfy;
    		  //  Find the point of contact (with respect to the center of the cross)
    		  double dCx = s*nxA - 0.5*dDx;
    		  double dCy = s*nyA - 0.5*dDy;
    		  //double dCx = s*nxA;
    		  //double dCy = s*nyA;
    		  dFt[thid] += dCx * dPfy - dCy * dPfx;
    		  if (bCalcStress) {
		    sData[3*blockDim.x + thid ] -= dPfx * dCx / (dL * dL);
		    sData[3*blockDim.x + thid + offset] -= dPfy * dCx / (dL * dL);
		    sData[3*blockDim.x + thid + 2*offset] -= dPfx * dCy / (dL * dL);
		    sData[3*blockDim.x + thid + 3*offset] -= dPfy * dCy / (dL * dL);
		    if (nAdjPID > nPID) {
		      sData[3*blockDim.x + thid + 4*offset] += dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dL * dL);
		      
		      //if (thid == 0) {
		      //  printf("Stresses updated on block %d\n", blockIdx.x);
		      //}
		    }
    		  }

    		  //printf("nPID: %d, spi: %d, nAdjPID: %d, spj: %d, dPhi: %g, dA: %g, dPhiB: %g, dB: %g, s*nxA: %g, s*nyA: %g, t*nxB %g, t*nyB: %g, Dx: %g, Dy: %g, Fx: %g, Fy: %g\n",
    		  //	   nPID, spi, nAdjPID, spj, dPhi, dA, dPhiB, dB, dCx, dCy, t*nxB, t*nyB, dDx, dDy, dPfx, dPfy);
    	  }
      }

      if (spi == 0) {
    	  dFx[thid] += dFx[thid + 2];
    	  dFy[thid] += dFy[thid + 2];
    	  dFt[thid] += dFt[thid + 2];
    	  if (spj == 0) {
    		  dFx[thid] += dFx[thid + 1];
    		  dFy[thid] += dFy[thid + 1];
    		  dFt[thid] += dFt[thid + 1];
    		  //printf("PID: %d block: %d thid: %d - fx: %g fy: %g t: %g\n", nPID, blockIdx.x, thid, dFx[thid], dFy[thid], dFt[thid]);

    		  pdFx[nPID] = dFx[thid];
    		  pdFy[nPID] = dFy[thid];
    		  pdFt[nPID] = dFt[thid];
		  double dArea = pdArea[nPID];
		  dFx[thid] /= (dKd*dArea);
		  dFy[thid] /= (dKd*dArea);
		  dFt[thid] /= (dKd*dArea*pdMOI[nPID]);

    		  pdTempX[nPID] = dX + dStep * (dFx[thid] - dGamma * dFy[thid]);
    		  pdTempY[nPID] = dY + dStep * dFy[thid];
    		  double dRIso = 0.5*(1-pdIsoC[nPID]*cos(2*dPhi));
    		  pdTempPhi[nPID] = dPhi + dStep * (dFt[thid] - dStrain * dRIso);
          }
      }
      
      nPID += nThreads/4;
  }

  if (bCalcStress) {
	  __syncthreads();

	  // Now we do a parallel reduction sum to find the total number of contacts
	  int stride = blockDim.x / 2;  // stride is 1/2 block size, all threads perform two adds
	  int base = 3*blockDim.x + thid % stride + offset * (thid / stride);
	  sData[base] += sData[base + stride];
	  base += 2*offset;
	  sData[base] += sData[base + stride];
	  if (thid < stride) {
	    base += 2*offset;
	    sData[base] += sData[base + stride];
	  }
	  stride /= 2; // stride is 1/4 block size, all threads perform 1 add
	  __syncthreads();
	  base = 3*blockDim.x + thid % stride + offset * (thid / stride);
	  sData[base] += sData[base + stride];
	  if (thid < stride) {
	    base += 4*offset;
	    sData[base] += sData[base + stride];
	  }
	  stride /= 2;
	  __syncthreads();
	  while (stride > 6) {
		  if (thid < 5 * stride) {
			  base = 3*blockDim.x + thid % stride + offset * (thid / stride);
			  sData[base] += sData[base + stride];
		  }
		  stride /= 2;
		  __syncthreads();
	  }
	  if (thid < 20) {
	    base = 3*blockDim.x + thid % 4 + offset * (thid / 4);
	    sData[base] += sData[base + 4];
	    if (thid < 10) {
	      base = 3*blockDim.x + thid % 2 + offset * (thid / 2);
	      sData[base] += sData[base + 2];
	      if (thid < 5) {
		sData[3*blockDim.x + thid * offset] += sData[3*blockDim.x + thid * offset + 1];
		float tot = atomicAdd(pfSE+thid, (float)sData[3*blockDim.x + thid*offset]);
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
__global__ void heun_corr(int nCross, int *pnNPP,int *pnNbrList, double dL, double dGamma,  double dStrain, 
			  double dStep, double *pdX, double *pdY, double *pdPhi, double *pdR, double *pdAx, 
			  double *pdAy, double dKd, double *pdArea, double *pdMOI, double *pdIsoC, double *pdFx,
			  double *pdFy, double *pdFt, double *pdTempX, double *pdTempY, double *pdTempPhi, 
			  double *pdXMoved, double *pdYMoved, double dEpsilon, int *bNewNbrs)
{ 
  int thid = threadIdx.x;
  int spi = (thid/2) % 2;
  int spj = thid % 2;
  int nPID = (thid + blockIdx.x * blockDim.x)/4;
  //printf("blockId: %d thid: %d nPID: %d spi: %d spj: %d\n", blockIdx.x, thid, nPID, spi, spj);
  int nThreads = blockDim.x * gridDim.x;
  // Declare shared memory pointer, the size is passed at the kernel launch
  extern __shared__ double sData[];
  double *dFx = sData;
  double *dFy = sData+blockDim.x;
  double *dFt = sData+2*blockDim.x;

  while (nPID < nCross) {
      dFx[thid] = 0.0;
      dFy[thid] = 0.0;
      dFt[thid] = 0.0;
      //if (thid == 0) {
      //printf("Forces reset on block %d\n", blockIdx.x);
      //}
      
      double dX = pdTempX[nPID];
      double dY = pdTempY[nPID];
      double dPhi = pdTempPhi[nPID] + spi*D_PI/2;
      double dR = pdR[nPID];
      double dA = spi == 0 ? pdAx[nPID] : pdAy[nPID];
      double dNewGamma = dGamma + dStep * dStrain;

      int nNbrs = pnNPP[nPID];
      for (int p = 0; p < nNbrs; p++) {
    	  int nAdjPID = pnNbrList[nPID + p * nCross];
    	  double dAdjX = pdTempX[nAdjPID];
    	  double dAdjY = pdTempY[nAdjPID];

    	  double dDeltaX = dX - dAdjX;
    	  double dDeltaY = dY - dAdjY;
    	  double dPhiB = pdTempPhi[nAdjPID] + spj*D_PI/2;
    	  double dSigma = dR + pdR[nAdjPID];
    	  double dB = spj == 0 ? pdAx[nAdjPID] : pdAy[nAdjPID];
    	  // Make sure we take the closest distance considering boundary conditions
    	  dDeltaX += dL * ((dDeltaX < -0.5*dL) - (dDeltaX > 0.5*dL));
    	  dDeltaY += dL * ((dDeltaY < -0.5*dL) - (dDeltaY > 0.5*dL));
    	  // Transform from shear coordinates to lab coordinates
    	  dDeltaX += dNewGamma * dDeltaY;
	  
    	  double nxA = dA * cos(dPhi);
    	  double nyA = dA * sin(dPhi);
    	  double nxB = dB * cos(dPhiB);
    	  double nyB = dB * sin(dPhiB);

    	  double a = dA * dA;
    	  double b = -(nxA * nxB + nyA * nyB);
    	  double c = dB * dB;
    	  double d = nxA * dDeltaX + nyA * dDeltaY;
    	  double e = -nxB * dDeltaX - nyB * dDeltaY;
    	  double delta = a * c - b * b;

    	  double t = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
    	  double s = -(b*t+d)/a;
    	  double sarg = fabs(s);
    	  s = fmin( fmax(s,-1.), 1. );
    	  if (sarg > 1)
    		  t = fmin( fmax( -(b*s+e)/c, -1.), 1.);

    	  // Check if they overlap and calculate forces
    	  double dDx = dDeltaX + s*nxA - t*nxB;
    	  double dDy = dDeltaY + s*nyA - t*nyB;
    	  double dDSqr = dDx * dDx + dDy * dDy;
    	  if (dDSqr < dSigma*dSigma) {
    		  double dDij = sqrt(dDSqr);
    		  double dDVij;
    		  //double dAlpha;
    		  if (ePot == HARMONIC)	{
    			  dDVij = (1.0 - dDij / dSigma) / dSigma;
    			  //dAlpha = 2.0;
    		  }
    		  else if (ePot == HERTZIAN) {
    			  dDVij = (1.0 - dDij / dSigma) * sqrt(1.0 - dDij / dSigma) / dSigma;
    			  //dAlpha = 2.5;
    		  }
    		  double dPfx = dDx * dDVij / dDij;
    		  double dPfy = dDy * dDVij / dDij;
    		  dFx[thid] += dPfx;
    		  dFy[thid] += dPfy;
    		  //double dCx = s*nxA - 0.5*dDx;
    		  //double dCy = s*nyA - 0.5*dDy;
    		  double dCx = s*nxA;
    		  double dCy = s*nyA;
    		  dFt[thid] += dCx * dPfy - dCy * dPfx;
    	  }
      }
      //if (thid == 0) {
      //	printf("Forces calculated on block %d\n", blockIdx.x);
      //}

      if (spi == 0) {
	//printf("Thread %d on block %d summing forces\n", thid, blockIdx.x);
    	  dFx[thid] += dFx[thid + 2];
    	  dFy[thid] += dFy[thid + 2];
    	  dFt[thid] += dFt[thid + 2];
    	  if (spj == 0) {
	    //printf("Thread %d on block %d summing forces\n", thid, blockIdx.x);
	    dFx[thid] += dFx[thid + 1];
	    dFy[thid] += dFy[thid + 1];
	    dFt[thid] += dFt[thid + 1];
	    
	    double dArea = pdArea[nPID];
	    double dMOI = pdMOI[nPID];
	    double dIsoC = pdIsoC[nPID];
	    dFy[thid] /= (dKd*dArea);
	    dFx[thid] = dFx[thid] / (dKd*dArea) - dNewGamma * dFy[thid];
	    dFt[thid] = dFt[thid] / (dKd*dArea*dMOI) - dStrain * 0.5 * (1 - dIsoC*cos(2*dPhi));
	    
	    double dFy0 = pdFy[nPID] / (dKd*dArea);
	    double dFx0 = pdFx[nPID] / (dKd*dArea) - dGamma * dFy0;
	    double dPhi0 = pdPhi[nPID];
	    
	    double dFt0 = pdFt[nPID] / (dKd*dArea*dMOI) - dStrain * 0.5 * (1 - dIsoC*cos(2*dPhi0));
	    
	    double dDx = 0.5 * dStep * (dFx0 + dFx[thid]);
	    double dDy = 0.5 * dStep * (dFy0 + dFy[thid]);
	    pdX[nPID] += dDx;
	    pdY[nPID] += dDy;
	    pdPhi[nPID] += 0.5 * dStep * (dFt0 + dFt[thid]);
	    
	    pdXMoved[nPID] += dDx;
	    pdYMoved[nPID] += dDy;
	    if (fabs(pdXMoved[nPID]) > 0.5*dEpsilon || fabs(pdYMoved[nPID]) > 0.5*dEpsilon)
	      *bNewNbrs = 1;
    	  }
      }

      nPID += nThreads/4;
    }
  //if (thid == 0) {
  // printf("Exiting block %d\n", blockIdx.x);
  //}
}


////////////////////////////////////////////////////////////////////////
//
//
////////////////////////////////////////////////////////////////////
void Cross_Box::strain_step(long unsigned int tTime, bool bSvStress, bool bSvPos)
{
  if (bSvStress)
    {
      cudaMemset((void *) d_pfSE, 0, 5*sizeof(float));

      switch (m_ePotential)
	{
	case HARMONIC:
	  euler_est <HARMONIC, 1> <<<4*m_nGridSize, m_nBlockSize, m_nSM_CalcSE+m_nSM_CalcF>>>
	    (m_nCross, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, m_dStrainRate, m_dStep, 
	     d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdAx, d_pdAy, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
	     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
	  break;
	case HERTZIAN:
	  euler_est <HERTZIAN, 1> <<<4*m_nGridSize, m_nBlockSize, m_nSM_CalcSE+m_nSM_CalcF>>>
	    (m_nCross, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, m_dStrainRate, m_dStep, 
	     d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdAx, d_pdAy, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
	     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
	}
      cudaThreadSynchronize();
      checkCudaError("Estimating new particle positions, calculating stresses");


      cudaMemcpyAsync(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
      /*
      cudaMemcpy(h_pdFx, d_pdFx, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_pdFy, d_pdFy, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_pdFt, d_pdFt, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
      for (int p = 0; p < 10; p++) {
    	  printf("%d: %g %g %g\n", p, h_pdFx[p], h_pdFy[p], h_pdFt[p]);
      }
      */
      if (bSvPos)
	{
	  cudaMemcpyAsync(h_pdX, d_pdX, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
	  cudaMemcpyAsync(h_pdY, d_pdY, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
	  cudaMemcpyAsync(h_pdPhi, d_pdPhi, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
	}
      cudaThreadSynchronize();
    }
  else
    {
      switch (m_ePotential)
	{
	case HARMONIC:
	  euler_est <HARMONIC, 0> <<<4*m_nGridSize, m_nBlockSize, m_nSM_CalcF>>>
	    (m_nCross, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, m_dStrainRate, m_dStep, 
	     d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdAx, d_pdAy, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC,
	     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
	  break;
	case HERTZIAN:
	  euler_est <HERTZIAN, 0> <<<4*m_nGridSize, m_nBlockSize, m_nSM_CalcF>>>
	    (m_nCross, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, m_dStrainRate, m_dStep, 
	     d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdAx, d_pdAy, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC,
	     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
	}
      cudaDeviceSynchronize();
      checkCudaError("Estimating new particle positions");
    }

  switch (m_ePotential)
    {
    case HARMONIC:
      heun_corr <HARMONIC> <<<4*m_nGridSize, m_nBlockSize, m_nSM_CalcF>>>
	(m_nCross, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, m_dStrainRate, m_dStep, 
	 d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdAx, d_pdAy, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC,
	 d_pdFx, d_pdFy, d_pdFt, d_pdTempX, d_pdTempY, d_pdTempPhi, d_pdXMoved, d_pdYMoved, m_dEpsilon, d_bNewNbrs);
      break;
    case HERTZIAN:
      heun_corr <HERTZIAN> <<<4*m_nGridSize, m_nBlockSize, m_nSM_CalcF>>>
	(m_nCross, d_pnNPP, d_pnNbrList, m_dL, m_dGamma,m_dStrainRate, m_dStep, 
	 d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdAx, d_pdAy, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC,
	 d_pdFx, d_pdFy, d_pdFt, d_pdTempX, d_pdTempY, d_pdTempPhi, d_pdXMoved, d_pdYMoved, m_dEpsilon, d_bNewNbrs);
    }

  if (bSvStress)
    {
      m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
      fprintf(m_pOutfSE, "%lu %.9g %.9g %.9g %.9g %.9g %.9g\n", 
	      tTime, *m_pfEnergy, m_fP, *m_pfPxx, *m_pfPxy, *m_pfPyx, *m_pfPyy);
      if (bSvPos)
	save_positions(tTime);
    }

  cudaDeviceSynchronize();
  checkCudaError("Updating estimates, moving particles");
  
  cudaMemcpyAsync(h_bNewNbrs, d_bNewNbrs, sizeof(int), cudaMemcpyDeviceToHost);

  m_dGamma += m_dStep * m_dStrainRate;
  m_dTotalGamma += m_dStep * m_dStrainRate;
  cudaDeviceSynchronize();

  if (m_dGamma > 0.5) {
    set_back_gamma();
    //printf("Setting back gamma at time %ld\n", tTime);
  }
  else if (*h_bNewNbrs) {
    find_neighbors();
    //printf("Updating neighbor list at time %ld\n", tTime);
  }
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
  fprintf(pOutfPos, "%d %.15g %f %.15g %.15g %g %g\n",
	  m_nCross, m_dL, m_dPacking, m_dGamma, m_dTotalGamma, m_dStrainRate, m_dStep);
  for (int p = 0; p < m_nCross; p++)
    {
      i = h_pnMemID[p];
      fprintf(pOutfPos, "%.15g %.15g %.15g %f %f %f\n",
	      h_pdX[i], h_pdY[i], h_pdPhi[i], h_pdR[i], h_pdAx[i], h_pdAy[i]);
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
      if (bSvPos) {
	strain_step(nTime, 1, 1);
	fflush(m_pOutfSE);
      }
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
  cudaMemcpyAsync(h_pdPhi, d_pdPhi, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
  m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
  fprintf(m_pOutfSE, "%lu %.9g %.9g %.9g %.9g %.9g %.9g\n", 
	  nTime, *m_pfEnergy, m_fP, *m_pfPxx, *m_pfPxy, *m_pfPyx, *m_pfPyy);
  fflush(m_pOutfSE);
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


//////////////////////////////////////////////////////////////////////////////////
//
//  Resize box
//
/////////////////////////////////////////////////////////////////////////////
template<Potential ePot, int bCalcStress>
__global__ void rs_euler_est(int nCross, int *pnNPP, int *pnNbrList, double dL, double dGamma, double dRsRate, double dStep, 
			     double *pdX, double *pdY, double *pdPhi, double *pdR, double *pdAx, double *pdAy, double dKd, 
			     double *pdArea, double *pdMOI, double *pdFx, double *pdFy, double *pdFt, float *pfSE, 
			     double *pdTempX, double *pdTempY, double *pdTempPhi)
{ 
  int thid = threadIdx.x;
  int spi = (thid/2) % 2;
  int spj = thid % 2;
  int nPID = (thid + blockIdx.x * blockDim.x)/4;
  int nThreads = blockDim.x * gridDim.x;
  // Declare shared memory pointer, the size is passed at the kernel launch
  extern __shared__ double sData[];
  int offset = blockDim.x + 8; // +8 should help to avoid a few bank conflict
  if (bCalcStress) {
      for (int i = 0; i < 5; i++)
        sData[3*blockDim.x + i*offset + thid] = 0.0;
    __syncthreads();  // synchronizes every thread in the block before going on
  }
  //if (thid == 0) {
  //  printf("Shared data allocated and zeroed on block %d\n", blockIdx.x);
  //}
    
  double *dFx = sData;
  double *dFy = sData+blockDim.x;
  double *dFt = sData+2*blockDim.x;

  while (nPID < nCross) {
      dFx[thid] = 0.0;
      dFy[thid] = 0.0;
      dFt[thid] = 0.0;
      //if (thid == 0) {
      //	printf("Forces reset on block %d\n", blockIdx.x);
      //}
      
      double dX = pdX[nPID];
      double dY = pdY[nPID];
      double dPhi = pdPhi[nPID] + spi*D_PI/2;
      double dR = pdR[nPID];
      double dA = spi == 0 ? pdAx[nPID] : pdAy[nPID];
      
      int nNbrs = pnNPP[nPID];

      for (int p = 0; p < nNbrs; p++) {
    	  int nAdjPID = pnNbrList[nPID + p * nCross];
	  //if (spi == 0 && spj == 0) {
	  //  printf("nPID: %d nAdjPID: %d\n", nPID, nAdjPID);
	  //}
    	  double dAdjX = pdX[nAdjPID];
    	  double dAdjY = pdY[nAdjPID];
	  
    	  double dDeltaX = dX - dAdjX;
    	  double dDeltaY = dY - dAdjY;
    	  double dPhiB = pdPhi[nAdjPID] + spj*D_PI/2;
    	  double dSigma = dR + pdR[nAdjPID];
    	  double dB = spj == 0 ? pdAx[nPID] : pdAy[nPID];


    	  // Make sure we take the closest distance considering boundary conditions
    	  dDeltaX += dL * ((dDeltaX < -0.5*dL) - (dDeltaX > 0.5*dL));
    	  dDeltaY += dL * ((dDeltaY < -0.5*dL) - (dDeltaY > 0.5*dL));
    	  // Transform from shear coordinates to lab coordinates
    	  dDeltaX += dGamma * dDeltaY;
	  
    	  double nxA = dA * cos(dPhi);
    	  double nyA = dA * sin(dPhi);
    	  double nxB = dB * cos(dPhiB);
    	  double nyB = dB * sin(dPhiB);

    	  double a = dA * dA;
    	  double b = -(nxA * nxB + nyA * nyB);
    	  double c = dB * dB;
    	  double d = nxA * dDeltaX + nyA * dDeltaY;
    	  double e = -nxB * dDeltaX - nyB * dDeltaY;
    	  double delta = a * c - b * b;

    	  double t = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
    	  double s = -(b*t+d)/a;
    	  double sarg = fabs(s);
    	  s = fmin( fmax(s,-1.), 1. );
    	  if (sarg > 1)
    		  t = fmin( fmax( -(b*s+e)/c, -1.), 1.);
	  
    	  // Check if they overlap and calculate forces
    	  double dDx = dDeltaX + s*nxA - t*nxB;
    	  double dDy = dDeltaY + s*nyA - t*nyB;
    	  double dDSqr = dDx * dDx + dDy * dDy;

    	  //printf("nPID: %d, spi: %d, nAdjPID: %d, spj: %d, dPhi: %g, dPhiB: %g, s: %g, t: %g, Dx: %g, Dy: %g, Dt: %g\n",
    	  //    	   nPID, spi, nAdjPID, spj, dPhi, dPhiB, s, t, dDx, dDy, atan(dDy/dDx));
    	  if (dDSqr < dSigma*dSigma) {
    		  double dDij = sqrt(dDSqr);
    		  double dDVij;
    		  double dAlpha;
    		  if (ePot == HARMONIC) {
		    dDVij = (1.0 - dDij / dSigma) / dSigma;
		    dAlpha = 2.0;
    		  }
    		  else if (ePot == HERTZIAN) {
		    dDVij = (1.0 - dDij / dSigma) * sqrt(1.0 - dDij / dSigma) / dSigma;
		    dAlpha = 2.5;
    		  }
    		  double dPfx = dDx * dDVij / dDij;
    		  double dPfy = dDy * dDVij / dDij;
    		  dFx[thid] += dPfx;
    		  dFy[thid] += dPfy;
    		  //  Find the point of contact (with respect to the center of the cross)
    		  double dCx = s*nxA - 0.5*dDx;
    		  double dCy = s*nyA - 0.5*dDy;
    		  //double dCx = s*nxA;
    		  //double dCy = s*nyA;
    		  dFt[thid] += dCx * dPfy - dCy * dPfx;
    		  if (bCalcStress) {
		    sData[3*blockDim.x + thid ] -= dPfx * dCx / (dL * dL);
		    sData[3*blockDim.x + thid + offset] -= dPfy * dCx / (dL * dL);
		    sData[3*blockDim.x + thid + 2*offset] -= dPfx * dCy / (dL * dL);
		    sData[3*blockDim.x + thid + 3*offset] -= dPfy * dCy / (dL * dL);
		    if (nAdjPID > nPID) {
		      sData[3*blockDim.x + thid + 4*offset] += dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dL * dL);
		      
		      //if (thid == 0) {
		      //  printf("Stresses updated on block %d\n", blockIdx.x);
		      //}
		    }
    		  }

    		  //printf("nPID: %d, spi: %d, nAdjPID: %d, spj: %d, dPhi: %g, dA: %g, dPhiB: %g, dB: %g, s*nxA: %g, s*nyA: %g, t*nxB %g, t*nyB: %g, Dx: %g, Dy: %g, Fx: %g, Fy: %g\n",
    		  //	   nPID, spi, nAdjPID, spj, dPhi, dA, dPhiB, dB, dCx, dCy, t*nxB, t*nyB, dDx, dDy, dPfx, dPfy);
    	  }
      }

      if (spi == 0) {
	dFx[thid] += dFx[thid + 2];
	dFy[thid] += dFy[thid + 2];
	dFt[thid] += dFt[thid + 2];
	if (spj == 0) {
	  dFx[thid] += dFx[thid + 1];
	  dFy[thid] += dFy[thid + 1];
	  dFt[thid] += dFt[thid + 1];
	  //printf("PID: %d block: %d thid: %d - fx: %g fy: %g t: %g\n", nPID, blockIdx.x, thid, dFx[thid], dFy[thid], dFt[thid]);
	  
	  pdFx[nPID] = dFx[thid];
	  pdFy[nPID] = dFy[thid];
	  pdFt[nPID] = dFt[thid];

	  double dArea = pdArea[nPID];
	  dFx[thid] /= (dKd*dArea);
	  dFy[thid] /= (dKd*dArea);
	  dFt[thid] /= (dKd*dArea*pdMOI[nPID]);
	  pdTempX[nPID] = dX + dStep * (dRsRate*dX + dFx[thid] - dGamma * dFy[thid]);
	  pdTempY[nPID] = dY + dStep * (dRsRate*dY + dFy[thid]);
	  pdTempPhi[nPID] = dPhi + dStep * dFt[thid];
	}
      }
      
      nPID += nThreads/4;
  }

  if (bCalcStress) {
    __syncthreads();
    
    // Now we do a parallel reduction sum to find the total number of contacts
    int stride = blockDim.x / 2;  // stride is 1/2 block size, all threads perform two adds
    int base = 3*blockDim.x + thid % stride + offset * (thid / stride);
    sData[base] += sData[base + stride];
    base += 2*offset;
    sData[base] += sData[base + stride];
    if (thid < stride) {
      base += 2*offset;
      sData[base] += sData[base + stride];
    }
    stride /= 2; // stride is 1/4 block size, all threads perform 1 add
    __syncthreads();
    base = 3*blockDim.x + thid % stride + offset * (thid / stride);
    sData[base] += sData[base + stride];
    if (thid < stride) {
      base += 4*offset;
      sData[base] += sData[base + stride];
    }
    stride /= 2;
    __syncthreads();
    while (stride > 6) {
      if (thid < 5 * stride) {
	base = 3*blockDim.x + thid % stride + offset * (thid / stride);
	sData[base] += sData[base + stride];
      }
      stride /= 2;
      __syncthreads();
    }
    if (thid < 20) {
      base = 3*blockDim.x + thid % 4 + offset * (thid / 4);
      sData[base] += sData[base + 4];
      if (thid < 10) {
	base = 3*blockDim.x + thid % 2 + offset * (thid / 2);
	sData[base] += sData[base + 2];
	if (thid < 5) {
	  sData[3*blockDim.x + thid * offset] += sData[3*blockDim.x + thid * offset + 1];
	  float tot = atomicAdd(pfSE+thid, (float)sData[3*blockDim.x + thid*offset]);
	}
      }
    }
  }
}


template<Potential ePot>
__global__ void rs_heun_corr(int nCross, int *pnNPP,int *pnNbrList,double dL, double dGamma, double dRsRate, double dStep, 
			     double *pdX, double *pdY, double *pdPhi, double *pdR, double *pdAx, double *pdAy, double dKd,
			     double *pdArea, double *pdMOI, double *pdFx, double *pdFy, double *pdFt, double *pdTempX, double *pdTempY, 
			     double *pdTempPhi, double *pdXMoved, double *pdYMoved, double dEpsilon, int *bNewNbrs)
{ 
  int thid = threadIdx.x;
  int spi = (thid/2) % 2;
  int spj = thid % 2;
  int nPID = (thid + blockIdx.x * blockDim.x)/4;
  //printf("blockId: %d thid: %d nPID: %d spi: %d spj: %d\n", blockIdx.x, thid, nPID, spi, spj);
  int nThreads = blockDim.x * gridDim.x;
  // Declare shared memory pointer, the size is passed at the kernel launch
  extern __shared__ double sData[];
  double *dFx = sData;
  double *dFy = sData+blockDim.x;
  double *dFt = sData+2*blockDim.x;

  while (nPID < nCross) {
      dFx[thid] = 0.0;
      dFy[thid] = 0.0;
      dFt[thid] = 0.0;
      //if (thid == 0) {
      //printf("Forces reset on block %d\n", blockIdx.x);
      //}
      
      double dX = pdTempX[nPID];
      double dY = pdTempY[nPID];
      double dPhi = pdTempPhi[nPID] + spi*D_PI/2;
      double dR = pdR[nPID];
      double dA = spi == 0 ? pdAx[nPID] : pdAy[nPID];
      double dNewL = dStep * dRsRate * dL;

      int nNbrs = pnNPP[nPID];
      for (int p = 0; p < nNbrs; p++) {
    	  int nAdjPID = pnNbrList[nPID + p * nCross];
    	  double dAdjX = pdTempX[nAdjPID];
    	  double dAdjY = pdTempY[nAdjPID];

    	  double dDeltaX = dX - dAdjX;
    	  double dDeltaY = dY - dAdjY;
    	  double dPhiB = pdTempPhi[nAdjPID] + spj*D_PI/2;
    	  double dSigma = dR + pdR[nAdjPID];
    	  double dB = spj == 0 ? pdAx[nAdjPID] : pdAy[nAdjPID];
    	  // Make sure we take the closest distance considering boundary conditions
    	  dDeltaX += dNewL * ((dDeltaX < -0.5*dNewL) - (dDeltaX > 0.5*dNewL));
    	  dDeltaY += dNewL * ((dDeltaY < -0.5*dNewL) - (dDeltaY > 0.5*dNewL));
    	  // Transform from shear coordinates to lab coordinates
    	  dDeltaX += dGamma * dDeltaY;
	  
    	  double nxA = dA * cos(dPhi);
    	  double nyA = dA * sin(dPhi);
    	  double nxB = dB * cos(dPhiB);
    	  double nyB = dB * sin(dPhiB);

    	  double a = dA * dA;
    	  double b = -(nxA * nxB + nyA * nyB);
    	  double c = dB * dB;
    	  double d = nxA * dDeltaX + nyA * dDeltaY;
    	  double e = -nxB * dDeltaX - nyB * dDeltaY;
    	  double delta = a * c - b * b;

    	  double t = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
    	  double s = -(b*t+d)/a;
    	  double sarg = fabs(s);
    	  s = fmin( fmax(s,-1.), 1. );
    	  if (sarg > 1)
	    t = fmin( fmax( -(b*s+e)/c, -1.), 1.);

    	  // Check if they overlap and calculate forces
    	  double dDx = dDeltaX + s*nxA - t*nxB;
    	  double dDy = dDeltaY + s*nyA - t*nyB;
    	  double dDSqr = dDx * dDx + dDy * dDy;
    	  if (dDSqr < dSigma*dSigma) {
    		  double dDij = sqrt(dDSqr);
    		  double dDVij;
    		  //double dAlpha;
    		  if (ePot == HARMONIC)	{
		    dDVij = (1.0 - dDij / dSigma) / dSigma;
		    //dAlpha = 2.0;
    		  }
    		  else if (ePot == HERTZIAN) {
		    dDVij = (1.0 - dDij / dSigma) * sqrt(1.0 - dDij / dSigma) / dSigma;
		    //dAlpha = 2.5;
    		  }
    		  double dPfx = dDx * dDVij / dDij;
    		  double dPfy = dDy * dDVij / dDij;
    		  dFx[thid] += dPfx;
    		  dFy[thid] += dPfy;
    		  //double dCx = s*nxA - 0.5*dDx;
    		  //double dCy = s*nyA - 0.5*dDy;
    		  double dCx = s*nxA;
    		  double dCy = s*nyA;
    		  dFt[thid] += dCx * dPfy - dCy * dPfx;
    	  }
      }
      //if (thid == 0) {
      //	printf("Forces calculated on block %d\n", blockIdx.x);
      //}

      if (spi == 0) {
	//printf("Thread %d on block %d summing forces\n", thid, blockIdx.x);
	dFx[thid] += dFx[thid + 2];
	dFy[thid] += dFy[thid + 2];
	dFt[thid] += dFt[thid + 2];
	if (spj == 0) {
	  //printf("Thread %d on block %d summing forces\n", thid, blockIdx.x);
	  dFx[thid] += dFx[thid + 1];
	  dFy[thid] += dFy[thid + 1];
	  dFt[thid] += dFt[thid + 1];
	    
	  double dArea = pdArea[nPID];
	  double dMOI = pdMOI[nPID];
	  dFy[thid] /= (dKd*dArea);
	  dFx[thid] = dFx[thid] / (dKd*dArea) - dGamma*dFy[thid];
	  dFt[thid] = dFt[thid] / (dKd*dArea*dMOI);
	    
	  double dFy0 = pdFy[nPID] / (dKd*dArea);
	  double dFx0 = pdFx[nPID] / (dKd*dArea) - dGamma * dFy0;
	  //double dPhi0 = pdPhi[nPID];
	  double dFt0 = pdFt[nPID] / (dKd*dArea*dMOI);
	    
	  double dDx = 0.5 * dStep * (dRsRate * (dX + pdX[nPID]) + dFx0 + dFx[thid]);
	  double dDy = 0.5 * dStep * (dRsRate * (dY + pdY[nPID]) + dFy0 + dFy[thid]);
	  pdX[nPID] += dDx;
	  pdY[nPID] += dDy;
	  pdPhi[nPID] += 0.5 * dStep * (dFt0 + dFt[thid]);
	    
	  pdXMoved[nPID] += dDx;
	  pdYMoved[nPID] += dDy;
	  if (fabs(pdXMoved[nPID]) > 0.5*dEpsilon || fabs(pdYMoved[nPID]) > 0.5*dEpsilon)
	    *bNewNbrs = 1;
	}
      }

      nPID += nThreads/4;
    }
  //if (thid == 0) {
  // printf("Exiting block %d\n", blockIdx.x);
  //}
}



void Cross_Box::resize_step(double dRsRate, long unsigned int nTime, bool bSvStress, bool bSvPos) {
   
  if (bSvStress) {
    cudaMemset((void *) d_pfSE, 0, 5*sizeof(float));

    switch (m_ePotential) {
    case HARMONIC:
      rs_euler_est <HARMONIC, 1> <<<4*m_nGridSize, m_nBlockSize, m_nSM_CalcSE+m_nSM_CalcF>>>
	(m_nCross, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, dRsRate, m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, 
	 d_pdAx, d_pdAy, m_dKd, d_pdArea, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
      break;
    case HERTZIAN:
      rs_euler_est <HERTZIAN, 1> <<<4*m_nGridSize, m_nBlockSize, m_nSM_CalcSE+m_nSM_CalcF>>>
	(m_nCross, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, dRsRate, m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, 
	 d_pdAx, d_pdAy, m_dKd, d_pdArea, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
    }
    cudaThreadSynchronize();
    checkCudaError("Estimating new particle positions, calculating stresses");

    cudaMemcpyAsync(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
    /*
      cudaMemcpy(h_pdFx, d_pdFx, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_pdFy, d_pdFy, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_pdFt, d_pdFt, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
      for (int p = 0; p < 10; p++) {
    	  printf("%d: %g %g %g\n", p, h_pdFx[p], h_pdFy[p], h_pdFt[p]);
      }
    */
    if (bSvPos) {
      cudaMemcpyAsync(h_pdX, d_pdX, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdY, d_pdY, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdPhi, d_pdPhi, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
    }
    cudaThreadSynchronize();
  }
  else {
    switch (m_ePotential) {
    case HARMONIC:
      rs_euler_est <HARMONIC, 0> <<<4*m_nGridSize, m_nBlockSize, m_nSM_CalcF>>>
	(m_nCross, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, dRsRate, m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, 
	 d_pdAx, d_pdAy, m_dKd, d_pdArea, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
      break;
    case HERTZIAN:
      rs_euler_est <HERTZIAN, 0> <<<4*m_nGridSize, m_nBlockSize, m_nSM_CalcF>>>
	(m_nCross, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, dRsRate, m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, 
	 d_pdAx, d_pdAy, m_dKd, d_pdArea, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
    }
    cudaDeviceSynchronize();
    checkCudaError("Estimating new particle positions");
  }

  switch (m_ePotential) {
  case HARMONIC:
    rs_heun_corr <HARMONIC> <<<4*m_nGridSize, m_nBlockSize, m_nSM_CalcF>>>
      (m_nCross, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, dRsRate, m_dStep, d_pdX, d_pdY, d_pdPhi, 
       d_pdR, d_pdAx, d_pdAy, m_dKd, d_pdArea, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pdTempX, 
       d_pdTempY, d_pdTempPhi, d_pdXMoved, d_pdYMoved, m_dEpsilon, d_bNewNbrs);
    break;
  case HERTZIAN:
    rs_heun_corr <HERTZIAN> <<<4*m_nGridSize, m_nBlockSize, m_nSM_CalcF>>>
      (m_nCross, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, dRsRate, m_dStep, d_pdX, d_pdY, d_pdPhi, 
       d_pdR, d_pdAx, d_pdAy, m_dKd, d_pdArea, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pdTempX, 
       d_pdTempY, d_pdTempPhi, d_pdXMoved, d_pdYMoved, m_dEpsilon, d_bNewNbrs);
  }

  if (bSvStress) {
    m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
    fprintf(m_pOutfSE, "%lu %.9g %.9g %.9g %.9g %.9g %.9g\n", 
	    nTime, *m_pfEnergy, m_fP, *m_pfPxx, *m_pfPxy, *m_pfPyx, *m_pfPyy);
    if (bSvPos)
      save_positions(nTime);
  }

  cudaDeviceSynchronize();
  checkCudaError("Updating estimates, moving particles");
  
  cudaMemcpyAsync(h_bNewNbrs, d_bNewNbrs, sizeof(int), cudaMemcpyDeviceToHost);

  m_dL += m_dStep*dRsRate*m_dL;
  cudaDeviceSynchronize();

 if (*h_bNewNbrs) {
   reconfigure_cells();
   find_neighbors();
   //printf("Updating neighbor list at time %ld\n", tTime);
 }
}

/////////////////////////
//
//  TODO:  
//
////////////////////
void Cross_Box::resize_box(double dResizeRate, double dFinalPacking, double dSvStressRate, double dSvPosRate)
{
  if (dResizeRate == 0.0) {
    fprintf(stderr, "Cannot resize with zero rate\n");
    exit(1);
  }
  else if (copysign(1.0,dResizeRate) != copysign(1.0,m_dPacking - dFinalPacking)) {
    fprintf(stderr, "Resize rate must have same sign as FinalPacking - InitialPacking\n");
    exit(1);
  }

  printf("Beginnig resize run with resize rate: %g and step %g\n", dResizeRate, m_dStep);
  fflush(stdout);

  if (dSvStressRate < fabs(dResizeRate) * m_dStep)
    dSvStressRate = fabs(dResizeRate) * m_dStep;
  if (dSvPosRate < fabs(dResizeRate))
    dSvPosRate = fabs(dResizeRate);

  // +0.5 to cast to nearest integer rather than rounding down
  unsigned long int nTime = 0;
  unsigned long int nTotalStep = 0;
  unsigned int nIntStep = (unsigned int)(1.0 / m_dStep + 0.5);
  unsigned int nSvStressInterval = (unsigned int)(dSvStressRate / (fabs(dResizeRate) * m_dStep) + 0.5);
  unsigned int nSvPosInterval = (unsigned int)(dSvPosRate / fabs(dResizeRate) + 0.5);
  //unsigned int nReorderInterval = (unsigned int)(1.0 / m_dStrainRate + 0.5);
  
  printf("Resize run configured\n");
  printf("Int step: %lu\n", nIntStep);
  printf("Stress save int: %lu, Pos save int: %lu\n", nSvStressInterval, nSvPosInterval);
  fflush(stdout);



  char szBuf[200];
  sprintf(szBuf, "%s/%s", m_szDataDir, m_szFileSE);
  const char *szPathSE = szBuf;
  m_pOutfSE = fopen(szPathSE, "w");
  if (m_pOutfSE == NULL) {
    fprintf(stderr, "Could not open file for writing");
    exit(1);
  }

  // Run strain for specified number of steps
  while (copysign(1.0, dResizeRate)*(m_dPacking - dFinalPacking) > 0)
    {
      bool bSvPos = (nTime % nSvPosInterval == 0);
      if (bSvPos) {
	resize_step(dResizeRate, nTime, 1, 1);
	fflush(m_pOutfSE);
      }
      else
	{
	  bool bSvStress = (nTotalStep % nSvStressInterval == 0);
	  resize_step(dResizeRate, nTime, bSvStress, 0);
	}
      nTotalStep += 1;
      for (unsigned int nI = 1; nI < nIntStep; nI++)
	{
	  bool bSvStress = (nTotalStep % nSvStressInterval == 0); 
	  resize_step(dResizeRate, nTime, bSvStress, 0);
	  nTotalStep += 1;
	}
      nTime += 1;
      m_dPacking = calculate_packing();
      reconfigure_cells();
      if (*h_bNewNbrs) {
	find_neighbors();
      }
      //if (nTime % nReorderInterval == 0)
      //reorder_particles();
    }
  
  // Save final configuration
  calculate_stress_energy();
  cudaMemcpyAsync(h_pdX, d_pdX, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdY, d_pdY, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdPhi, d_pdPhi, m_nCross*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
  m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
  fprintf(m_pOutfSE, "%lu %.9g %.9g %.9g %.9g %.9g %.9g\n", 
	  nTime, *m_pfEnergy, m_fP, *m_pfPxx, *m_pfPxy, *m_pfPyx, *m_pfPyy);
  fflush(m_pOutfSE);
  save_positions(nTime);
  
  fclose(m_pOutfSE);
}
