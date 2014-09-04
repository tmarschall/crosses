/*  cross_box.h
 *
 *  Arrays with the prefix "h_" are allocated on the host (CPU), 
 *   those with the prefix "d_" are allocated on the device (GPU).
 *  Anything starting with "g_" is solely used in "gold" functions 
 *   on the cpu (for error checking) and will only be compiled if 
 *   the environmental variable GOLD_FUNCS=1
 *  
 */

#ifndef GOLD_FUNCS
#define GOLD_FUNCS 0
#endif

#ifndef SPHEROCYL_BOX_H
#define SPHEROCYL_BOX_H

#include <stdio.h>
#include <string.h>

enum Potential {HARMONIC = 2, HERTZIAN = 5};

class Cross_Box
{
 private:
  int m_nCross;
  double m_dL;  // Box side length
  double m_dPacking;  // Packing fraction
  double m_dGamma;   // Strain parameter
  double m_dTotalGamma;
  Potential m_ePotential;  // Soft core interaction (harmonic or hertzian)
  double m_dStrainRate;
  double m_dStep;

  // file output data
  FILE *m_pOutfSE;  // output steam
  const char *m_szDataDir;  // directory to save data (particle configurations)
  const char *m_szFileSE;  // filename for stress energy data

  // Coordinates etc. for cross
  double m_dRMax;  // Largest radius
  double m_dAMax;  // Largest cross half-shaft
  // Position and orientation arrays to be allocated on the CPU (host)
  double *h_pdX;
  double *h_pdY;
  double *h_pdPhi;
  double *h_pdR;  // Radii
  double *h_pdAx;
  double *h_pdAy;
  int *h_pnMemID;  // IDs of particles reordered in memory
  int *h_bNewNbrs;
  // Position and orientation arrays to be allocated on the GPU (device)
  double *d_pdX;
  double *d_pdY;
  double *d_pdPhi;
  double *d_pdR;
  double *d_pdAx;
  double *d_pdAy;
  double *d_pdMOI;  // Moment of Inertia
  double *d_pdIsoC;  // Coefficient for isolated rotation function
  bool m_bMOI;
  double *d_pdTempX;  // Temporary positions used in several routines
  double *d_pdTempY;  //  only needed on the device
  double *d_pdTempPhi;
  double *d_pdXMoved;  // Amount each crossinder moved since last finding neighbors
  double *d_pdYMoved;
  int *d_pnInitID;  // The original ID of the particle
  int *d_pnMemID;  // ID of the current location of the particle in memory
  int *d_bNewNbrs;  // Set to 1 when particle moves more than dEpsilon
  // Arrays for "gold" routines (for testing)
#if GOLD_FUNCS == 1
  double *g_pdX;
  double *g_pdY;
  double *g_pdPhi;
  double *g_pdR;
  double *g_pdAx;
  double *g_pdAy;
  double *g_pdTempX;
  double *g_pdTempY;
  double *g_pdTempPhi;
  int *g_pnInitID;
  int *g_pnMemID;
  double *g_pdXMoved;
  double *g_pdYMoved;
  int *g_bNewNbrs;
#endif

  // Stresses and forces
  float *m_pfEnergy;
  float *m_pfPxx;
  float *m_pfPyy;
  float *m_pfPxy;
  float m_fP;  // Total pressure
  float *h_pfSE;  // Array for transfering the stresses and energy
  double *h_pdFx;
  double *h_pdFy;
  double *h_pdFt;
  // GPU
  float *d_pfSE;
  double *d_pdFx;
  double *d_pdFy;
  double *d_pdFt;
#if GOLD_FUNCS == 1
  float *g_pfSE;
  double *g_pdFx;
  double *g_pdFy;
  double *g_pdFt;
#endif
  
  // These variables are used for spatial subdivision and
  // neighbor lists for faster contact detection etc.
  int m_nCells; 
  int m_nCellRows;
  int m_nCellCols;
  double m_dCellW;    // Cell width
  double m_dCellH;    // Cell height
  int *h_pnCellID;    // Cell ID for each particle
  int *h_pnAdjCells;  // Which cells are next to each other
  int m_nMaxPPC;      // Max particles per cell
  int *h_pnPPC;       // Number of particles in each cell
  int *h_pnCellList;  // List of particles in each cell
  int m_nMaxNbrs;     // Max neighbors a particle can have
  int *h_pnNPP;       // Number of neighbors (possible contacts) of each particle
  int *h_pnNbrList;   // List of each particle's neighbors
  // GPU arrays
  int *d_pnCellID;
  int *d_pnAdjCells;
  int *d_pnPPC;
  int *d_pnCellList;
  int *d_pnNPP;
  int *d_pnNbrList;
#if GOLD_FUNCS == 1
  int *g_pnCellID;
  int *g_pnAdjCells;
  int *g_pnPPC;
  int *g_pnCellList;
  int *g_pnNPP;
  int *g_pnNbrList;
#endif

  // Used to not have to update neighbors every step when things are moving
  double m_dEpsilon;

  // These variables define configurations for kernel cuda kernel launches
  int m_nGridSize;  // Grid size (# of thread blocks) for finding cell IDs
  int m_nBlockSize;  // Block size (# threads per block)
  int m_nSM_CalcF;
  int m_nSM_CalcSE;  // Shared memory per block
  int m_nDeviceMem;
  
  void construct_defaults();
  void configure_cells();
  void set_kernel_configs();
  double calculate_packing();
  
  void save_positions(long unsigned int nTime);
  void strain_step(long unsigned int nTime, bool bSvStress = 0, bool bSvPos = 0);

 public:
  Cross_Box(int nCross, double dL, double dR, double dAx, double dAy,
		  	double dEpsilon = 0.1,  int nMaxPPC = 10, int nMaxNbrs = 20,
		  	Potential ePotential = HARMONIC);
  Cross_Box(int nCross, double dL, double *pdX, double *pdY, double *pdPhi,
		  	double *pdR, double *pdAx, double *pdAy, double dEpsilon = 0.1,
		  	int nMaxPPC = 10, int nMaxNbrs = 20, Potential ePotential = HARMONIC);
  ~Cross_Box();

  void place_random_cross(int seed = 0, bool bRandAngle = 1);
  void place_random_0e_cross(int seed = 0, bool bRandAngle = 1);
  void find_neighbors();
  void set_back_gamma();
  void reorder_particles();
  void reset_IDs();
  void calculate_stress_energy();
  bool check_for_contacts();
  bool check_for_contacts(int nIndex);
  bool check_for_intersection(int nIndex, double dEpsilon = 1e-5);
  void run_strain(double dStartGam, double dStopGam, double dSvStressGam, double dSvPosGam);
  void run_strain(long unsigned int nSteps);
  
#if GOLD_FUNCS == 1
  void calculate_stress_energy_gold();
#endif

  void display(bool bParticles = 1, bool bCells = 1, bool bNbrs = 1, bool bStress = 1);

  void set_gamma(double dGamma) { m_dGamma = dGamma; }
  void set_total_gamma(double dTotalGamma) { m_dTotalGamma = dTotalGamma; }
  void set_step(double dStep) { m_dStep = dStep; }
  void set_strain(double dStrain) { m_dStrainRate = dStrain; }
  void set_data_dir(const char *szDataDir) { m_szDataDir = szDataDir; }
  void set_se_file(const char *szFileSE) { m_szFileSE = szFileSE; }
};

#endif
