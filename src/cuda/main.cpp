 /*
 *
 *
 */

#include "cross_box.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include "file_input.h"
#include <string>

using namespace std;

const double D_PI = 3.14159265358979;

int int_input(int argc, char* argv[], int argn, char description[] = "")
{
  if (argc > argn) {
    int input = atoi(argv[argn]);
    cout << description << ": " << input << endl;
    return input;
  }
  else {
    int input;
    cout << description << ": ";
    cin >> input;
    return input;
  }
}
double float_input(int argc, char* argv[], int argn, char description[] = "")
{
  if (argc > argn) {
    double input = atof(argv[argn]);
    cout << description << ": " << input << endl;
    return input;
  }
  else {
    double input;
    cout << description << ": ";
    cin >> input;
    return input;
  }
}
string string_input(int argc, char* argv[], int argn, char description[] = "")
{
  if (argc > argn) {
    string input = argv[argn];
    cout << description << ": " << input << endl;
    return input;
  }
  else {
    string input;
    cout << description << ": ";
    cin >> input;
    return input;
  }
}

int main(int argc, char* argv[])
{
  int argn = 0;
  string strFile = string_input(argc, argv, ++argn, "Data file ('r' for random)");
  const char* szFile = strFile.c_str();
  cout << strFile << endl;
  string strDir = string_input(argc, argv, ++argn, "Output data directory");
  const char* szDir = strDir.c_str();
  cout << strDir << endl;
  int nCross = int_input(argc, argv, ++argn, "Number of particles");
  cout << nCross << endl;
  double dStrainRate = float_input(argc, argv, ++argn, "Strain rate");
  cout << dStrainRate << endl;
  double dStep = float_input(argc, argv, ++argn, "Integration step size");
  cout << dStep << endl;
  double dRunLength = float_input(argc, argv, ++argn, "Strain run length (gamma)");
  cout << dRunLength << endl;
  double dPosSaveRate = float_input(argc, argv, ++argn, "Position data save rate");
  cout << dPosSaveRate << endl;
  double dStressSaveRate = float_input(argc, argv, ++argn, "Stress data save rate");
  cout << dStressSaveRate << endl;
  double dDR = float_input(argc, argv, ++argn, "Cell padding");
  cout << dDR << endl;

  if (dStressSaveRate < dStrainRate * dStep)
    dStressSaveRate = dStrainRate * dStep;
  if (dPosSaveRate < dStrainRate)
    dPosSaveRate = dStrainRate;

  double dLx;
  double dLy;
  double dL;
  double *pdX = new double[nCross];
  double *pdY = new double[nCross];
  double *pdPhi = new double[nCross];
  double *pdR = new double[nCross];
  double *pdAx = new double[nCross];
  double *pdAy = new double[nCross];
  double dR = 0.0;
  double dAx = 0.0;
  double dAy = 0.0;
  double dGamma;
  double dTotalGamma;
  long unsigned int nTime = 0;
  double dPacking;
  double dKd = 1;
  double dBidispersity;
  if (strFile == "r")
  {
    double dPacking = float_input(argc, argv, ++argn, "Packing Fraction");
    cout << dPacking << endl;
    dR = float_input(argc, argv, ++argn, "Radius");
    cout << dR << endl;
    dAx = float_input(argc, argv, ++argn, "Half-shaft length (long axis)");
    cout << dAx << endl;
    dAy = float_input(argc, argv, ++argn, "Half-shaft length (short axis)");
    cout << dAy << endl;
    dBidispersity = float_input(argc, argv, ++ argn, "Size bidispersity ratio");
    cout << dBidispersity << endl;
    double dArea = nCross*(1+dBidispersity)*(2*dAx + 2*dAy + D_PI*dR - 2*dR)*dR;  // Total particle area
    dL = sqrt(dArea / dPacking);
    cout << "Box length L: " << dL << endl;
    dKd = 1./(dR*(4*dAx + 4*dAy + 2*D_PI*dR - 4*dR));  // disspipative constant: 1/A_s
    /*
    srand(time(0) + static_cast<int>(1000*dPacking));
    for (int p = 0; p < nCross; p++)
    {
      dX[p] = dL * static_cast<double>(rand() % 1000000000) / 1000000000.;
      dY[p] = dL * static_cast<double>(rand() % 1000000000) / 1000000000.;
      dPhi[p] = 2.*pi * static_cast<double>(rand() % 1000000000) / 1000000000.;
      dRad[p] = dR;
      dA[p] = dA.; 
    }
    */
    dGamma = 0.;
    dTotalGamma = 0.;
  }
  else
  {
    cout << "Loading file: " << strFile << endl;
    DatFileInput cData(szFile, 1);

    if (cData.getHeadInt(0) != nCross) {
    	cout << "Warning: Number of particles in data file may not match requested number" << endl;
    	cerr << "Warning: Number of particles in data file may not match requested number" << endl;
    }
    cData.getColumn(pdX, 0);
    cData.getColumn(pdY, 1);
    cData.getColumn(pdPhi, 2);
    cData.getColumn(pdR, 3);
    cData.getColumn(pdAx, 4);
    cData.getColumn(pdAy, 5);
    
    double dRmin = pdR[0];
    double dAreaMin = (4*pdAx[0] + 4*pdAy[0] + 2*D_PI*pdR[0] - 4*pdR[0])*pdR[0];
    for  (int i = 1; i < nCross; i++) {
      if (pdR[i] < dRmin) {
	dRmin = pdR[i];
	dAreaMin = (4*pdAx[i] + 4*pdAy[i] + 2*D_PI*pdR[i] - 4*pdR[i])*pdR[i];
      }
    }
    dKd = 1/dAreaMin;
    dL = cData.getHeadFloat(1);
    dPacking = cData.getHeadFloat(2);
    dGamma = cData.getHeadFloat(3);
    dTotalGamma = cData.getHeadFloat(4);
    if (cData.getHeadFloat(5) != dStrainRate) {
      cerr << "Warning: Strain rate in data file does not match the requested rate" << endl;
    }
    if (cData.getHeadFloat(6) != dStep) {
      cerr << "Warning: Integration step size in data file does not match requested step" << endl;
    }
    
    int nFileLen = strFile.length();
    string strNum = strFile.substr(nFileLen-14, 10);
    nTime = atoi(strNum.c_str());
    cout << "Time: " << strNum << " " << nTime << endl;
  }
  
  int tStart = time(0);

  Cross_Box *cCross;
  if (strFile == "r") {
    cout << "Initializing box of length " << dL << " with " << nCross << " particles.";
    cCross = new Cross_Box(nCross, dL, dR, dAx, dAy, dKd, dBidispersity, dDR);
  }
  else {
    cout << "Initializing box from file of length " << dL << " with " << nCross << " particles.";
    cCross = new Cross_Box(nCross, dL, pdX, pdY, pdPhi, pdR, pdAx, pdAy, dKd, dDR);
  }
  cout << "Cross initialized" << endl;

  cCross->set_gamma(dGamma);
  cCross->set_total_gamma(dTotalGamma);
  cCross->set_strain(dStrainRate);
  cCross->set_step(dStep);
  cCross->set_data_dir(szDir);
  cout << "Configuration set" << endl;
  
  //cCross.place_random_cross();
  //cout << "Random cross placed" << endl;
  //cCross.reorder_particles();
  //cCross.reset_IDs();
  //cout << "Cross reordered" << endl;
  cCross->find_neighbors();
  cout << "Neighbor lists created" << endl;
  cCross->calculate_stress_energy();
  cout << "Stresses calculated" << endl;
  cCross->display(1,1,1,1);

  /*
  cCross.find_neighbors();
  cout << "Neighbor lists created" << endl;
  cCross.display(1,0,1,0);
  cCross.reorder_particles();
  cCross.reset_IDs();
  cout << "Cross spatially reordered" << endl;
  cCross.display(1,0,1,0);
  cCross.set_gamma(0.5);
  cCross.set_back_gamma();
  cout << "Gamma reset" << endl;
  cCross.display(1,0,1,0);
  cCross.calculate_stress_energy();
  cout << "Energy and stress tensor calculated" << endl;
  cCross.display(0,0,0,1);
  */

  cCross->run_strain(dTotalGamma, dTotalGamma+dRunLength, dStressSaveRate, dPosSaveRate);
  cCross->display(1,0,0,1);

  int tStop = time(0);
  cout << "\nRun Time: " << tStop - tStart << endl;

  delete[] pdX; delete[] pdY; delete[] pdPhi; 
  delete[] pdR; delete[] pdAx; delete[] pdAy;

  return 0;
}
