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

  double dL;
  double *dX = new double[nCross];
  double *dY = new double[nCross];
  double *dPhi = new double[nCross];
  double *dRad = new double[nCross];
  double *dA = new double[nCross];
  double dRMax = 0.0;
  double dAMax = 0.0;
  double dGamma;
  double dTotalGamma;
  long unsigned int nTime = 0;
  double dPacking;
  if (strFile == "r")
  {
    double dPacking = float_input(argc, argv, ++argn, "Packing Fraction");
    cout << dPacking << endl;
    double dA = float_input(argc, argv, ++argn, "Half-shaft length");
    cout << dA << endl;
    double dR = float_input(argc, argv, ++argn, "Radius");
    cout << dR << endl;
    double dArea = nCross*(4*dA + D_PI * dR)*dR;
    dL = sqrt(dArea / dPacking);
    cout << "Box length L: " << dL << endl;
    dRMax = dR;
    dAMax = dA;
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
    cData.getColumn(dX, 0);
    cData.getColumn(dY, 1);
    cData.getColumn(dPhi, 2);
    cData.getColumn(dRad, 3);
    cData.getColumn(dA, 4);
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
    for (int p = 0; p < nCross; p++) {
    	if (dA[p] > dAMax) { dAMax = dA[p]; }
    	if (dRad[p] > dRMax) { dRMax = dRad[p]; }
    }
  }
  
  int tStart = time(0);

  Cross_Box *cCross;
  if (strFile == "r") {
    cout << "Initializing box of length " << dL << " with " << nCross << " particles.";
    cCross = new Cross_Box(nCross, dL, dRMax, dAMax, dDR);
  }
  else {
    cout << "Initializing box from file of length " << dL << " with " << nCross << " particles.";
    cCross = new Cross_Box(nCross, dL, dX, dY, dPhi, dRad, dA, dDR);
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

  delete[] dX; delete[] dY; delete[] dPhi; 
  delete[] dRad; delete[] dA;

  return 0;
}
