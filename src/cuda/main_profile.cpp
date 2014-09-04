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

int main()
{
  cout << "Setting parameters testing... \n";
  int nCross = 16384;
  cout << "Number of particles = " << nCross << endl;
  double dPacking = 0.89;
  cout << "Packing fraction =  " << dPacking << endl;
  double dL = sqrt(0.5 * nCross * D_PI * (0.5 * 0.5 + 0.7 * 0.7) / dPacking);
  cout << "Box dimension = " << dL << endl;
  double dStrain = 0.0001;
  cout << "Strain rate for shearing = " << dStrain << endl;
  double dStep = 0.5;
  cout << "Step size for shearing = " << dStep << endl;
  double dEpsilon = 0.08;
  cout << "Neighbor movement buffer = " << dEpsilon << "\n\n";

  double *dX = new double[nCross];
  double *dY = new double[nCross];
  double *dR = new double[nCross];
  srand(time(0));
  for (int p = 0; p < nCross; p++)
    {
      dX[p] = dL * static_cast<double>(rand()) / RAND_MAX;
      dY[p] = dL * static_cast<double>(rand()) / RAND_MAX;
      dR[p] = 0.5 + (p % 2) * 0.2;
    }
  
  cout << "Testing routines... " << endl;
  Cross_Box cTest(nCross, dL, dX, dY, dR, dEpsilon);
  cTest.set_gamma(-0.4999999999);
  cTest.set_total_gamma(0.0);
  cTest.set_strain(dStrain);
  cTest.set_step(dStep);

  cout << "Finding neighbors... " << endl;
  cTest.find_neighbors();
  //cout << "Calculating stresses... " << endl;
  //cTest.calculate_stress_energy();
  //cout << "Running strain step... " << endl;
  //cTest.run_strain(0, dStrain, dStrain*dStep, dStrain);
  //cout << "Setting back shear coordinates... " << endl;
  //cTest.set_gamma(0.5000000001);
  //cTest.set_back_gamma();
  
  cout << "Reordering particles... " << endl;
  cTest.reorder_particles();
  cTest.reset_IDs();
  cout << "Calculating stresses... " << endl;
  cTest.calculate_stress_energy();
  cout << "Running strain step... " << endl;
  cTest.run_strain(0, dStrain, dStrain*dStep, dStrain);
  cout << "Setting back shear coordinates... " << endl;
  cTest.set_gamma(0.5000000001);
  cTest.set_back_gamma();

  delete[] dX; delete[] dY; delete[] dR;
  return 0;
}
