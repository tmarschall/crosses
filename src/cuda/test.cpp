#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace std;

int main()
{
  double a,b;
  srand(time(0));

  clock_t start = clock();
  for (int i = 0; i < 1000000; i++)
    {  
      a = (double)rand()/(double)RAND_MAX;
      b = -a;

      cout << a << " " << a ^ -0.0 << " " << b << " " << b ^ -0.0 << endl;

      /*double c = a - b;
      //c += 100.0 * ((c <= -50.0) - (c >= 50.0));
      if (c >= 50.0)
	c -= 100.0;
      else if (c <= -50.0)
        c += 100.0;
      
	cout << a << " " << b << " " << c << endl;*/
    }
  clock_t stop = clock();

  cout << "\nRun time: " << stop - start << endl;

  return 0;
}
