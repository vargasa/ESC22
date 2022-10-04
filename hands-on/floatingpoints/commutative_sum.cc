#include <algorithm>
#include <cmath>
#include <quadmath.h>
#include <random>
#include <vector>

int main() {

  int seed = time(NULL);
     // Create vector v1 and fill it with a random uniform distribution:
  std::vector<double>  v1;
  std::uniform_real_distribution<double> unif(-1., 1.);
  // std::uniform_real_distribution<double> unif(-100000000.,100000000.);
  std::mt19937 rng(seed);
  for (size_t i = 0; i < 1000; ++i) {
    v1.push_back(unif(rng));
  }
   // Duplicate v1

  // Sort v2 but not v1 

  // Compute the sums! 

  //Compare the results

  return 0;
   
}
