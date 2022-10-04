---
title: Associative sum
layout: main
section: floatingpoints
---
File: `hands-on/floatingpoints/commutative_sum.cc`
In this simple example we:

1. fill one array with a random uniform distribution of floating-point values, 
2. copy this array in a second one,
3. sort the values in the second array (but not in the first array),
4. sum togheter all the values of the first array  
4. sum togheter all the values of the second array
5. given that the sum is associative, we expect the two sums to be the same

       #include<algorithm>
       #include<cmath>
       #include<random>
       #include<vector>
       #include <quadmath.h>  
       int main() {

          int seed = time(NULL);


          // Create vector v1 and fill it with a random uniform distribution:
          std::vector<double> v1;
          std::uniform_real_distribution<double> unif(-1.,1.);
          //std::uniform_real_distribution<double> unif(-100000000.,100000000.);
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

Implement a standard sum:

1. Is the sum commutative? 
2. What if the random values are in an interval larger than [-1,1]?
3. Try with different precisions
   - You can use `__float128` (from quadmath) to compare your results
     - Uncomment `#include <quadmath.h>` and compile with `-lquadmath`
     - You can perform the sum in this precision and cast back the result to a different precision
4. Implement the compensated sum
5. Compare the results
  

