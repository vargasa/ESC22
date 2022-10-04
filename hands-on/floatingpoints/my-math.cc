//gcc -O0 my-math.c -o safe-math
//gcc -O3 my-math.c -o safe-math
//gcc -Ofast my-math.c -o safe-math
//gcc -O3 -funsafe-math-optimizations my-math.c -o unsafe-math

#include <stdio.h>
#include <stdlib.h>
#include <chrono>  
#include <iostream>

typedef float data_t;
int main() {

  data_t sum = 0.0;
  data_t d = 0.0;

  // Stop measuring time
  auto start = std::chrono::high_resolution_clock::now(); 

  for (size_t i=1; i<=10000000; i++) {	
    d = ((data_t)i / 3) + ((data_t)i / 7);
    sum += d;
  }

  // Stop measuring time

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
  printf("Result is: %.20e Computed in %.03ld milliseconds\n", sum, duration.count());
 
  return 0;

}

