//
// compile with
//  c++ -O2 -Wall -fopt-info-vec -march=native matmul.cpp
//  change -O2 in -Ofast
//  add -funroll-loops
//
//  change order of loops
//  change N (x2)
//  change float to double
//  change mult with div (or even /sqrt )
//

#ifndef FLOAT
#define FLOAT float
#warning "using float"
#endif


void mmult(FLOAT const * a, FLOAT const * b, FLOAT * c, int N) {

  for ( int i = 0; i < N; ++i ) { 
    for ( int j = 0; j < N; ++j ) { 
      for ( int k = 0; k < N; ++k ) { 
	c[ i * N + j ]  +=   a[ i * N + k ]  *  b[ k * N + j ]; 
      } 
    } 
  }
  
}



void init(FLOAT * x, int N, FLOAT y) {
  for ( int i = 0; i < N; ++i ) x[i]=y;
}


FLOAT * alloc(int N) {
  return new FLOAT[N];
  
}


#include <chrono>
#include <array>
#include <iostream>
#include "benchmark.h"


int main() {
  using namespace std;
  auto start = chrono::high_resolution_clock::now();


  int N = 1000;
  
  int size = N*N;
  FLOAT * a = alloc(size);
  FLOAT * b = alloc(size);
  FLOAT * c = alloc(size);
  
  init(c,size,0.f);
  init(a,size,1.3458f);
  init(b,size,2.467f);

  auto delta = start - start;
  benchmark::touch(a);
  benchmark::touch(b);
  delta -= (chrono::high_resolution_clock::now()-start);
  for (int k=0; k<10; ++k) {
    mmult(a,b,c,N);
    benchmark::keep(c);
  }
  delta += (chrono::high_resolution_clock::now()-start);
  std::cout << " Computation took "
              << chrono::duration_cast<chrono::milliseconds>(delta).count()
              << " ms" << std::endl;

  return c[N];
  
}

