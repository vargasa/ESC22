// C++ standard headers
#include <cassert>
#include <iostream>
#include <vector>

// CUDA headers
#include <cuda_runtime.h>

// local headers
#include "cuda_check.h"

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0

__global__ void sum(const int* a, const int* b, int* c){
  c[threadIdx.x ]
}

///////////////////////////////////////////////////////////////////////////////
// Program main
///////////////////////////////////////////////////////////////////////////////
int main()
{
  // Choose one CUDA device
  CUDA_CHECK(cudaSetDevice(MYDEVICE));

  // Create a CUDA stream to execute asynchronous operations on this device
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Pointer and dimension for host memory
  size_t dimA = 8;
  std::vector<float> h_a(dimA);
  //std::vector<float> h_b(dimA);
  std::vector<float> h_c(dimA);

  // Allocate and initialize host memory
  for (uint i = 0; i < dimA; ++i) {
     h_a[i] = i;
     //h_b[i] = 2*i;
  }

  // Pointers for device memory
  float *d_a, *d_b, *d_c;

  // Part 1 of 5: allocate the device memory
  size_t memSize = dimA * sizeof(float);
  //size_t floatSize = sizeof(float);

  cudaMalloc(&d_a, memSize);
  cudaMalloc(&d_b, memSize);

  // CUDA_CHECK(cudaMallocAsync(___));
  // CUDA_CHECK(cudaMallocAsync(___));

  // Part 2 of 5: host to device memory copy
  // Hint: the raw pointer to the underlying array of a vector
  // can be obtained by calling std::vector<T>::data()
  cudaMemcpy(d_a, h_a.data(), memSize, cudaMemcpyHostToDevice);
  
  //CUDA_CHECK(cudaMemcpyAsync(___));

  // Part 3 of 5: device to device memory copy
  //CUDA_CHECK(cudaMemcpyAsync(___));
  cudaMemcpy(d_b, d_a, memSize, cudaMemcpyDeviceToDevice);

  // Clear the host memory
  //std::fill(h_a.begin(), h_a.end(), 0);

  // Part 4 of 5: device to host copy
  cudaMemcpy(h_c.data(),d_c, memSize, cudaMemcpyDeviceToHost);
  //CUDA_CHECK(cudaMemcpyAsync(___));

  // Wait for all asynchronous operations to complete
  //CUDA_CHECK(cudaStreamSynchronize(stream));
  sum<<<1, dimA, 0>>>(d_a,d_b,d_c);

  // Part 5 of 5: free the device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  //CUDA_CHECK(cudaFreeAsync(___));
  //CUDA_CHECK(cudaFreeAsync(___));

  // Verify the data on the host is correct
  for (int i = 0; i < dimA; ++i) {
    std::cout << h_c[i] << '\n';
    //assert(h_a[i] == (float)i);
  }

  // Destroy the CUDA stream
  CUDA_CHECK(cudaStreamDestroy(stream));

  // If the program makes it this far, then the results are correct and
  // there are no run-time errors.  Good work!
  std::cout << "Correct!" << std::endl;

  return 0;
}
