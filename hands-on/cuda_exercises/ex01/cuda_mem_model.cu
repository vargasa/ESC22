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

__global__ void sum(const float* a, const float* b, float* c){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    c[tid] = a[tid] + b[tid];
}

///////////////////////////////////////////////////////////////////////////////
// Program main
///////////////////////////////////////////////////////////////////////////////
int main()
{
  // Choose one CUDA device
  CUDA_CHECK(cudaSetDevice(MYDEVICE));

  // Create a CUDA stream to execute asynchronous operations on this device
  //cudaStream_t stream;
  //CUDA_CHECK(cudaStreamCreate(&stream));

  // Pointer and dimension for host memory
  size_t dimA = 1024*2;
  std::vector<float> h_a(dimA);
  std::vector<float> h_b(dimA);
  std::vector<float> h_c(dimA);

  // Allocate and initialize host memory
  for (uint i = 0; i < dimA; ++i) {
     h_a[i] = i;
     h_b[i] = 2*i;
  }

  // Pointers for device memory
  float *d_a, *d_b, *d_c;

  // Part 1 of 5: allocate the device memory
  size_t memSize = dimA * sizeof(float);
  //size_t floatSize = sizeof(float);

  cudaMalloc(&d_a, memSize);
  cudaMalloc(&d_b, memSize);
  cudaMalloc(&d_c, memSize);

  // CUDA_CHECK(cudaMallocAsync(___));
  // CUDA_CHECK(cudaMallocAsync(___));

  // Part 2 of 5: host to device memory copy
  // Hint: the raw pointer to the underlying array of a vector
  // can be obtained by calling std::vector<T>::data()
  cudaMemcpy(d_a, h_a.data(), memSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), memSize, cudaMemcpyHostToDevice);
  
  //CUDA_CHECK(cudaMemcpyAsync(___));

  // Part 3 of 5: device to device memory copy
  //CUDA_CHECK(cudaMemcpyAsync(___));
  // cudaMemcpy(d_b, d_a, memSize, cudaMemcpyDeviceToDevice);
  // cudaMemcpy(d_a, d_b, memSize, cudaMemcpyDeviceToDevice);
  
  // Clear the host memory
  //std::fill(h_a.begin(), h_a.end(), 0);

  // Part 4 of 5: device to host copy
  dim3 blocks(2,1,1); // define the number of blocks available as a 3d shape
  //dim3 blocks();
  //blocks.x = 1
  sum<<<blocks /*number of blocks*/, 1024 /*threads within the block*/, 0>>>(d_a,d_b,d_c);
  cudaDeviceSynchronize();
  //cudaStreamSynchronize(stream);
  cudaMemcpy(h_c.data(),d_c, memSize, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  for (int i = 0; i < dimA; ++i) {
    std::cout << h_c[i] << '\n';
    //assert(h_c[i] == 3.*i );
  }


  //CUDA_CHECK(cudaMemcpyAsync(___));

  // Wait for all asynchronous operations to complete
  //CUDA_CHECK(cudaStreamSynchronize(stream));

  // Part 5 of 5: free the device memory
  //CUDA_CHECK(cudaFreeAsync(___));
  //CUDA_CHECK(cudaFreeAsync(___));

  // Verify the data on the host is correct


  // Destroy the CUDA stream
  //CUDA_CHECK(cudaStreamDestroy(stream));

  // If the program makes it this far, then the results are correct and
  // there are no run-time errors.  Good work!
  //std::cout << "Correct!" << std::endl;

  return 0;
}
