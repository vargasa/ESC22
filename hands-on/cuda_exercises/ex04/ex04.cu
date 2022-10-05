// C++ standard headers
#include <iostream>
#include <vector>

// CUDA headers
#include <cuda_runtime.h>

// local headers
#include "cuda_check.h"

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0

__global__ void saxpy(unsigned int n, double a, double const* __restrict__ x, double* __restrict__ y)
{
  unsigned int const thread = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int const stride = blockDim.x * gridDim.x;
  for (auto i = thread; i < n; i += stride)
    y[i] = a * x[i] + y[i];
}

int main(void)
{
  CUDA_CHECK(cudaSetDevice(MYDEVICE));

  cudaStream_t queue;
  CUDA_CHECK(cudaStreamCreate(&queue));

  // 1<<N is equivalent to 2^N
  unsigned int N = 20 * (1 << 20);
  double *d_x, *d_y;
  std::vector<double> x(N, 1.);
  std::vector<double> y(N, 2.);

  CUDA_CHECK(cudaMallocAsync(&d_x, N * sizeof(double), queue));
  CUDA_CHECK(cudaMallocAsync(&d_y, N * sizeof(double), queue));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaMemcpyAsync(d_x, x.data(), N * sizeof(double), cudaMemcpyHostToDevice, queue));
  CUDA_CHECK(cudaMemcpyAsync(d_y, y.data(), N * sizeof(double), cudaMemcpyHostToDevice, queue));

  CUDA_CHECK(cudaEventRecord(start, queue));

  int threadsPerBlock = 512;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  saxpy<<<blocksPerGrid, threadsPerBlock, 0, queue>>>(N, 2.0, d_x, d_y);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(stop, queue));

  CUDA_CHECK(cudaMemcpyAsync(y.data(), d_y, N * sizeof(double), cudaMemcpyDeviceToHost, queue));

  CUDA_CHECK(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

  double maxError = 0.;
  for (unsigned int i = 0; i < N; i++) {
    maxError = max(maxError, abs(y[i] - 4.0));
  }

  CUDA_CHECK(cudaFreeAsync(d_x, queue));
  CUDA_CHECK(cudaFreeAsync(d_y, queue));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(queue));
}
