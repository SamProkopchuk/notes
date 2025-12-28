#include <array>
#include <iostream>

// For conciseness, assume int is 32-bit
using u32 = unsigned int;

__global__ void matvecmul_by_row(const float *pA, const float *px, float *py,
                                 const int M, const int N) {
  // MxN @ Nx1
  const u32 row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M) {
    float acc = 0.0f;
    // Col of the A matrix
    for (u32 col = 0; col < N; ++col) {
      acc += pA[row * N + col] * px[col];
    }
    py[row] = acc;
  }
}

int main() {
  constexpr int M = 4;
  constexpr int N = 4;

  constexpr int blockSize = 256;
  constexpr int gridSize = (M + blockSize - 1) / blockSize;

  std::array<float, M * N> A{}; // Zero-initialize all elements
  std::array<float, N> x{};
  std::array<float, M> y{};
  // With A defined this way, A@x = y will contain the sum of each row of A
  for (int i = 0; i < A.size(); ++i) {
    A[i] = static_cast<float>(i);
  }
  for (int i = 0; i < x.size(); ++i) {
    x[i] = 1.0f;
  }
  // Allocate and copy to device
  float *pAdev, *pxdev, *pydev;
  cudaMalloc(&pAdev, A.size() * sizeof(float));
  cudaMalloc(&pxdev, x.size() * sizeof(float));
  cudaMalloc(&pydev, y.size() * sizeof(float));
  cudaMemcpy(pAdev, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(pxdev, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
  matvecmul_by_row<<<gridSize, blockSize>>>(pAdev, pxdev, pydev, M, N);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }
  cudaMemcpy(y.data(), pydev, y.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(pAdev);
  cudaFree(pxdev);
  cudaFree(pydev);
  // Print result
  for (int row = 0; row < M; ++row) {
    std::cout << y[row];
    std::cout << std::endl;
  }
  return 0;
}
