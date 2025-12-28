#include <array>
#include <iostream>

// For conciseness, assume int is 32-bit
using u32 = unsigned int;

__global__ void matmul_by_row(const float *pA, const float *pB, float *pC,
                              const int M, const int K, const int N) {
  // MxK @ KxN
  const u32 row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M) {
    // Col of the B matrix
    for (u32 col = 0; col < N; ++col) {
      float acc = 0.0f;
      for (u32 k = 0; k < K; ++k) {
        acc += pA[row * K + k] * pB[k * N + col];
      }
      pC[row * N + col] = acc;
    }
  }
}

int main() {
  constexpr int M = 4;
  constexpr int K = 4;
  constexpr int N = 4;

  constexpr int blockSize = 256;
  constexpr int gridSize = (M + blockSize - 1) / blockSize;

  std::array<float, M * K> A{}; // Zero-initialize all elements
  std::array<float, K * N> B{};
  std::array<float, M * N> C{};
  // With A defined this way, A@B = C will be B flipped
  for (int row = 0; row < std::min(M, K); ++row) {
    A[row * K + K - 1 - row] = 1.0f;
  }
  for (int i = 0; i < B.size(); ++i) {
    B[i] = i;
  }
  // Allocate and copy to device
  float *pAdev, *pBdev, *pCdev;
  cudaMalloc(&pAdev, A.size() * sizeof(float));
  cudaMalloc(&pBdev, B.size() * sizeof(float));
  cudaMalloc(&pCdev, C.size() * sizeof(float));
  cudaMemcpyAsync(pAdev, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(pBdev, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);
  matmul_by_row<<<gridSize, blockSize>>>(pAdev, pBdev, pCdev, M, K, N);
  cudaMemcpy(C.data(), pCdev, C.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(pAdev);
  cudaFree(pBdev);
  cudaFree(pCdev);
  // Print first row of C
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      std::cout << C[row * N + col] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  return 0;
}
