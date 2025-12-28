#include <array>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

// For conciseness, assume char is 8-bit and int is 32-bit
using u32 = unsigned int;
using u8 = unsigned char;

// Stuff to read/write PPM images

struct Image {
  Image() = default;
  Image(const int width, const int height)
      : width(width), height(height), data(width * height * 3) {}
  int width, height;
  std::vector<u8> data; // RGB interleaved
};

Image readPPM(const std::filesystem::path &filepath) {
  std::ifstream file(filepath);
  if (!file) {
    throw std::runtime_error("Cannot open file: " + filepath.string());
  }

  std::string magic;
  int width, height, maxval;
  file >> magic >> width >> height >> maxval;

  if (magic != "P3" && magic != "P6") {
    throw std::runtime_error("Unsupported PPM format: " + magic);
  }

  Image img;
  img.width = width;
  img.height = height;
  img.data.resize(width * height * 3);

  if (magic == "P3") {
    // ASCII format
    for (int i = 0; i < width * height * 3; ++i) {
      int val;
      file >> val;
      img.data[i] = static_cast<u8>(val);
    }
  } else {
    // Binary format (P6)
    file.get(); // consume the single whitespace after maxval
    file.read(reinterpret_cast<char *>(img.data.data()), img.data.size());
  }

  return img;
}

void writePPM(const std::filesystem::path &filepath, const Image &img) {
  std::ofstream file(filepath);
  if (!file) {
    throw std::runtime_error("Cannot create file: " + filepath.string());
  }

  // Write header
  file << "P3\n";
  file << img.width << " " << img.height << "\n";
  file << "255\n";

  // Write pixel data (ASCII format for readability)
  for (int i = 0; i < img.width * img.height * 3; i += 3) {
    file << static_cast<int>(img.data[i]) << " "
         << static_cast<int>(img.data[i + 1]) << " "
         << static_cast<int>(img.data[i + 2]);
    if ((i / 3 + 1) % img.width == 0) {
      file << "\n";
    } else {
      file << "  ";
    }
  }
}

} // namespace

constexpr int BLUR_RADIUS = 3;

__global__ void gaussian_blur(const u8 *pX, u8 *pY, const int H, const int W,
                              const int C) {
  assert(C == 3);
  const u32 col = blockIdx.x * blockDim.x + threadIdx.x;
  const u32 row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col < W && row < H) {
    const u32 idx = (row * W + col) * C;
    for (int c = 0; c < C; ++c) {
      u32 acc = 0;
      u32 count = 0;
      for (int dc = -BLUR_RADIUS; dc <= BLUR_RADIUS; ++dc) {
        for (int dr = -BLUR_RADIUS; dr <= BLUR_RADIUS; ++dr) {
          const u32 col2 = col + dc;
          const u32 row2 = row + dr;
          if (col2 < W && row2 < H) {
            const u32 idx2 = (row2 * W + col2) * C;
            acc += pX[idx2 + c];
            ++count;
          }
        }
      }
      pY[idx + c] = acc / count;
    }
  }
}

int main() {
  Image img = readPPM("../../assets/image.ppm");
  const int H = img.height;
  const int W = img.width;
  const int C = 3;

  const dim3 blockDim(16, 16);
  const dim3 gridDim((W + blockDim.x - 1) / blockDim.x,
                     (H + blockDim.y - 1) / blockDim.y);
  u8 *pXdev, *pYdev;
  cudaMalloc(&pXdev, img.data.size() * sizeof(u8));
  cudaMalloc(&pYdev, img.data.size() * sizeof(u8));
  cudaMemcpy(pXdev, img.data.data(), img.data.size() * sizeof(u8),
             cudaMemcpyHostToDevice);
  gaussian_blur<<<gridDim, blockDim>>>(pXdev, pYdev, H, W, C);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }
  Image out(W, H);
  cudaMemcpy(out.data.data(), pYdev, out.data.size() * sizeof(u8),
             cudaMemcpyDeviceToHost);
  cudaFree(pXdev);
  cudaFree(pYdev);

  writePPM("../../assets/out.ppm", out);
  return 0;
}
