/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <iostream>
#include <ctime>
#include <fstream>
#include <chrono>
#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuFloatUtils.cuh"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/float/scan.cu"
using namespace dietgpu;

uint16_t float32ToBFloat16(float f) {
  // FIXME: does not round to nearest even
  static_assert(sizeof(float) == sizeof(uint32_t), "");
  uint32_t x;
  std::memcpy(&x, &f, sizeof(float));

  x >>= 16;
  return x;
}

uint16_t float32ToFloat16(float f) {
  static_assert(sizeof(float) == sizeof(uint32_t), "");
  uint32_t x;
  std::memcpy(&x, &f, sizeof(float));

  uint32_t u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
  uint32_t sign, exponent, mantissa;

  // Get rid of +NaN/-NaN case first.
  if (u > 0x7f800000U) {
    return 0x7fffU;
  }

  sign = ((x >> 16) & 0x8000);

  // Get rid of +Inf/-Inf, +0/-0.
  if (u > 0x477fefffU) {
    return sign | 0x7c00U;
  }
  if (u < 0x33000001U) {
    return (sign | 0x0000);
  }

  exponent = ((u >> 23) & 0xff);
  mantissa = (u & 0x7fffff);

  if (exponent > 0x70) {
    shift = 13;
    exponent -= 0x70;
  } else {
    shift = 0x7e - exponent;
    exponent = 0;
    mantissa |= 0x800000;
  }
  lsb = (1 << shift);
  lsb_s1 = (lsb >> 1);
  lsb_m1 = (lsb - 1);

  // Round to nearest even.
  remainder = (mantissa & lsb_m1);
  mantissa >>= shift;
  if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
    ++mantissa;
    if (!(mantissa & 0x3ff)) {
      ++exponent;
      mantissa = 0;
    }
  }

  return (sign | (exponent << 10) | mantissa);
}

template <FloatType FT>
struct GenerateFloat;

template <>
struct GenerateFloat<FloatType::kFloat16> {
  static FloatTypeInfo<FloatType::kFloat16>::WordT gen(float v) {
    return float32ToFloat16(v);
  }
};

template <>
struct GenerateFloat<FloatType::kBFloat16> {
  static FloatTypeInfo<FloatType::kBFloat16>::WordT gen(float v) {
    return float32ToBFloat16(v);
  }
};

template <>
struct GenerateFloat<FloatType::kFloat32> {
  static FloatTypeInfo<FloatType::kFloat32>::WordT gen(float v) {
    FloatTypeInfo<FloatType::kFloat32>::WordT out;
    std::memcpy(&out, &v, sizeof(float));
    return out;
  }
};

template <>
struct GenerateFloat<FloatType::kFloat64> {
  static FloatTypeInfo<FloatType::kFloat64>::WordT gen(double v) {
    FloatTypeInfo<FloatType::kFloat64>::WordT out;
    // v *= 1e8; // float64 has a larger exponent, so we need to make larger floats
    std::memcpy(&out, &v, sizeof(double));
    return out;
  }
};

template <FloatType FT>
std::vector<typename FloatTypeInfo<FT>::WordT> generateFloats(int num) {
  std::mt19937 gen(10 + num);
  std::normal_distribution<double> dist;
  std::random_device rd;  // Initialize a random seed
  std::default_random_engine generator(rd());

  // Create a uniform distribution for floating-point numbers between 0 and 1
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  auto out = std::vector<typename FloatTypeInfo<FT>::WordT>(num);
  for (auto& v : out) {
    v = GenerateFloat<FT>::gen(dist(gen));
    double random_number = distribution(generator);
    if(random_number < 0.5) {
      v = 0.0;
    }
  }

  return out;
}


template <FloatType FT>
std::vector<typename FloatTypeInfo<FT>::WordT> generateSparseFloats(int num, float sparsityPercent=0.5) {
  std::mt19937 gen(10 + num);
  std::normal_distribution<float> dist;

  std::random_device rd;  // Initialize a random seed
  std::default_random_engine generator(rd());

  // Create a uniform distribution for floating-point numbers between 0 and 1
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  // Generate a random number between 0 and 1
  auto out = std::vector<typename FloatTypeInfo<FT>::WordT>(num);
  for (auto& v : out) {
    v = GenerateFloat<FT>::gen(dist(gen));

    double random_number = distribution(generator);
    if(random_number < sparsityPercent) {
      v = 0.0;
    }
  }

  return out;
}

template <FloatType FT>
void runBatchPointerTest(
    StackDeviceMemory& res,
    int probBits,
    const std::vector<uint32_t>& batchSizes,
    int idx) {
  using FTI = FloatTypeInfo<FT>;

  // run on a different stream to test stream assignment
  auto stream = CudaStream::makeNonBlocking();


  std::ofstream outputFile("/home/ee274_mfguo_nsagan/nsagan/dietgpu_fork/dietgpu/float/benchmark_data/sparse_cuda_nsagan.txt", std::ios::app);
  

  int numInBatch = batchSizes.size();
  
  uint32_t totalSize = 0;
  
  uint32_t maxSize = 0;
  for (auto v : batchSizes) {
    totalSize += v;
    maxSize = std::max(maxSize, v);
  }

  auto origSparse = generateSparseFloats<FT>(totalSize);
  
  auto origSparse_dev = res.copyAlloc<typename FTI::WordT>(stream, origSparse);

  auto start = std::chrono::system_clock::now();

  auto inPtrs = std::vector<const void*>(batchSizes.size());
  {
    uint32_t curOffset = 0;
    for (int i = 0; i < inPtrs.size(); ++i) {
      inPtrs[i] = (const typename FTI::WordT*) origSparse_dev.data() + curOffset;
      curOffset += batchSizes[i];
    }
  }

  auto maxCompressedSize = getMaxSparseFloatCompressedSize(FT, maxSize);
  auto enc_dev = res.alloc<uint8_t>(stream, numInBatch * maxCompressedSize);

  auto encPtrs = std::vector<void*>(batchSizes.size());
  {
    for (int i = 0; i < inPtrs.size(); ++i) {
      encPtrs[i] = (uint8_t*)enc_dev.data() + i * maxCompressedSize;
    }
  }

  auto outBatchSize_dev = res.alloc<uint32_t>(stream, numInBatch);
  auto compConfig =
      FloatCompressConfig(FT, ANSCodecConfig(probBits), false, true);

  // std::cout<<"start floatCompress"<<std::endl;
  floatCompressSparse(
      res,
      compConfig,
      numInBatch,
      inPtrs.data(),
      batchSizes.data(),
      encPtrs.data(),
      outBatchSize_dev.data(),
      stream);

    cudaDeviceSynchronize();
  // std::cout<<"after floatCompress"<<std::endl;

  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end-start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  // Decode data
  auto start2 = std::chrono::system_clock::now();
  auto dec_dev = res.alloc<typename FTI::WordT>(stream, totalSize);

  auto decPtrs = std::vector<void*>(batchSizes.size());
  {
    uint32_t curOffset = 0;
    for (int i = 0; i < inPtrs.size(); ++i) {
      decPtrs[i] = (typename FTI::WordT*)dec_dev.data() + curOffset;
      curOffset += batchSizes[i];
    }
  }

  auto outSuccess_dev = res.alloc<uint8_t>(stream, numInBatch);
  auto outSize_dev = res.alloc<uint32_t>(stream, numInBatch);

  auto decompConfig =
      FloatDecompressConfig(FT, ANSCodecConfig(probBits), false, true);

  floatDecompressSparse(
      res,
      decompConfig,
      numInBatch,
      (const void**)encPtrs.data(),
      decPtrs.data(),
      batchSizes.data(),
      outSuccess_dev.data(),
      outSize_dev.data(),
      stream);

  cudaDeviceSynchronize();

  auto end2 = std::chrono::system_clock::now();
    
  std::chrono::duration<double> elapsed_seconds2 = end2-start2;
  
  auto outSuccess = outSuccess_dev.copyToHost(stream);
  auto outSize = outSize_dev.copyToHost(stream);

  for (int i = 0; i < outSuccess.size(); ++i) {
    EXPECT_TRUE(outSuccess[i]);
    EXPECT_EQ(outSize[i], batchSizes[i]);
  }

  auto decSparseFinal = (typename FloatTypeInfo<FT>::WordT *)malloc(sizeof(typename FloatTypeInfo<FT>::WordT) * totalSize);

  cudaMemcpy(decSparseFinal, dec_dev.data(), sizeof(typename FloatTypeInfo<FT>::WordT) * totalSize, cudaMemcpyDeviceToHost);

  std::vector<typename FloatTypeInfo<FT>::WordT> decSparse(decSparseFinal, decSparseFinal + totalSize);


  // get empirical compression ratio
  auto outSizeComp = outBatchSize_dev.copyToHost(stream);
  float totalNFloats = 0;
  float outSizeTotal = 0;
  for (int i = 0; i < numInBatch; ++i) {
    outSizeTotal += outSizeComp[i];
    totalNFloats += batchSizes[i];
  }
  float inSizeTotal = totalNFloats * sizeof(typename FTI::WordT);

  float compressionRatio = (outSizeTotal) / inSizeTotal;

  double elapsed_seconds_r = std::chrono::duration_cast<std::chrono::microseconds>(elapsed_seconds).count() / 1e6;
  double elapsed_seconds2_r = std::chrono::duration_cast<std::chrono::microseconds>(elapsed_seconds2).count() / 1e6;
  float bw1 = (inSizeTotal / 1e9) / elapsed_seconds_r;
  float bw2 = (inSizeTotal / 1e9) / (elapsed_seconds2_r);

  if (outputFile.is_open()) {
      // Write the current time to the file
      // float compressionRatio = 0;
      // if (idx == 1) {
      //   compressionRatio = (11 + 2.7) / 16;
      // }
      // else if (idx == 2) {
      //   compressionRatio = (8 + 2.7) / 16;
      // }
      // else if (idx == 3) {
      //   compressionRatio = (24 + 2.7) / 32;
      // }
      // else if (idx == 4) {
      //   compressionRatio = (48 + 2.7*2) / 64;
      // }
      outputFile << idx <<" "<< 9 << " " << compressionRatio<< " "<< totalNFloats / 1e6 <<" "<<bw1<<" "<<bw2 << std::endl;

      // Close the file
      outputFile.close();
      std::cout << "Current time has been written to 'time.txt'" << std::endl;
  } else {
      std::cerr << "Error opening the file 'time.txt'" << std::endl;
  }

  

  for (int i = 0; i < origSparse.size(); ++i) {
    if (origSparse[i] != decSparse[i]) {
      printf(
          "mismatch at %d / %d: 0x%08X 0x%08X\n",
          i,
          (int)origSparse.size(),
          origSparse[i],
          decSparse[i]);
      std::cout<<"origSparse[i]: "<<origSparse[i]<<std::endl;
      std::cout<<"decSparse[i]: "<<decSparse[i]<<std::endl;
      break;
    }
  }

  EXPECT_EQ(origSparse, decSparse);
}

void runBatchPointerTest(
    StackDeviceMemory& res,
    FloatType ft,
    int probBits,
    const std::vector<uint32_t>& batchSizes) {
  switch (ft) {
    case FloatType::kFloat16:
      runBatchPointerTest<FloatType::kFloat16>(res, probBits, batchSizes, 1);
      break;
    case FloatType::kBFloat16:
      runBatchPointerTest<FloatType::kBFloat16>(res, probBits, batchSizes, 2);
      break;
    case FloatType::kFloat32:
      runBatchPointerTest<FloatType::kFloat32>(res, probBits, batchSizes, 3);
      break;
    case FloatType::kFloat64:
      runBatchPointerTest<FloatType::kFloat64>(res, probBits, batchSizes, 4);
      break;
    default:
      CHECK(false);
      break;
  }
}

void runBatchPointerTest(
    StackDeviceMemory& res,
    FloatType ft,
    int probBits,
    int numInBatch,
    uint32_t multipleOf = 1) {
  std::mt19937 gen(10 + numInBatch);
  std::uniform_int_distribution<uint32_t> dist(1, 1);

  auto batchSizes = std::vector<uint32_t>(numInBatch);
  for (auto& v : batchSizes) {
    v = roundUp(dist(gen), multipleOf);
  }

  runBatchPointerTest(res, ft, probBits, batchSizes);
}

TEST(FloatTest, Batch) {
  
  auto res = makeStackMemory(10000000000);
  int idx = 0;
  for (auto ft :
       {FloatType::kFloat16, FloatType::kBFloat16, FloatType::kFloat32,FloatType::kFloat64}) {
        // {FloatType::kFloat32}) {
    idx++;
    for (auto numInBatch : {1}) {
      for (auto probBits : {9}) {
        // for (long long multipleOf : {5000}) {
        // 100000000
        // for (int multipleOf : {100}) {
        
        for (int multipleOf : {100000, 150000, 1000000, 1500000, 10000000, 15000000}) {

          // , 150000, 1000000, 1500000, 10000000, 15000000, 10000000

        

        runBatchPointerTest(res, ft, probBits, numInBatch, multipleOf);
        // Some computation here


        }
      }
    }
  }
}

// TEST(FloatTest, Batch) {
  
//   auto res = makeStackMemory(10000000000);
//   int idx = 0;
//   for (auto ft :
//        {FloatType::kFloat16, FloatType::kBFloat16, FloatType::kFloat32, FloatType::kFloat64}) {
//         // {FloatType::kFloat64}) {
//     idx++;
//     for (auto numInBatch : {1}) {
//       for (auto probBits : {9}) {
//         // for (long long multipleOf : {5000}) {
//         // 100000000
//         for (int multipleOf : {100000}) {
        
//         // for (int multipleOf : {100000, 150000, 1000000, 1500000, 10000000, 15000000}) {

        

//         runBatchPointerTest(res, ft, probBits, numInBatch, multipleOf);
//         // Some computation here


//         }
//       }
//     }
//   }
// }



// TEST(FloatTest, Batch) {
//   auto res = makeStackMemory();
//   int idx = 0;
//   for (auto ft :
//        {FloatType::kFloat16, FloatType::kBFloat16, FloatType::kFloat32, FloatType::kFloat64}) {
//     idx++;
//     for (auto numInBatch : {1, 50, 100, 150, 200}) {
//       for (auto probBits : {9, 10, 11}) {
        

        
//         const int batchSize = numInBatch; // Adjust as needed.

//         std::ofstream outputFile("/home/ee274_mfguo_nsagan/mfguo/dietgpu_fork/dietgpu/float/time_comet.txt", std::ios::app);
        
        

//         auto start = std::chrono::system_clock::now();
    

//         runBatchPointerTest(res, ft, probBits, numInBatch);
//         // Some computation here
//         auto end = std::chrono::system_clock::now();
    
//         std::chrono::duration<double> elapsed_seconds = end-start;
//         std::time_t end_time = std::chrono::system_clock::to_time_t(end);
        

//         if (outputFile.is_open()) {
//             // Write the current time to the file
//             outputFile << idx << " " << probBits<< " "<<numInBatch <<" "<<elapsed_seconds.count() << std::endl;

//             // Close the file
//             outputFile.close();
//             std::cout << "Current time has been written to 'time.txt'" << std::endl;
//         } else {
//             std::cerr << "Error opening the file 'time.txt'" << std::endl;
//         }
//       //   }
//       }
//     }
//   }
// }

