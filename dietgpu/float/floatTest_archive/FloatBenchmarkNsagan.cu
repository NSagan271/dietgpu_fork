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

#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuFloatUtils.cuh"
#include "dietgpu/utils/StackDeviceMemory.h"

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

  auto out = std::vector<typename FloatTypeInfo<FT>::WordT>(num);
  for (auto& v : out) {
    v = GenerateFloat<FT>::gen(dist(gen));
  }

  return out;
}

std::vector<uint64_t> readFloats(char* filename, uint32_t max_read=10000000) {
  std::ifstream is (filename, std::ifstream::binary);
  if (is) {
    // get length of file:
    is.seekg (0, is.end);
    uint32_t length = is.tellg();
    is.seekg (0, is.beg);

    uint32_t num = std::min(length / 8, max_read);
    std::cout << num << std::endl;
    auto out = std::vector<uint64_t>(num);
    is.read((char*) out.data(), num*8);
    is.close();

    return out;
  }
  return std::vector<uint64_t>(1);
}

void benchmarkFloat64Data(
    StackDeviceMemory& res,
    int probBits,
    char* data_filename,
    int batch_size=500000) {
  
  auto FT = FloatType::kFloat64;
  using FTI = FloatTypeInfo<FloatType::kFloat64>;

  // run on a different stream to test stream assignment
  auto stream = CudaStream::make();

  std::ofstream outputFile("/home/ee274_mfguo_nsagan/nsagan/dietgpu_fork/dietgpu/float/time_comp_decompress_50.txt", std::ios::app);

  auto orig = readFloats(data_filename);

  int numInBatch = (orig.size() + batch_size - 1) / batch_size;

  std::vector<uint32_t> batchSizes(numInBatch, batch_size);
  batchSizes[numInBatch - 1] = orig.size() - batch_size * (numInBatch - 1);
  uint32_t totalSize = orig.size();
  uint32_t maxSize = batch_size;

  auto maxCompressedSize = getMaxFloatCompressedSize(FT, maxSize);

  auto orig_dev = res.copyAlloc(stream, orig);

  auto start = std::chrono::system_clock::now();
  auto inPtrs = std::vector<const void*>(batchSizes.size());
  {
    uint32_t curOffset = 0;
    for (int i = 0; i < inPtrs.size(); ++i) {
      inPtrs[i] = (const typename FTI::WordT*)orig_dev.data() + curOffset;
      curOffset += batchSizes[i];
    }
  }

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

  floatCompress(
      res,
      compConfig,
      numInBatch,
      inPtrs.data(),
      batchSizes.data(),
      encPtrs.data(),
      outBatchSize_dev.data(),
      stream);
  

  cudaDeviceSynchronize();

  auto end = std::chrono::system_clock::now();
    
  std::chrono::duration<double> dur1 = end-start;
  double elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>(dur1).count() / 1e6;

  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  // Decode data
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

  // printf("data compress config\n");
  // printf("data compress\n");

 auto start2 = std::chrono::system_clock::now();

  floatDecompress(
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
    
  std::chrono::duration<double> dur2 = end2-start2;
  double elapsed_seconds2 = std::chrono::duration_cast<std::chrono::microseconds>(dur2).count() / 1e6;
  std::time_t end_time2 = std::chrono::system_clock::to_time_t(end2);

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

  float bw1 = (inSizeTotal / 1e9) / elapsed_seconds;
  float bw2 = (inSizeTotal / 1e9) / elapsed_seconds2;

  if (outputFile.is_open()) {

      outputFile <<" "<< 9 << " " << compressionRatio<< " "<< totalNFloats / 1e6 <<" "<<bw1<<" "<<bw2 << std::endl;

      // Close the file
      outputFile.close();
      std::cout << "Current time has been written to 'time.txt'" << std::endl;
  } else {
      std::cerr << "Error opening the file 'time.txt'" << std::endl;
  }

  auto outSuccess = outSuccess_dev.copyToHost(stream);
  auto outSize = outSize_dev.copyToHost(stream);

  for (int i = 0; i < outSuccess.size(); ++i) {
    EXPECT_TRUE(outSuccess[i]);
    EXPECT_EQ(outSize[i], batchSizes[i]);
  }

  auto dec = dec_dev.copyToHost(stream);

  for (int i = 0; i < orig.size(); ++i) {
    if (orig[i] != dec[i]) {
      printf(
          "mismatch at %d / %d: 0x%08X 0x%08X\n",
          i,
          (int)orig.size(),
          orig[i],
          dec[i]);
      break;
    }
  }

  EXPECT_EQ(orig, dec);
}

template <FloatType FT>
void runBatchPointerTest(
    StackDeviceMemory& res,
    int probBits,
    const std::vector<uint32_t>& batchSizes,
    int idx) {
      
  using FTI = FloatTypeInfo<FT>;

  // run on a different stream to test stream assignment
  auto stream = CudaStream::make();


  std::ofstream outputFile("/home/ee274_mfguo_nsagan/nsagan/dietgpu_fork/dietgpu/float/time_comp_decompress_50.txt", std::ios::app);

  int numInBatch = batchSizes.size();
  
  uint32_t totalSize = 0;
  uint32_t maxSize = 0;
  for (auto v : batchSizes) {
    totalSize += v;
    maxSize = std::max(maxSize, v);
  }

  
  auto maxCompressedSize = getMaxFloatCompressedSize(FT, maxSize);
  // const std::string filename = "/home/ee274_mfguo_nsagan/mfguo/dietgpu_fork/dietgpu/float/num_comet.trace.fpc"; // Replace with your file name


  // std::ifstream inputFile(filename, std::ios::binary);

  // if (!inputFile) {
  //     std::cerr << "Failed to open file: " << filename << std::endl;
  // }

  // // Create a vector to hold the float64 data
  // std::vector<double> data(totalSize);


        

  // auto maxCompressedSize = getMaxFloatCompressedSize(FT, maxSize);


  // // Read and process data in batches.
  // while (true) {
  //     // Read a batch of data from the file
  //     inputFile.read(data.data(), totalSize * 16);
      
  //     // Check how many elements were actually read
  //     std::streamsize bytesRead = inputFile.gcount();
  //     if (bytesRead == 0) {
  //         break; // End of file reached
  //     }


  //     auto orig = data;
  //     auto orig_dev = res.copyAlloc(stream, orig);

  //     auto inPtrs = std::vector<const void*>(batchSizes.size());
  //     {
  //       uint32_t curOffset = 0;
  //       for (int i = 0; i < inPtrs.size(); ++i) {
  //         inPtrs[i] = (const typename FTI::WordT*)orig_dev.data() + curOffset;
  //         curOffset += batchSizes[i];
  //       }
  //     }

  //     auto enc_dev = res.alloc<uint8_t>(stream, numInBatch * maxCompressedSize);

  //     auto encPtrs = std::vector<void*>(batchSizes.size());
  //     {
  //       for (int i = 0; i < inPtrs.size(); ++i) {
  //         encPtrs[i] = (uint8_t*)enc_dev.data() + i * maxCompressedSize;
  //       }
  //     }

  //     auto outBatchSize_dev = res.alloc<uint32_t>(stream, numInBatch);

  //     auto compConfig =
  //         FloatCompressConfig(FT, ANSCodecConfig(probBits), false, true);
      
  //     floatCompress(
  //         res,
  //         compConfig,
  //         numInBatch,
  //         inPtrs.data(),
  //         batchSizes.data(),
  //         encPtrs.data(),
  //         outBatchSize_dev.data(),
  //         stream);
      
  //     // Decode data
  //     auto dec_dev = res.alloc<typename FTI::WordT>(stream, totalSize);

  //     auto decPtrs = std::vector<void*>(batchSizes.size());
  //     {
  //       uint32_t curOffset = 0;
  //       for (int i = 0; i < inPtrs.size(); ++i) {
  //         decPtrs[i] = (typename FTI::WordT*)dec_dev.data() + curOffset;
  //         curOffset += batchSizes[i];
  //       }
  //     }

  //     auto outSuccess_dev = res.alloc<uint8_t>(stream, numInBatch);
  //     auto outSize_dev = res.alloc<uint32_t>(stream, numInBatch);

  //     auto decompConfig =
  //         FloatDecompressConfig(FT, ANSCodecConfig(probBits), false, true);

  //     // printf("data compress config\n");
  //     // printf("data compress\n");

  //     floatDecompress(
  //         res,
  //         decompConfig,
  //         numInBatch,
  //         (const void**)encPtrs.data(),
  //         decPtrs.data(),
  //         batchSizes.data(),
  //         outSuccess_dev.data(),
  //         outSize_dev.data(),
  //         stream);

  //     auto outSuccess = outSuccess_dev.copyToHost(stream);
  //     auto outSize = outSize_dev.copyToHost(stream);

  //     for (int i = 0; i < outSuccess.size(); ++i) {
  //       EXPECT_TRUE(outSuccess[i]);
  //       EXPECT_EQ(outSize[i], batchSizes[i]);
  //     }

  //     auto dec = dec_dev.copyToHost(stream);

  //     for (int i = 0; i < orig.size(); ++i) {
  //       if (orig[i] != dec[i]) {
  //         printf(
  //             "mismatch at %d / %d: 0x%08X 0x%08X\n",
  //             i,
  //             (int)orig.size(),
  //             orig[i],
  //             dec[i]);
  //         break;
  //       }
  //     }

  //     EXPECT_EQ(orig, dec);

  // }

  auto orig = generateFloats<FT>(totalSize);
  auto orig_dev = res.copyAlloc(stream, orig);

  auto start = std::chrono::system_clock::now();
  auto inPtrs = std::vector<const void*>(batchSizes.size());
  {
    uint32_t curOffset = 0;
    for (int i = 0; i < inPtrs.size(); ++i) {
      inPtrs[i] = (const typename FTI::WordT*)orig_dev.data() + curOffset;
      curOffset += batchSizes[i];
    }
  }

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

  floatCompress(
      res,
      compConfig,
      numInBatch,
      inPtrs.data(),
      batchSizes.data(),
      encPtrs.data(),
      outBatchSize_dev.data(),
      stream);
  

  cudaDeviceSynchronize();

  auto end = std::chrono::system_clock::now();
    
  std::chrono::duration<double> dur1 = end-start;
  double elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>(dur1).count() / 1e6;

  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  // Decode data
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

  // printf("data compress config\n");
  // printf("data compress\n");

 auto start2 = std::chrono::system_clock::now();

  floatDecompress(
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
    
  std::chrono::duration<double> dur2 = end2-start2;
  double elapsed_seconds2 = std::chrono::duration_cast<std::chrono::microseconds>(dur2).count() / 1e6;
  std::time_t end_time2 = std::chrono::system_clock::to_time_t(end2);

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

  float bw1 = (inSizeTotal / 1e9) / elapsed_seconds;
  float bw2 = (inSizeTotal / 1e9) / elapsed_seconds2;

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

  auto outSuccess = outSuccess_dev.copyToHost(stream);
  auto outSize = outSize_dev.copyToHost(stream);

  for (int i = 0; i < outSuccess.size(); ++i) {
    EXPECT_TRUE(outSuccess[i]);
    EXPECT_EQ(outSize[i], batchSizes[i]);
  }

  auto dec = dec_dev.copyToHost(stream);

  for (int i = 0; i < orig.size(); ++i) {
    if (orig[i] != dec[i]) {
      printf(
          "mismatch at %d / %d: 0x%08X 0x%08X\n",
          i,
          (int)orig.size(),
          orig[i],
          dec[i]);
      break;
    }
  }

  EXPECT_EQ(orig, dec);
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
  benchmarkFloat64Data(res, 9, "../data/obs_temp.out", 1000000);

  // int idx = 0;
  // for (auto ft :
  //     //  {FloatType::kFloat16, FloatType::kBFloat16, FloatType::kFloat32, FloatType::kFloat64}) {
  //       {FloatType::kFloat64}) {
  //   idx++;
  //   for (auto numInBatch : {1}) {
  //     for (auto probBits : {9}) {
  //       for (long long multipleOf : {100000, 150000, 1000000, 1500000, 10000000, 15000000, 100000000}) {

        

  //       runBatchPointerTest(res, ft, probBits, numInBatch, multipleOf);
  //       // Some computation here


  //       }
  //     }
  //   }
  // }
}



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

