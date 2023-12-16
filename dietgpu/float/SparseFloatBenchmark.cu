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
    std::memcpy(&out, &v, sizeof(double));
    return out;
  }
};


template <FloatType FT>
std::vector<typename FloatTypeInfo<FT>::WordT> generateSparseFloats(int num, float sparsityPercent=0.5) {
  std::mt19937 gen(10 + num);
  std::normal_distribution<float> dist;

  std::random_device rd;  // Initialize a random seed
  std::default_random_engine generator(rd());

  // Create a uniform distribution for floating-point numbers between 0 and 1
  // This will determine the sparsity pattern.
  std::uniform_real_distribution<double> sparsity_pattern_dist(0.0, 1.0);

  auto out = std::vector<typename FloatTypeInfo<FT>::WordT>(num);
  for (auto& v : out) {
    v = GenerateFloat<FT>::gen(dist(gen)); // Float from normal dist

    double random_number = sparsity_pattern_dist(generator);
    if(random_number < sparsityPercent) {
      v = 0.0;
    }
  }

  return out;
}

/*
 * Benchmark sparse float compression bandwidth and compression ratio, and
 * test correctness.
 * 
 * This function generates a specified number of floats with a specified
 * sparsity ratio and compresses them using sparse float compression. The
 * compression and decompression times are measured,
 * and the bandwidth is calculated as (time / GB of floats compressed). The
 * compression ratio is calculated as (output bytes / input bytes). 
 *
 * Outputs are written in CSV format to the specified output file
 */
template <FloatType FT>
void runBenchmark(
    StackDeviceMemory& res,
    int probBits,
    const std::vector<uint32_t>& batchSizes,
    char* outputFilename,
    bool writeHeader=false) {
  using FTI = FloatTypeInfo<FT>;

  // run on a different stream to test stream assignment
  auto stream = CudaStream::makeNonBlocking();

if (writeHeader) {
    // Text file to which the output will be written.
    std::ofstream outputFile(outputFilename, std::ios_base::out);
    if (outputFile.is_open()) {
      outputFile << "float_type, prob_bits, num_batches, million_floats, sparsity, comp_bandwidth_gbps, decomp_bandwidth_gbps" << std::endl;
      outputFile.close();
    }
  }
  // Now, we should append to the file instead of overwriting!
  std::ofstream outputFile(outputFilename, std::ios_base::app);

  // Number of input batches
  int numInBatch = batchSizes.size();

  // Find the maximum number of floats in a batch, as well as the total number
  // across all batch  
  uint32_t totalSize = 0;
  
  uint32_t maxSize = 0;
  for (auto v : batchSizes) {
    totalSize += v;
    maxSize = std::max(maxSize, v);
  }

  auto origSparse = generateSparseFloats<FT>(totalSize);
  
  // Copy the floats to the device before starting the timer to be consistent
  // with benchmark.py, where the data is already loaded onto the GPU via
  // PyTorch before benchmarking begins
  auto origSparse_dev = res.copyAlloc<typename FTI::WordT>(stream, origSparse);

  // start the timer for measuring compression performance
  auto comp_start = std::chrono::system_clock::now();

  // Set up pointers to input data to be compressed
  auto inPtrs = std::vector<const void*>(batchSizes.size());
  {
    uint32_t curOffset = 0;
    for (int i = 0; i < inPtrs.size(); ++i) {
      inPtrs[i] = (const typename FTI::WordT*) origSparse_dev.data() + curOffset;
      curOffset += batchSizes[i];
    }
  }

    // Max compressed size for a single float
  auto maxCompressedSize = getMaxSparseFloatCompressedSize(FT, maxSize);

  // This is the memory for the compression outpu  
  auto enc_dev = res.alloc<uint8_t>(stream, numInBatch * maxCompressedSize);

  // Output pointers
  auto encPtrs = std::vector<void*>(batchSizes.size());
  {
    for (int i = 0; i < inPtrs.size(); ++i) {
      encPtrs[i] = (uint8_t*)enc_dev.data() + i * maxCompressedSize;
    }
  }

  // floatCompressSparse will populate this array with the size of the output, per batch
  auto outBatchSize_dev = res.alloc<uint32_t>(stream, numInBatch);
  auto compConfig =
      FloatCompressConfig(FT, ANSCodecConfig(probBits), false, true);

  floatCompressSparse(
      res,
      compConfig,
      numInBatch,
      inPtrs.data(),
      batchSizes.data(),
      encPtrs.data(),
      outBatchSize_dev.data(),
      stream);

  // Make sure all CUDA operations are finished before we end the timer
  cudaDeviceSynchronize();

  // Stop the timer and see how long compression took
  auto comp_end = std::chrono::system_clock::now();
    
  std::chrono::duration<double> comp_dur = comp_end-comp_start;
  double elapsed_seconds_comp =
      std::chrono::duration_cast<std::chrono::microseconds>(comp_dur).count() / 1e6;

  // Decompression
  auto dec_dev = res.alloc<typename FTI::WordT>(stream, totalSize);

  // Set up the input pointers for decompression
  auto decPtrs = std::vector<void*>(batchSizes.size());
  {
    uint32_t curOffset = 0;
    for (int i = 0; i < inPtrs.size(); ++i) {
      decPtrs[i] = (typename FTI::WordT*)dec_dev.data() + curOffset;
      curOffset += batchSizes[i];
    }
  }

  // These will be populated with decompression success and output size, per batch
  auto outSuccess_dev = res.alloc<uint8_t>(stream, numInBatch);
  auto outSize_dev = res.alloc<uint32_t>(stream, numInBatch);

  auto decompConfig =
      FloatDecompressConfig(FT, ANSCodecConfig(probBits), false, true);

  // start the timer
  auto decomp_start = std::chrono::system_clock::now();

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
  // wait for CUDA operations to finish and stop the timer
  cudaDeviceSynchronize();

  auto end2 = std::chrono::system_clock::now();
  auto decomp_end = std::chrono::system_clock::now();
    
  std::chrono::duration<double> decomp_dur = decomp_end-decomp_start;
  double elapsed_seconds_decomp =
      std::chrono::duration_cast<std::chrono::microseconds>(decomp_dur).count() / 1e6;
  
  auto outSuccess = outSuccess_dev.copyToHost(stream);
  auto outSize = outSize_dev.copyToHost(stream);


  // get empirical compression ratio by computing the output bytes, divided by
  // input bytes
  auto outSizeComp = outBatchSize_dev.copyToHost(stream);
  float totalNFloats = 0;
  float outSizeTotal = 0;
  for (int i = 0; i < numInBatch; ++i) {
    outSizeTotal += outSizeComp[i];
    totalNFloats += batchSizes[i];
  }
  float inSizeTotal = totalNFloats * sizeof(typename FTI::WordT);

  float compressionRatio = (outSizeTotal) / inSizeTotal;

  // compute bandwidths in GB/s
  float bw_comp = (inSizeTotal / 1e9) / elapsed_seconds_comp;
  float bw_decomp = (inSizeTotal / 1e9) / elapsed_seconds_decomp;

  char* float_type;
  switch (FT) {
    case FloatType::kFloat16:
      float_type = "Float16";
      break;
    case FloatType::kBFloat16:
      float_type = "BFloat16";
      break;
    case FloatType::kFloat32:
      float_type = "Float32";
      break;
    case FloatType::kFloat64:
      float_type = "Float64";
      break;
    default:
      assert(false);
  }

// Write the benchmark data to the specified file
  if (outputFile.is_open()) {
      outputFile << float_type << ", "<< 9 << ", " << numInBatch << ", " << totalNFloats / 1e6 << ", " << 0.5 <<
        ", " << compressionRatio<< ", " << bw_comp << ", " << bw_decomp << std::endl;

      // Close the file
      outputFile.close();
      std::cout << "Current time has been written to " << outputFilename << std::endl;
  } else {
      std::cerr << "Error opening the file" << outputFilename << std::endl;
  }


  // Also, test for correctness
  for (int i = 0; i < outSuccess.size(); ++i) {
    EXPECT_TRUE(outSuccess[i]);
    EXPECT_EQ(outSize[i], batchSizes[i]);
  }

  // Get the final decompressed floats into host memory 
  auto decSparseFinal = (typename FloatTypeInfo<FT>::WordT *)malloc(sizeof(typename FloatTypeInfo<FT>::WordT) * totalSize);
  cudaMemcpy(decSparseFinal, dec_dev.data(), sizeof(typename FloatTypeInfo<FT>::WordT) * totalSize, cudaMemcpyDeviceToHost);
  std::vector<typename FloatTypeInfo<FT>::WordT> decSparse(decSparseFinal, decSparseFinal + totalSize);

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

void runBenchmark(
    StackDeviceMemory& res,
    FloatType ft,
    int probBits,
    const std::vector<uint32_t>& batchSizes,
    char* outputFile,
    bool writeHeader=false) {
  switch (ft) {
    case FloatType::kFloat16:
      runBenchmark<FloatType::kFloat16>(res, probBits, batchSizes, outputFile, writeHeader);
      break;
    case FloatType::kBFloat16:
      runBenchmark<FloatType::kBFloat16>(res, probBits, batchSizes, outputFile, writeHeader);
      break;
    case FloatType::kFloat32:
      runBenchmark<FloatType::kFloat32>(res, probBits, batchSizes, outputFile, writeHeader);
      break;
    case FloatType::kFloat64:
      runBenchmark<FloatType::kFloat64>(res, probBits, batchSizes, outputFile, writeHeader);
      break;
    default:
      CHECK(false);
      break;
  }
}

void runBenchmark(
    StackDeviceMemory& res,
    FloatType ft,
    int probBits,
    int numInBatch,
    uint32_t multipleOf,
    char* outputFile,
    bool writeHeader=false) {
  std::mt19937 gen(10 + numInBatch);
  std::uniform_int_distribution<uint32_t> dist(1, 1);

  auto batchSizes = std::vector<uint32_t>(numInBatch);
  for (auto& v : batchSizes) {
    v = roundUp(dist(gen), multipleOf);
  }

  runBenchmark(res, ft, probBits, batchSizes, outputFile, writeHeader);
}

int main(int argc, char *argv[])
{

  char* outputFile;
  if (argc == 1) {
    std::cout << "\n-----------------------------------------------------------------" << std::endl;
    std::cout << "No output file specified; writing to './benchmark.txt'." << std::endl;
    std::cout << "\nYou can specify an output file as the first command-line argument" << std::endl;
    std::cout << "to this executable, e.g., ./sparse_float_benchmark fileName.txt." << std::endl;
    std::cout << "-----------------------------------------------------------------\n" << std::endl;
    outputFile = "./benchmark.txt";
  } else {
    outputFile = argv[1];
  }
  
  // Make a large enough stack memory or we will get performance
  // issues from resizing the stack memory
  auto res = makeStackMemory(10000000000);
  bool first = true;
  for (auto ft :
       {FloatType::kFloat16, FloatType::kBFloat16, FloatType::kFloat32,FloatType::kFloat64}) {
    for (auto numInBatch : {1, 3, 5}) {
      for (int multipleOf : {100000, 150000, 1000000, 1500000, 10000000, 15000000}) {
        runBenchmark(res, ft, 9, numInBatch, multipleOf, outputFile, first);
        first = false;
      }
    }
  }
}

