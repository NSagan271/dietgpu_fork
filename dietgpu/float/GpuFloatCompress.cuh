/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "dietgpu/ans/BatchProvider.cuh"
#include "dietgpu/ans/GpuANSEncode.cuh"
#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/ans/GpuChecksum.cuh"
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuFloatUtils.cuh"
#include "dietgpu/utils/DeviceDefs.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"

#include <glog/logging.h>
#include <cmath>
#include <memory>
#include <vector>

// Each float will have at most two bytes to be compressed
// (and this is only the case for Float64; the rest only have one byte to be
// compressed).
#define MAX_NUM_COMP_OUTS 2

namespace dietgpu {

/* Each type of floating point number has qualitatively different float
 * splitting: float 16 has one byte of compressed output and one byte of
 * non-compressed output, float32 has one byte of compressed output and
 * two bytes of uncompressed output, and float 64 has two bytes of compressed
 * output (which form two deparate ANS datasets) and size bytes of
 * non-compressed output.
 *
 * So, to avoid repeating large amounts of code, we put the code that differs
 * into UpdateCompAndHist.update, which splits a single float.
 *
 * This specific iteration of UpdateCompAndHist works for Float16 and BFloat16.
 */
template <FloatType FT>
struct UpdateCompAndHist {
  static __device__ void update(
      // One input float
      const typename FloatTypeInfo<FT>::WordT inWord,

      // Pointer to the compressed output
      typename FloatTypeInfo<FT>::CompT* compOuts,

      // For Float32 and Float64, the non-compressed output is split into two
      // different sections of memory (24 and 48 bits are left uncompressed,
      // respectively, neither of which is a power of two). So, this function
      // lets you pass in two separate non-compressed outputs. 
      //
      // For float16, nonComp2Out is unused.
      typename FloatTypeInfo<FT>::NonCompSplit1T& nonComp1Out,
      typename FloatTypeInfo<FT>::NonCompSplit2T& nonComp2Out,

      // For Float64, there are two different ANS datasets, requiring two
      // separate histograms, so warpHistograms has to be a 2D array with
      // two "rows," one for each histogram.
      // For Float16 and Float32, this is still a 2D array, but it only has
      // one "row."
      uint32_t** warpHistograms) {

    using FTI = FloatTypeInfo<FT>;
    using CompT = typename FTI::CompT;
    using NonCompT = typename FTI::NonCompT;
    using WordT = typename FTI::WordT;

    NonCompT nonComp;
    FTI::split(inWord, compOuts, nonComp);

    nonComp1Out = nonComp;

    // Add to the ANS histogram
    atomicAdd(&warpHistograms[0][compOuts[0]], 1);
  }
};

/* Float32 specialization for a single float-splitting step. See the first
 * first definition of struct UpdateCompAndHist for full details.
 *
 * Float32 has two different non-compressed outputs, but only one compressed
 * output (and one warp histogram)
 */
template<>
struct UpdateCompAndHist<FloatType::kFloat32> {
  static __device__ void update(
      // One input float
      const typename FloatTypeInfo<FloatType::kFloat32>::WordT inWord,

      // Pointer to the compressed output
      typename FloatTypeInfo<FloatType::kFloat32>::CompT* compOuts,

      // The lower 16 bits of the non-compressed output
      typename FloatTypeInfo<FloatType::kFloat32>::NonCompSplit1T& nonComp1Out,

      // The upper 8 bits of the non-compressed output
      typename FloatTypeInfo<FloatType::kFloat32>::NonCompSplit2T& nonComp2Out,

      // This needs to be a 2D array because Float64 compression requires two
      // different ANS histograms, but, for Float32, it only has one row
      uint32_t** warpHistograms) {

    using FTI = FloatTypeInfo<FloatType::kFloat32>;
    using CompT = typename FTI::CompT;
    using NonCompT = typename FTI::NonCompT;
    using WordT = typename FTI::WordT;

    NonCompT nonComp;
    FTI::split(inWord, compOuts, nonComp);

    nonComp1Out = nonComp & 0xffffU;
    nonComp2Out = nonComp >> 16;

    // Add to the ANS histogram
    atomicAdd(&warpHistograms[0][compOuts[0]], 1);
  }
};

/* Float64 specialization for a single float-splitting step. See the first
 * first definition of struct UpdateCompAndHist for full details.
 *
 * Float64 has two different non-compressed outputs, but and two different
 * compressed outputs. Each compressed output corresponds to a separate ANs
 * dataset, so we require two different ANS histograms.
 */
template<>
struct UpdateCompAndHist<FloatType::kFloat64> {
  static __device__ void update(
      // One input float
      const typename FloatTypeInfo<FloatType::kFloat64>::WordT inWord,

      // Two-element array for the two bytes that are going to be compressed
      typename FloatTypeInfo<FloatType::kFloat64>::CompT* compOuts,

      // The lower 32 bits of the non-compressed output
      typename FloatTypeInfo<FloatType::kFloat64>::NonCompSplit1T& nonComp1Out,

      // The upper 16 bits of the non-compressed output
      typename FloatTypeInfo<FloatType::kFloat64>::NonCompSplit2T& nonComp2Out,

      // This has two "rows": one for each ANS dataset
      uint32_t** warpHistograms) {

    using FTI = FloatTypeInfo<FloatType::kFloat64>;
    using CompT = typename FTI::CompT;
    using NonCompT = typename FTI::NonCompT;
    using WordT = typename FTI::WordT;

    NonCompT nonComp;
    FTI::split(inWord, compOuts, nonComp);

    nonComp1Out = nonComp & 0xffffffffU;
    nonComp2Out = nonComp >> 32;

    // Add to both ANS histograms
    atomicAdd(&warpHistograms[0][compOuts[0]], 1);
    atomicAdd(&warpHistograms[1][compOuts[1]], 1);
  }
};

  /* Perform float splitting for data that is not 16-byte aligned. 16-byte
   * aligned data would allow us to use vectorized memory operations, which
   * are more efficient (but also have a less straightforward implementation).
   */
  template <FloatType FT, int Threads>
  struct SplitFloatNonAligned {
    static __device__ void split(
        // Input array
        const typename FloatTypeInfo<FT>::WordT* in,

        // Number of input floats
        uint32_t size,

        // Output array of all data that will be ANS compressed (aka, the
        // exponents). For Float64 and Float64 only, this will be two different
        // ANS datasets, separated by compDatasetStride number of elements
        typename FloatTypeInfo<FT>::CompT* compOuts,

        // Location in memory to write all of the non-compressed bytes
        typename FloatTypeInfo<FT>::NonCompT* nonCompOut,

        // Histogram of byte frequencies for ANS compression. For Float64 and
        // Float64 only, this will be two different histograms, separated
        // by histDatasetStride number of elements
        uint32_t* warpHistograms,

        // For Float64 compression, the second ANS dataset will be compDatasetStride
        // elements after the beginning of compOuts. For other float types, this
        // is irrelevant
        uint32_t compDatasetStride,

        // For Float64 compression, the second ANS histogram will be
        // histDatasetStride elements after the beginning of warpHistograms.
        // For other float types, this is irrelevant.
        uint32_t histDatasetStride) {
      using FTI = FloatTypeInfo<FT>;
      using CompT = typename FTI::CompT;
      using NonCompT = typename FTI::NonCompT;
      using NonCompSplit1T = typename FTI::NonCompSplit1T;
      using NonCompSplit2T = typename FTI::NonCompSplit2T;

      // Turns warpHistograms into a 2D array (the second "row" is only relevant
      // for Float64, as described above). This is the histogram that will be
      // passed into UpdateCompAndHist<FT>::update
      uint32_t* warpHistogram2DArr[MAX_NUM_COMP_OUTS] = {warpHistograms,
                                      warpHistograms + histDatasetStride};

      // For Float32 and Float64, the non-compressed output is split into two
      // different sections of memory (24 and 48 bits are left uncompressed,
      // respectively, neither of which is a power of two). So, we need support
      // for two different non-compressed datasets (but the second is only used
      // for Float32 or Float64) 
      NonCompSplit1T* nonCompOut1 = (NonCompSplit1T*) nonCompOut;
      NonCompSplit2T* nonCompOut2 = (NonCompSplit2T*) nonCompOut;

      
      if (FTI::getIfNonCompSplit()) // i.e., if it's Float32 or Float64
        nonCompOut2 = (NonCompSplit2T*) (nonCompOut1 + roundUp(size, 16 / sizeof(NonCompSplit1T)));

      for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
          i += gridDim.x * blockDim.x) {
        CompT comps[MAX_NUM_COMP_OUTS];
        NonCompSplit1T nonComp1;
        NonCompSplit2T nonComp2;

        // Specialized float-splitting step for the given datatype
        UpdateCompAndHist<FT>::update(in[i], comps, nonComp1, nonComp2, warpHistogram2DArr);
        nonCompOut1[i] = nonComp1;
        if (FTI::getIfNonCompSplit())
          nonCompOut2[i] = nonComp2;
        
        // Except for Float64, this only loops once. For Float64, this loops
        // twice.
        for (int k = 0; k < FTI::getNumCompSegments(); k++) {
          compOuts[i + k*compDatasetStride] = comps[k];
        }
      }
    }
  };

/* Perform float splitting for data that is 16-byte aligned. This allows us to
 * use vectorized operations, which involve reading 16-byte segments at a time,
 * which allows us to take full advantage of GPU memory bandwidth.
*/
template <FloatType FT, int Threads>
struct SplitFloatAligned16 {
  static __device__ void split(
      // Input array
      const typename FloatTypeInfo<FT>::WordT* __restrict__ in,

      // Number of input floats
      uint32_t size,

      // Output array of all data that will be ANS compressed (aka, the
      // exponents). For Float64 and Float64 only, this will be two different
      // ANS datasets, separated by compDatasetStride number of elements
      typename FloatTypeInfo<FT>::CompT* __restrict__ compOuts,

      // Location in memory to write all of the non-compressed bytes
      typename FloatTypeInfo<FT>::NonCompT* __restrict__ nonCompOut,

      // Histogram of byte frequencies for ANS compression. For Float64 and
      // Float64 only, this will be two different histograms, separated
      // by histDatasetStride number of elements
      uint32_t* warpHistograms,

      // For Float64 compression, the second ANS dataset will be compDatasetStride
      // elements after the beginning of compOuts. For other float types, this
      // is irrelevant. This must be a multiple of 16.
      uint32_t compDatasetStride,

      // For Float64 compression, the second ANS histogram will be
      // histDatasetStride elements after the beginning of warpHistograms.
      // For other float types, this is irrelevant.
      uint32_t histDatasetStride) {

    using FTI = FloatTypeInfo<FT>;

    using WordT = typename FTI::WordT;
    using CompT = typename FTI::CompT;
    using NonCompT = typename FTI::NonCompT;
    using NonCompSplit1T = typename FTI::NonCompSplit1T;
    using NonCompSplit2T = typename FTI::NonCompSplit2T;

    using VecT = typename FTI::VecT;
    using CompVecT = typename FTI::CompVecT;
    using NonCompVecT = typename FTI::NonCompVecT;
    using NonCompVecSplit1T = typename FTI::NonCompVecSplit1T;
    using NonCompVecSplit2T = typename FTI::NonCompVecSplit2T;

    // Loop unrolling for a performance boost 
    constexpr int kOuterUnroll = 2;

    // Number of floats in one 16-byte read. We will be working with vectors
    // of size kInnerUnroll.
    constexpr int kInnerUnroll = sizeof(VecT) / sizeof(WordT);

    // This will be 2 for Float64 and 1 for everything else
    int numCompSegments = FTI::getNumCompSegments();

    // Turns warpHistograms into a 2D array (the second "row" is only relevant
    // for Float64, as described above). This is the histogram that will be
    // passed into UpdateCompAndHist<FT>::update
    uint32_t *warpHistogram2DArr[MAX_NUM_COMP_OUTS] = {warpHistograms, warpHistograms+histDatasetStride};

    // For Float32 and Float64, the non-compressed output is split into two
    // different sections of memory (24 and 48 bits are left uncompressed,
    // respectively, neither of which is a power of two). So, we need support
    // for two different non-compressed datasets (but the second is only used
    // for Float32 or Float64) 
    NonCompSplit1T* nonCompOut1 = (NonCompSplit1T*)nonCompOut;
    NonCompSplit2T* nonCompOut2 = (NonCompSplit2T*)nonCompOut;
    if (FTI::getIfNonCompSplit()) { // i.e., if it's Float32 or Float64
      nonCompOut2 = (NonCompSplit2T*) (nonCompOut1 + roundUp(size, 16 / sizeof(NonCompSplit1T)));
    }

    // Cast the inputs and outputs to vectors so we can perform vectorized
    // memory operations of size kInnerUnroll
    const VecT* inV = (const VecT*)in;
    CompVecT* compOutsV = (CompVecT*)compOuts;
    NonCompVecSplit1T* nonCompOutV1 = (NonCompVecSplit1T*)nonCompOut1;
    NonCompVecSplit2T* nonCompOutV2 = (NonCompVecSplit2T*)nonCompOut2;

    // For Float64 compression, the second (vectorized) ANS dataset will be
    // compDatasetStride elements after the beginning of compOutsV. For other
    //  float types, this is irrelevant
    uint32_t compDatasetStrideVec = compDatasetStride / kInnerUnroll;

    // Each block handles Threads * kOuterUnroll * kInnerUnroll inputs/outputs
    // at a time, or Threads * kOuterUnroll 16-byte words at a time

    constexpr int kWordsPerBlock = Threads * kOuterUnroll;
    constexpr int kFloatsPerBlock = kWordsPerBlock * kInnerUnroll;
    uint32_t fullBlocks = divDown(size, kFloatsPerBlock);

    // Handle by block
    uint32_t startBlock = blockIdx.x * kWordsPerBlock;
    inV += startBlock + threadIdx.x;
    compOutsV += startBlock + threadIdx.x;
    nonCompOutV1 += startBlock + threadIdx.x;
    nonCompOutV2 += startBlock + threadIdx.x;

    for (uint32_t b = blockIdx.x; b < fullBlocks; b += gridDim.x,
                  inV += gridDim.x * kWordsPerBlock,
                  compOutsV += gridDim.x * kWordsPerBlock,
                  nonCompOutV1 += gridDim.x * kWordsPerBlock,
                  nonCompOutV2 += gridDim.x * kWordsPerBlock) {
      VecT v[kOuterUnroll];

#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
        v[i] = inV[i * Threads];
      }

      CompVecT compV[roundUp(kOuterUnroll, 16 / sizeof(CompVecT))*MAX_NUM_COMP_OUTS];
      NonCompVecSplit1T nonCompV1[kOuterUnroll];
      NonCompVecSplit2T nonCompV2[kOuterUnroll];

#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
#pragma unroll
        for (int j = 0; j < kInnerUnroll; ++j) {
          CompT comps[MAX_NUM_COMP_OUTS];
          NonCompSplit1T nonComp1;
          NonCompSplit2T nonComp2;

          // Specialized float-splitting step for the given datatype
          UpdateCompAndHist<FT>::update(v[i].x[j], comps, nonComp1, nonComp2, warpHistogram2DArr);
          nonCompV1[i].x[j] = nonComp1;
          if (FTI::getIfNonCompSplit())
            nonCompV2[i].x[j] = nonComp2;

          // Except for Float64, this only loops once. For Float64, this loops
          // twice.
          for (int k = 0; k < numCompSegments; ++k) {
            compV[k*roundUp(kOuterUnroll, 16 / sizeof(CompVecT)) + i].x[j] = comps[k];
          }
        }
      }

#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
        for (int k = 0; k < numCompSegments; ++k) {
          compOutsV[i*Threads + k*compDatasetStrideVec] = compV[k*roundUp(kOuterUnroll, 16 / sizeof(CompVecT)) + i];
        }

        nonCompOutV1[i * Threads] = nonCompV1[i];
        if (FTI::getIfNonCompSplit())
          nonCompOutV2[i * Threads] = nonCompV2[i];
      }
    }

    // Handle last (partial) block
    for (uint32_t i =
             fullBlocks * kFloatsPerBlock + blockIdx.x * Threads + threadIdx.x;
         i < size;
         i += gridDim.x * Threads) {
      CompT comps[MAX_NUM_COMP_OUTS];
      NonCompSplit1T nonComp1;
      NonCompSplit2T nonComp2;

      UpdateCompAndHist<FT>::update(in[i], comps, nonComp1, nonComp2, warpHistogram2DArr);
      nonCompOut1[i] = nonComp1;
      if (FTI::getIfNonCompSplit())
        nonCompOut2[i] = nonComp2;

      for (int k = 0; k < numCompSegments; k++) {
        compOuts[i + k*compDatasetStride] = comps[k];
      }
    }
  }
};

/*
 * 
 */
template <
    typename InProvider,
    typename NonCompProvider,
    FloatType FT,
    int Threads>
__global__ void splitFloat(
    // BatchProvider with input float data
    InProvider inProvider,

    // Whether to use a checksum for verification
    bool useChecksum,
    const uint32_t* __restrict__ checksum,

    // Output array of all data that will be ANS compressed (aka, the
    // exponents).
    void* __restrict__ compOuts,

    // The number bytes between different batches in compOuts. Not to be
    // confused with compDatasetStride, which is Float64-specific.
    uint32_t compOutStride,

    // For Float64, there are two separate ANS datasets. For each batch, the
    // second ANS dataset is compDatasetStride bytes after the start of the
    // first dataset. For other float types, this is irrelevant.
    uint32_t compDatasetStride,

    // For Float64 compression, the second ANS histogram will be
    // histDatasetStride elements after the beginning of histogramsOut.
    // For other float types, this is irrelevant.
    uint32_t histDatasetStride,

    // BatchProvider that tells us where to write non-compressed data
    NonCompProvider nonCompProvider,

    // Histogram(s) for ANS compression.
    uint32_t* __restrict__ histogramsOut) {
  using FTI = FloatTypeInfo<FT>;
  using WordT = typename FloatTypeInfo<FT>::WordT;
  using CompT = typename FloatTypeInfo<FT>::CompT;
  using NonCompT = typename FloatTypeInfo<FT>::NonCompT;

  constexpr int kWarps = Threads / kWarpSize;
  int numHists = FTI::getNumCompSegments();
  static_assert(Threads == kNumSymbols, "");

  int batch = blockIdx.y;
  int warpId = threadIdx.x / kWarpSize;

  uint32_t* curHistsOut = histogramsOut + batch * kNumSymbols;

  checksum += batch;

  // Give each warp its own ANS histogram. Float64 requires two different ANS
  // histograms, so allocate enough room for each warp to have two histograms.
  //
  // Each histogram is of size kNumSymbols + 1 in order to force very common
  // symbols that could overlap into different banks between different warps.
  //
  // This makes it more efficient for each warp to atomically update 
  // histogramsOut at the end (reducing the number of threads trying to write
  // to the same part of memory at once).
  __shared__ uint32_t histogram[kWarps][MAX_NUM_COMP_OUTS * roundUp(kNumSymbols + 1, 4)];
#pragma unroll
  for (int i = 0; i < kWarps; ++i) {
    // Set all histogram bins to zero
    for (int k = 0; k < numHists; k++) { // For Float64, also set the second histogram
        histogram[i][threadIdx.x + k*roundUp(kNumSymbols + 1, 4)] = 0;
    }
  }

  __syncthreads();

  // Histogram for this particular warp
  uint32_t* warpHistograms = histogram[warpId];

  // Inputs and outputs for the current batch
  auto curIn = (const WordT*)inProvider.getBatchStart(batch);
  auto headerOut = (GpuFloatHeader*)nonCompProvider.getBatchStart(batch);

  CompT* curCompOuts = (CompT*) compOuts + compOutStride * batch;
  auto curSize = inProvider.getBatchSize(batch);

  // Write size as a header
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    GpuFloatHeader h;
    h.setMagicAndVersion();
    h.size = curSize;
    h.setFloatType(FT);
    h.setUseChecksum(useChecksum);

    if (useChecksum) {
      h.setChecksum(*checksum);
    }

    *headerOut = h;
  }

  auto curNonCompOut = (NonCompT*)(headerOut + 2);

  // How many bytes are before the point where we are 16 byte aligned?
  auto nonAlignedBytes = getAlignmentRoundUp<sizeof(uint4)>(curIn);

  if (nonAlignedBytes > 0) {
    SplitFloatNonAligned<FT, Threads>::split(
        curIn, curSize, curCompOuts, curNonCompOut, warpHistograms, compDatasetStride, roundUp(kNumSymbols + 1, 4));
  } else {
    SplitFloatAligned16<FT, Threads>::split(
        curIn, curSize, curCompOuts, curNonCompOut, warpHistograms, compDatasetStride, roundUp(kNumSymbols + 1, 4));
  }

  // Accumulate warp histogram data and write into the gmem histogram
  __syncthreads();

  uint32_t sums[MAX_NUM_COMP_OUTS] = {histogram[0][threadIdx.x], histogram[0][threadIdx.x + roundUp(kNumSymbols + 1, 4)]};
#pragma unroll
  for (int j = 1; j < kWarps; ++j) {
    for (int k = 0; k < numHists; k++) {
      sums[k] += histogram[j][threadIdx.x + k*roundUp(kNumSymbols + 1, 4)];
    }
  }


  // The count for the thread's bucket could be 0
  for (int k = 0; k < numHists; k++) {
    if (sums[k]) {
      atomicAdd(&curHistsOut[threadIdx.x + k*histDatasetStride], sums[k]);
    }
  }
}

// Update the final byte counts for the batch to take into account the
// uncompressed and compressed portions. outSize is presumed to have the
// number bytes output by ANS, so we need to add the overhead from headers
// and non-compressed bytes.
template <FloatType FT, typename InProvider>
__global__ void
incOutputSizes(InProvider inProvider, uint32_t* outSize, uint32_t numInBatch) {
  uint32_t batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch < numInBatch) {
    outSize[batch] += sizeof(GpuFloatHeader) + sizeof(GpuFloatHeader2) +
        FloatTypeInfo<FT>::getUncompDataSize(inProvider.getBatchSize(batch));
  }
}

// For Float64 compression, there are two different ANS compression stages.
// After the second stage, outSize needs to be updated with the size of
// the second ANS-compressed output.
__global__ void
incOutputSizesF64(uint32_t* outSize, uint32_t* secondANSOutSize, uint32_t numInBatch) {
  uint32_t batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch < numInBatch) {
    outSize[batch] += secondANSOutSize[batch];
  }
}

// Provides the input data to ANS compression
template <typename SizeProvider>
struct FloatANSInProvider {
  using Writer = BatchWriter;

  __host__
  FloatANSInProvider(void* ptr_dev, uint32_t stride, SizeProvider& sizeProvider)
      : ptr_dev_(ptr_dev), stride_(stride), sizeProvider_(sizeProvider) {}

  __device__ void* getBatchStart(uint32_t batch) {
    return (uint8_t*)ptr_dev_ + batch * stride_;
  }

  __device__ const void* getBatchStart(uint32_t batch) const {
    return (uint8_t*)ptr_dev_ + batch * stride_;
  }

  __device__ BatchWriter getWriter(uint32_t batch) {
    return BatchWriter(getBatchStart(batch));
  }

  __device__ uint32_t getBatchSize(uint32_t batch) {
    return sizeProvider_.getBatchSize(batch);
  }

  void* ptr_dev_;
  uint32_t stride_;
  SizeProvider sizeProvider_;
};

// Provides the output data to ANS compression
template <FloatType FT, typename OutProvider, typename SizeProvider>
struct FloatANSOutProvider {
  using Writer = BatchWriter;
  using FTI = FloatTypeInfo<FT>;

  __host__ FloatANSOutProvider(
      OutProvider& outProvider,
      SizeProvider& sizeProvider,

      // This will be an array of all zeros, except for the second round of
      // ANS in Float64 compression, where it will be the number of bytes in
      // the first ANS-compressed segment.
      uint32_t* offsets)
      : outProvider_(outProvider), sizeProvider_(sizeProvider), offsets_(offsets) {
      }

  __device__ void* getBatchStart(uint32_t batch) {
    uint8_t* p = (uint8_t*)outProvider_.getBatchStart(batch);

    // Increment the pointer to past the floating point data
    ((GpuFloatHeader*)p)->checkMagicAndVersion();
    return p + offsets_[batch] + sizeof(GpuFloatHeader) + sizeof(GpuFloatHeader2) +
        FTI::getUncompDataSize(sizeProvider_.getBatchSize(batch));
  }

  __device__ const void* getBatchStart(uint32_t batch) const {
    const uint8_t* p = (const uint8_t*)outProvider_.getBatchStart(batch);

    // Increment the pointer to past the floating point data
    ((GpuFloatHeader*)p)->checkMagicAndVersion();
    return p + offsets_[batch] + sizeof(GpuFloatHeader) + sizeof(GpuFloatHeader2) +
        FTI::getUncompDataSize(sizeProvider_.getBatchSize(batch));
  }

  __device__ BatchWriter getWriter(uint32_t batch) {
    return BatchWriter(getBatchStart(batch));
  }

  OutProvider outProvider_;
  SizeProvider sizeProvider_;
  uint32_t* offsets_;
};

// For Float64 Compression, update the second float header and the ansOutOffset
// array to both include the size of the first ANS-compressed output. The header
// will be used in decompression, and ansOutOffset provides the "offset" input to
// FloatANSOutProvider.
template <typename InProvider, typename OutProvider, FloatType FT>
__global__ void setHeaderAndANSOutOffset(OutProvider outProvider, 
      FloatANSOutProvider<FT, OutProvider, InProvider> outProviderANS, 
      uint32_t* ansOutOffset, uint32_t numInBatch) {
  uint32_t batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch < numInBatch) {
    auto headerOut = ((GpuFloatHeader2*) outProvider.getBatchStart(batch)) + 1;
    ANSCoalescedHeader* ansHeader = (ANSCoalescedHeader*) outProviderANS.getBatchStart(batch);
    ansOutOffset[batch] = roundUp(ansHeader->getTotalCompressedSize(), 16);
    headerOut->setFirstCompSegmentBytes(ansOutOffset[batch]);
  }
}

// Main method, called by GpuFloatCompress.cu
template <typename InProvider, typename OutProvider>
void floatCompressDevice(
    // used for allocating all GPU memory
    StackDeviceMemory& res,

    // Config for float compression. See GpuFloatCodec.h
    const FloatCompressConfig& config,

    // Number of input batches
    uint32_t numInBatch,

    // BatchProvider that tells us where the input floats for each batch are,
    // as well as the number of floats from each batch.
    InProvider& inProvider,

    // Maximum number of floats across the batches
    uint32_t maxSize,

    // BatchProvider that tells us where to write the output (compressed) data.
    OutProvider& outProvider,

    // This will be populated with the number of compressed bytes for each
    // batch
    uint32_t* outSize_dev,

    // CUDA execution stream
    cudaStream_t stream) {

  auto maxUncompressedWords = maxSize / sizeof(ANSDecodedT);
  uint32_t maxNumCompressedBlocks =
      divUp(maxUncompressedWords, kDefaultBlockSize);

  // Compute checksum on input data (optional)
  auto checksum_dev = res.alloc<uint32_t>(stream, numInBatch);

  // not allowed in float mode
  assert(!config.ansConfig.useChecksum);

  if (config.useChecksum) {
    checksumBatch(numInBatch, inProvider, checksum_dev.data(), stream);
  }

  // Temporary space for the extracted exponents; all rows must be 16 byte
  // aligned
  uint32_t compRowStride = roundUp(maxSize, sizeof(uint4));
  
  // For Float64 data, toComp_dev contains two datasets to be ANS compressed,
  // with the start of the second dataset being compDatasetStride after the
  // start of the first. For other floats, there is only one ANS dataset
  // and compDatasetStride is irrelevant.
  uint32_t compDatasetStride = roundUp(numInBatch * compRowStride, 16);
  auto toComp_dev = res.alloc<uint8_t>(stream, compDatasetStride * MAX_NUM_COMP_OUTS);

  // For Float64 compression, this is the size of the second ANS-compressed
  // output. Otherwise, it is unused.
  auto tempOutSize_dev = res.alloc<uint32_t>(stream, numInBatch);

  // This is the "offset" array passed into the FloatANSOutProvider. This will
  // be all zero, except for the second round of ANS compression for Float64,
  // where it will contain the size of the first ANS-compressed output.
  auto ansOutOffset_dev = res.alloc<uint32_t>(stream, numInBatch);

  // We calculate a histogram of the symbols to be compressed as part of
  // extracting the compressible symbol from the float
  //
  // For Float64 data, there are two different ANS histograms, the second
  // being histDatasetStride elements after the start of the first.
  uint32_t histDatasetStride = roundUp(numInBatch * kNumSymbols, 4);
  auto histograms_dev = res.alloc<uint32_t>(stream, histDatasetStride * MAX_NUM_COMP_OUTS);

  // zero out buckets before proceeding, as we aggregate with atomic adds
  CUDA_VERIFY(cudaMemsetAsync(
      histograms_dev.data(),
      0,
      sizeof(uint32_t) * histDatasetStride * MAX_NUM_COMP_OUTS,
      stream));

  // Also zero out the ansOutOffset_dev array.
  CUDA_VERIFY(cudaMemsetAsync(
      ansOutOffset_dev.data(),
      0,
      sizeof(uint32_t) * numInBatch,
      stream));

  // Any code that requires the float type (i.e., the element of the enum
  // FloatType corresponding the the datatype we're compressing) to be a
  // constant, i.e., when the float type is passed in as a template argument,
  // must be in this sort of macro.
#define RUN_SPLIT(FLOAT_TYPE)                                      \
  do {                                                             \
    constexpr int kBlock = 256;                                    \
    auto& props = getCurrentDeviceProperties();                    \
    int maxBlocksPerSM = 0;                                        \
    CUDA_VERIFY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(     \
        &maxBlocksPerSM,                                           \
        splitFloat<InProvider, OutProvider, FLOAT_TYPE, kBlock>,   \
        kBlock,                                                    \
        0));                                                       \
    uint32_t maxGrid = maxBlocksPerSM * props.multiProcessorCount; \
    uint32_t perBatchGrid = 4 * divUp(maxGrid, numInBatch);        \
    auto grid = dim3(perBatchGrid, numInBatch);                    \
                                                                   \
    splitFloat<InProvider, OutProvider, FLOAT_TYPE, kBlock>        \
        <<<grid, kBlock, 0, stream>>>(                             \
            inProvider,                                            \
            config.useChecksum,                                    \
            checksum_dev.data(),                                   \
            toComp_dev.data(),                                     \
            compRowStride,                                         \
            compDatasetStride,                                     \
            histDatasetStride,                                     \
            outProvider,                                           \
            histograms_dev.data());                                \
  } while (false)

  switch (config.floatType) {
    case FloatType::kFloat16:
      RUN_SPLIT(FloatType::kFloat16);
      break;
    case FloatType::kBFloat16:
      RUN_SPLIT(FloatType::kBFloat16);
      break;
    case FloatType::kFloat32:
      RUN_SPLIT(FloatType::kFloat32);
      break;
    case FloatType::kFloat64:
      RUN_SPLIT(FloatType::kFloat64);
      break;
    default:
      assert(false);
      break;
  }

#undef RUN_SPLIT


uint32_t compSegment = 0; 
#define RUN_ANS(FT, nCompSegments)                                          \
  compSegment = 0; /* which round of ANS compression is this? */            \
  do {                                                                      \
    auto inProviderANS = FloatANSInProvider<InProvider>(                    \
        toComp_dev.data() + compSegment * compDatasetStride,                \
        compRowStride, inProvider);                                         \
                                                                            \
    auto outProviderANS = FloatANSOutProvider<FT, OutProvider, InProvider>( \
      outProvider, inProvider, ansOutOffset_dev.data());                    \
                                                                            \
    uint32_t* outSizes = (compSegment == 0) ? outSize_dev :                 \
                              tempOutSize_dev.data();                       \
                                                                            \
    ansEncodeBatchDevice(                                                   \
        res,                                                                \
        config.ansConfig,                                                   \
        numInBatch,                                                         \
        inProviderANS,                                                      \
        histograms_dev.data() + compSegment * histDatasetStride,            \
        maxSize,                                                            \
        outProviderANS,                                                     \
        outSizes,                                                           \
        stream);                                                            \
                                                                            \
    /* outSize as reported by ansEncode is just the ANS-encoded portion */  \
    /* of the data. We need to increment the sizes by the uncompressed */   \
    /* portion (header plus uncompressed float data) with incOutputSizes */ \
    if (compSegment == 0) {                                                 \
        incOutputSizes<FT><<<divUp(numInBatch, 128), 128, 0,                \
            stream>>>(inProvider, outSize_dev, numInBatch);                 \
        setHeaderAndANSOutOffset<InProvider, OutProvider, FT>               \
                                <<<divUp(numInBatch, 128), 128, 0,          \
                                    stream>>> ( outProvider, outProviderANS,\
                                    ansOutOffset_dev.data(), numInBatch);   \
    }                                                                       \
    else /* Update outSize with the size of the second dataset */           \
        incOutputSizesF64<<<divUp(numInBatch, 128), 128, 0,                 \
            stream>>>(outSize_dev, tempOutSize_dev.data(), numInBatch);     \
                                                                            \
  } while (++compSegment < nCompSegments)

  // We have written the non-compressed portions of the floats into the output,
  // along with a header that indicates how many floats there are.
  // For compression, we need to increment the address in which the compressed
  // outputs are written.

  switch (config.floatType) {
    case FloatType::kFloat16:
      RUN_ANS(FloatType::kFloat16, 1);
      break;
    case FloatType::kBFloat16:
      RUN_ANS(FloatType::kBFloat16, 1);
      break;
    case FloatType::kFloat32:
      RUN_ANS(FloatType::kFloat32, 1);
      break;
    case FloatType::kFloat64:
      RUN_ANS(FloatType::kFloat64, 2);
      break;
    default:
      assert(false);
      break;
  }

#undef RUN_ANS

  CUDA_TEST_ERROR();
}

} // namespace dietgpu
