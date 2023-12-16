/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/ans/GpuANSDecode.cuh"
#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuFloatInfo.cuh"
#include "dietgpu/float/GpuFloatUtils.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"

#include <glog/logging.h>
#include <cmath>
#include <sstream>
#include <vector>

// Each float will have at most two bytes to be compressed
// (and this is only the case for Float64; the rest only have one byte to be
// compressed).
#define MAX_NUM_COMP_OUTS 2
namespace dietgpu {

// Join the compressed and non-compressed and non-compressed data into a full
// dataset of floating-point words. There is a different JoinFloatNonAligned
// function for every flavor of float compression; this one works for Float16
// and BFloat16 (which have one compressed byte and one non-compressed byte).
//
// There are also specializations for Float32, which has two separate
// non-compressed datasets (because 24 bytes are left uncompressed, which is
// not a power of two) and Float64, which has both two separate non-compressed
// datasets and two rounds of ANS compression.
template <FloatType FT, int Threads>
struct JoinFloatNonAligned {
  static __device__ void join(
      // Dataset of ANS-decompressed exponents
      const typename FloatTypeInfo<FT>::CompT* __restrict__ compIn,

      // Only used for Float64, which has exponents larger than a byte and
      // therefore requires two separate rounds of ANS compression
      const typename FloatTypeInfo<FT>::CompT* __restrict__ compInUnused,

      // Dataset of non-compressed float sections
      const typename FloatTypeInfo<FT>::NonCompT* __restrict__ nonCompIn,

      // Number of floats
      uint32_t size,

      // Where to write the output data
      typename FloatTypeInfo<FT>::WordT* __restrict__ out) {
    for (uint32_t i = blockIdx.x * Threads + threadIdx.x; i < size;
         i += gridDim.x * Threads) {
      out[i] = FloatTypeInfo<FT>::join(&compIn[i], nonCompIn[i]);
    }
  }
};

// Float32 specialization: two separate non-compressed datasets
template <int Threads>
struct JoinFloatNonAligned<FloatType::kFloat32, Threads> {
  static __device__ void join(
    // Dataset of ANS-decompressed exponents
      const typename FloatTypeInfo<
          FloatType::kFloat32>::CompT* __restrict__ compIn,
      
      // Only used for Float64, which has exponents larger than a byte and
      // therefore requires two separate rounds of ANS compression
      const typename FloatTypeInfo<
          FloatType::kFloat32>::CompT* __restrict__ compInUnused,

      // Dataset of non-compressed float sections
      const typename FloatTypeInfo<
          FloatType::kFloat32>::CompT* __restrict__ compInUnused,
      const typename FloatTypeInfo<
          FloatType::kFloat32>::NonCompT* __restrict__ nonCompIn,

      // Number of floats
      uint32_t size,

      // Where we write the output floats
      typename FloatTypeInfo<FloatType::kFloat32>::WordT* __restrict__ out) {
    using FTI = FloatTypeInfo<FloatType::kFloat32>;
    using CompT = typename FTI::CompT;
    using NonCompT = typename FTI::NonCompT;

    // Where the low order 2 bytes are read
    uint16_t* nonComp2In = (uint16_t*)nonCompIn;

    // Where the high order byte is read
    uint8_t* nonComp1In = (uint8_t*)(nonComp2In + roundUp(size, 8));

    for (uint32_t i = blockIdx.x * Threads + threadIdx.x; i < size;
         i += gridDim.x * Threads) {
      // Full non-compressed section
      uint32_t nc =
          (uint32_t(nonComp1In[i]) * 65536U) + uint32_t(nonComp2In[i]);

      out[i] = FTI::join(&compIn[i], nc);
    }
  }
};

// Float64 specialization: two separate non-compressed datasets,
// and two separate rounds of ANS compression
template <int Threads>
struct JoinFloatNonAligned<FloatType::kFloat64, Threads> {
  static __device__ void join(
      // Second dataset of ANS-decompressed exponents
      const typename FloatTypeInfo<
          FloatType::kFloat64>::CompT* __restrict__ compIn,

      // First dataset of ANS-decompressed exponents
      const typename FloatTypeInfo<
          FloatType::kFloat64>::CompT* __restrict__ compInFirstDataset,

      // Dataset of non-compressed float sections
      const typename FloatTypeInfo<
          FloatType::kFloat64>::NonCompT* __restrict__ nonCompIn,

      // Number of floats
      uint32_t size,

      // Where we erite the output floats
      typename FloatTypeInfo<FloatType::kFloat64>::WordT* __restrict__ out) {
    using FTI = FloatTypeInfo<FloatType::kFloat64>;
    using CompT = typename FTI::CompT;
    using NonCompT = typename FTI::NonCompT;
    

    // Where the low order 4 bytes are read
    uint32_t* nonComp1In = (uint32_t*)nonCompIn;

    // Where the high order 2 bytes are read
    uint16_t* nonComp2In = (uint16_t*)(nonComp1In + roundUp(size, 4));

    for (uint32_t i = blockIdx.x * Threads + threadIdx.x; i < size;
         i += gridDim.x * Threads) {
      // Full non-compressed section
      uint64_t nc = (uint64_t(nonComp2In[i]) * 4294967296U) + uint64_t(nonComp1In[i]);

      CompT twoCompressedBytes[2] = {compInFirstDataset[i], compIn[i]};
      out[i] = FTI::join(twoCompressedBytes, nc);
    }
  }
};

// Join the compressed and non-compressed and non-compressed data into a full
// dataset of floating-point words. The inputs and outputs are guaranteed to
// be 16-byte aligned. This allows us to use vectorized operations, which
// involve writing 16-byte segments at a time, allowing us to take full
// advantage of GPU memory bandwidth.
//
// This function works for Float16; there are also specializations for Float32
// and Float64.
template <FloatType FT, int Threads>
struct JoinFloatAligned16 {
  static __device__ void join(
      // Dataset of ANS-decompressed exponents
      const typename FloatTypeInfo<FT>::CompT* __restrict__ compIn,

      // Only used for Float64, which has exponents larger than a byte and
      // therefore requires two separate rounds of ANS compression
      const typename FloatTypeInfo<FT>::CompT* __restrict__ compInUnused,

      // Dataset of non-compressed float sections
      const typename FloatTypeInfo<FT>::NonCompT* __restrict__ nonCompIn,

      // Number of floats
      uint32_t size,

      // Where we write the output floats
      typename FloatTypeInfo<FT>::WordT* __restrict__ out) {
    using FTI = FloatTypeInfo<FT>;

    using WordT = typename FTI::WordT;
    using CompT = typename FTI::CompT;
    using NonCompT = typename FTI::NonCompT;
    using VecT = typename FTI::VecT;
    using CompVecT = typename FTI::CompVecT;
    using NonCompVecT = typename FTI::NonCompVecT;

    // Loop unrolling for performance
    constexpr int kOuterUnroll = 2;

    // Number of floats written in one 16-byte chunk. This is the size of our
    // vectorized operations.
    constexpr int kInnerUnroll = sizeof(VecT) / sizeof(WordT);

    // Cast inputs and outputs to vectors of size kInnerUnroll.
    const CompVecT* compInV = (const CompVecT*)compIn;
    const NonCompVecT* nonCompInV = (const NonCompVecT*)nonCompIn;
    VecT* outV = (VecT*)out;

    // Each block handles Threads * kOuterUnroll * kInnerUnroll inputs/outputs
    // at a time, or Threads * kOuterUnroll 16-byte words at a time

    constexpr int kWordsPerBlock = Threads * kOuterUnroll;
    constexpr int kFloatsPerBlock = kWordsPerBlock * kInnerUnroll;
    uint32_t fullBlocks = divDown(size, kFloatsPerBlock);

    // Handle by block
    uint32_t startBlock = blockIdx.x * kWordsPerBlock;
    compInV += startBlock + threadIdx.x;
    nonCompInV += startBlock + threadIdx.x;
    outV += startBlock + threadIdx.x;

    for (uint32_t b = blockIdx.x; b < fullBlocks; b += gridDim.x,
                  compInV += gridDim.x * kWordsPerBlock,
                  nonCompInV += gridDim.x * kWordsPerBlock,
                  outV += gridDim.x * kWordsPerBlock) {
      CompVecT comp[kOuterUnroll];
      NonCompVecT nonComp[kOuterUnroll];

#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
        comp[i] = compInV[i * Threads];
        nonComp[i] = nonCompInV[i * Threads];
      }

      // Temporary output floats
      VecT v[kOuterUnroll];

      // Perform float joining for each element of the input vectors we just
      // read in
#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
#pragma unroll
        for (int j = 0; j < kInnerUnroll; ++j) {
          v[i].x[j] = FTI::join(&comp[i].x[j], nonComp[i].x[j]);
        }
      }

      // Vectorized write to output memory
#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
        outV[i * Threads] = v[i];
      }
    }

    // Handle last (partial) block
    for (uint32_t i =
             fullBlocks * kFloatsPerBlock + blockIdx.x * Threads + threadIdx.x;
         i < size;
         i += blockDim.x) {
      out[i] = FTI::join(&compIn[i], nonCompIn[i]);
    }
  }
};

// Float32 specialization for vectorized float joining
template <int Threads>
struct JoinFloatAligned16<FloatType::kFloat32, Threads> {
  static __device__ void join(
      // Dataset of ANS-decompressed exponents
      const typename FloatTypeInfo<
          FloatType::kFloat32>::CompT* __restrict__ compIn,

      // Only used for Float64, which has exponents larger than a byte and
      // therefore requires two separate rounds of ANS compression
      const typename FloatTypeInfo<
          FloatType::kFloat32>::CompT* __restrict__ compInUnused,

      // Dataset of non-compressed float sections
      const typename FloatTypeInfo<
          FloatType::kFloat32>::CompT* __restrict__ compInUnused,
      const typename FloatTypeInfo<
          FloatType::kFloat32>::NonCompT* __restrict__ nonCompIn,

      // Number of floats
      uint32_t size,

      // Where we write the output floats
      typename FloatTypeInfo<FloatType::kFloat32>::WordT* __restrict__ out) {
    using FTI = FloatTypeInfo<FloatType::kFloat32>;

    using WordT = typename FTI::WordT;
    using CompT = typename FTI::CompT;
    using NonCompT = typename FTI::NonCompT;

    constexpr int kOuterUnroll = 1;
    // Number of floats written in one 16-byte chunk. This is the size of our
    // vectorized operations.
    constexpr int kInnerUnroll = sizeof(uint32x4) / sizeof(uint32_t);

    // Cast inputs and outputs to vectors of size kInnerUnroll.
    auto compInV = (const uint8x4*)compIn;
    auto nonCompIn2 = (const uint16_t*)nonCompIn;
    auto nonCompIn1 = (const uint8_t*)(nonCompIn2 + roundUp(size, 8));

    auto nonCompInV2 = (uint16x4*)nonCompIn2;
    auto nonCompInV1 = (uint8x4*)nonCompIn1;

    auto outV = (uint32x4*)out;

    // Each block handles Threads * kOuterUnroll * kInnerUnroll inputs/outputs
    // at a time, or Threads * kOuterUnroll 16-byte words at a time
    constexpr int kWordsPerBlock = Threads * kOuterUnroll;
    constexpr int kFloatsPerBlock = kWordsPerBlock * kInnerUnroll;
    uint32_t fullBlocks = divDown(size, kFloatsPerBlock);

    // Handle by block
    uint32_t startBlock = blockIdx.x * kWordsPerBlock;
    compInV += startBlock + threadIdx.x;
    nonCompInV2 += startBlock + threadIdx.x;
    nonCompInV1 += startBlock + threadIdx.x;
    outV += startBlock + threadIdx.x;

    for (uint32_t b = blockIdx.x; b < fullBlocks; b += gridDim.x,
                  compInV += gridDim.x * kWordsPerBlock,
                  nonCompInV2 += gridDim.x * kWordsPerBlock,
                  nonCompInV1 += gridDim.x * kWordsPerBlock,
                  outV += gridDim.x * kWordsPerBlock) {
      uint8x4 comp[kOuterUnroll];
      uint16x4 nonComp2[kOuterUnroll];
      uint8x4 nonComp1[kOuterUnroll];

      // Gather compressed and non-compressed data
#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
        comp[i] = compInV[i * Threads];
        nonComp2[i] = nonCompInV2[i * Threads];
        nonComp1[i] = nonCompInV1[i * Threads];
      }

      uint32x4 nonComp[kOuterUnroll];
#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
#pragma unroll
        for (int j = 0; j < kInnerUnroll; ++j) {
          nonComp[i].x[j] = nonComp1[i].x[j] * 65536U + nonComp2[i].x[j];
        }
      }

      // Temporary storage for output floats
      uint32x4 v[kOuterUnroll];

      // Perform float joining for each element of the input vectors we just
      // read in
#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
#pragma unroll
        for (int j = 0; j < kInnerUnroll; ++j) {
          v[i].x[j] = FTI::join(&comp[i].x[j], nonComp[i].x[j]);
        }
      }

      // Vectorized write to output memory
#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
        outV[i * Threads] = v[i];
      }
    }

    // Handle last (partial) block
    for (uint32_t i =
             fullBlocks * kFloatsPerBlock + blockIdx.x * Threads + threadIdx.x;
         i < size;
         i += blockDim.x) {
      uint32_t nc2 = nonCompIn2[i];
      uint32_t nc1 = nonCompIn1[i];
      uint32_t nc = nc1 * 65536U + nc2;

      out[i] = FTI::join(&compIn[i], nc);
    }
  }
};


// Float32 specialization for vectorized float joining
template <int Threads>
struct JoinFloatAligned16<FloatType::kFloat64, Threads> {
  static __device__ void join(
      // Second dataset of ANS-decompressed exponents
      const typename FloatTypeInfo<
          FloatType::kFloat64>::CompT* __restrict__ compIn,

      // First dataset of ANS-decompressed exponents
      const typename FloatTypeInfo<
          FloatType::kFloat64>::CompT* __restrict__ compInFirstDataset,

      // Dataset of non-compressed float sections
      const typename FloatTypeInfo<
          FloatType::kFloat64>::NonCompT* __restrict__ nonCompIn,

      // Number of floats
      uint32_t size,

      // Where we write the output floats
      typename FloatTypeInfo<FloatType::kFloat64>::WordT* __restrict__ out) {
    using FTI = FloatTypeInfo<FloatType::kFloat64>;

    using WordT = typename FTI::WordT;
    using CompT = typename FTI::CompT;
    using NonCompT = typename FTI::NonCompT;

    constexpr int kOuterUnroll = 1;
    // Number of floats written in one 16-byte chunk. This is the size of our
    // vectorized operations.
    constexpr int kInnerUnroll = sizeof(uint64x2) / sizeof(uint64_t);

    // Cast inputs and outputs to vectors of size kInnerUnroll.
    auto compInV1 = (const uint8x2*)compIn;
    auto compInV2 = (const uint8x2*)compInFirstDataset;
    auto nonCompIn2 = (const uint32_t*)nonCompIn;
    auto nonCompIn1 = (const uint16_t*)(nonCompIn2 + roundUp(size, 4));

    auto nonCompInV2 = (uint32x2*)nonCompIn2;
    auto nonCompInV1 = (uint16x2*)nonCompIn1;

    auto outV = (uint64x2*)out;

    // Each block handles Threads * kOuterUnroll * kInnerUnroll inputs/outputs
    // at a time, or Threads * kOuterUnroll 16-byte words at a time
    constexpr int kWordsPerBlock = Threads * kOuterUnroll;
    constexpr int kFloatsPerBlock = kWordsPerBlock * kInnerUnroll;
    uint32_t fullBlocks = divDown(size, kFloatsPerBlock);

    // Handle by block
    uint32_t startBlock = blockIdx.x * kWordsPerBlock;
    compInV1 += startBlock + threadIdx.x;
    compInV2 += startBlock + threadIdx.x;
    nonCompInV2 += startBlock + threadIdx.x;
    nonCompInV1 += startBlock + threadIdx.x;
    outV += startBlock + threadIdx.x;

    for (uint32_t b = blockIdx.x; b < fullBlocks; b += gridDim.x,
                  compInV1 += gridDim.x * kWordsPerBlock,
                  compInV2 += gridDim.x * kWordsPerBlock,
                  nonCompInV2 += gridDim.x * kWordsPerBlock,
                  nonCompInV1 += gridDim.x * kWordsPerBlock,
                  outV += gridDim.x * kWordsPerBlock) {
      uint8x2 comp1[kOuterUnroll];
      uint8x2 comp2[kOuterUnroll];
      uint32x2 nonComp2[kOuterUnroll];
      uint16x2 nonComp1[kOuterUnroll];

      // Gather compressed and non-compressed data
#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
        comp1[i] = compInV1[i * Threads];
        comp2[i] = compInV2[i * Threads];
        nonComp2[i] = nonCompInV2[i * Threads];
        nonComp1[i] = nonCompInV1[i * Threads];
      }

      uint64x2 nonComp[kOuterUnroll];
#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
#pragma unroll
        for (int j = 0; j < kInnerUnroll; ++j) {
          nonComp[i].x[j] = uint64_t(nonComp1[i].x[j]) * 4294967296U + uint64_t(nonComp2[i].x[j]);
        }
      }

      // Temporary storage for output floats
      uint64x2 v[kOuterUnroll];

      // Perform float joining for each element of the input vectors we just
      // read in
#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
#pragma unroll
        for (int j = 0; j < kInnerUnroll; ++j) {
          CompT twoCompressedBytes[2] = {comp2[i].x[j], comp1[i].x[j]};
          v[i].x[j] = FTI::join(twoCompressedBytes, nonComp[i].x[j]);
        }
      }

      // Vectorized write to output memory
#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
        outV[i * Threads] = v[i];
      }
    }

    // Handle last (partial) block
    for (uint32_t i =
             fullBlocks * kFloatsPerBlock + blockIdx.x * Threads + threadIdx.x;
         i < size;
         i += blockDim.x) {
      uint64_t nc = uint64_t(nonCompIn1[i]) * 4294967296U + uint64_t(nonCompIn2[i]);
      CompT twoCompressedBytes[2] = {compInFirstDataset[i], compIn[i]};
      out[i] = FTI::join(twoCompressedBytes, nc);
    }
  }
};

// Join floats, calling either JoinFloatNonAligned or JoinFloatAligned16,
// depending on whether the inputs or outputs are 16-byte aligned.
template <FloatType FT, int Threads>
struct JoinFloatImpl {
  static __device__ void join(
      // For anything except Float64, this is the dataset of ANS-decompressed
      // exponents. For Float64, this is the output of the second round of
      // ANS decompression
      const typename FloatTypeInfo<FT>::CompT* compIn,

      // For anything but Float64, this is the same as compIn. For Float64,
      // this is the output of the first round of ANS decompression.
      const typename FloatTypeInfo<FT>::CompT* compInFirstDataset,

      // Dataset of non-compressed float sections
      const typename FloatTypeInfo<FT>::NonCompT* nonCompIn,

      // Number of floats
      uint32_t size,

      // Where to write the output floats
      typename FloatTypeInfo<FT>::WordT* out) {
    // compIn should always be aligned, as we decompress into temporary memory
    auto compUnalignedBytes = getAlignmentRoundUp<sizeof(uint4)>(compIn);
    auto nonCompUnalignedBytes = getAlignmentRoundUp<sizeof(uint4)>(nonCompIn);
    auto outUnalignedBytes = getAlignmentRoundUp<sizeof(uint4)>(out);
    // JoinFloatNonAligned<FT, Threads>::join(compIn, compInFirstDataset, nonCompIn, size, out);
    if (compUnalignedBytes || nonCompUnalignedBytes || outUnalignedBytes) {
      JoinFloatNonAligned<FT, Threads>::join(compIn, compInFirstDataset, nonCompIn, size, out);
      
    } else {
      JoinFloatAligned16<FT, Threads>::join(compIn, compInFirstDataset, nonCompIn, size, out);
    }
  }
};

// Join the compressed and non-compressed sections into full floats, for
// each batch of float data
template <
    typename InProviderComp,
    typename InProviderNonComp,
    typename OutProvider,
    FloatType FT,
    int Threads>
__global__ void joinFloat(
    // For anything but Float64, this is the BatchProvider for the
    // ANS-decompressed exponents. For Float64, this is the BatchProvider for
    // the output of the second ANS decompression round.
    InProviderComp inProviderComp,

    // For anything but Float64, this is the same as inProviderComp. For Float64,
    // this is the BatchProvider for the output of the first ANS
    // decompression round
    InProviderComp inProviderCompFirstDataset,

    // BatchProvider for the non-compressed data
    InProviderNonComp inProviderNonComp,

    // BatchProvider for the output data
    OutProvider outProvider,

    // Whether ANS decompression was successful, for each batch
    uint8_t* __restrict__ outSuccess,

    // ANS decompression output size, for each batch
    uint32_t* __restrict__ outSize) {
  using FTI = FloatTypeInfo<FT>;
  using WordT = typename FTI::WordT;
  using CompT = typename FTI::CompT;
  using NonCompT = typename FTI::NonCompT;

  int batch = blockIdx.y;

  // Inputs and outputs for the current batch
  auto curCompIn = (const CompT*)inProviderComp.getBatchStart(batch);
  auto curCompInFirstDataset = (const CompT*) inProviderCompFirstDataset.getBatchStart(batch);
  auto curHeaderIn =
      (const GpuFloatHeader*)inProviderNonComp.getBatchStart(batch);
  auto curOut = (WordT*)outProvider.getBatchStart(batch);

  // FIXME: test out capacity

  if (outSuccess && !outSuccess[batch]) {
    // ANS decompression failed, so nothing for us to do
    return;
  }

  // Get size as a header
  GpuFloatHeader h = *curHeaderIn;
  h.checkMagicAndVersion();

  auto curSize = h.size;

  if (outSize && (curSize != outSize[batch])) {
    // Reported size mismatch between ANS decompression and fp unpacking
    assert(false);
    return;
  }

  auto curNonCompIn = (const NonCompT*)(curHeaderIn + 2);

  JoinFloatImpl<FT, Threads>::join(curCompIn, curCompInFirstDataset, curNonCompIn, curSize, curOut);
}

// BatchProvider for the input to the ANS decompressor.
// The data to be ANS decompressed is stored after the floating-point
// compression headers and the the non-compressed data.
template <FloatType FT, typename InProvider>
struct FloatANSProvider {
  using FTI = FloatTypeInfo<FT>;

  __host__ FloatANSProvider(InProvider& provider) : inProvider_(provider) {}

  __device__ void* getBatchStart(uint32_t batch) {
    uint8_t* p = (uint8_t*)inProvider_.getBatchStart(batch);

    // This is the first place that touches the header
    GpuFloatHeader h = *((GpuFloatHeader*)p);
    h.checkMagicAndVersion();
    assert(FT == h.getFloatType());

    // Increment the pointer to past the floating point data
    return p + sizeof(GpuFloatHeader) + sizeof(GpuFloatHeader2) + FTI::getUncompDataSize(h.size);
  }

  __device__ const void* getBatchStart(uint32_t batch) const {
    const uint8_t* p = (const uint8_t*)inProvider_.getBatchStart(batch);

    // This is the first place that touches the header
    GpuFloatHeader h = *((const GpuFloatHeader*)p);
    h.checkMagicAndVersion();
    assert(FT == h.getFloatType());

    // Increment the pointer to past the floating point data
    return p + sizeof(GpuFloatHeader) + sizeof(GpuFloatHeader2) + FTI::getUncompDataSize(h.size);
  }

  InProvider inProvider_;
};

// This is the BatchProvider for the second round of ANS compression for Float64.
// The data to be decompressed is directly after the first dataset of
// ANS-compressed data, the size of which is given by the offsets input array.
template <FloatType FT, typename InProvider>
struct FloatANSProviderOffset {
  using FTI = FloatTypeInfo<FT>;

  __host__ FloatANSProviderOffset(InProvider& provider, uint32_t* offsets) : inProvider_(provider), offsets_(offsets) {}

  __device__ void* getBatchStart(uint32_t batch) {
    uint8_t* p = (uint8_t*)inProvider_.getBatchStart(batch);

    // This is the first place that touches the header
    GpuFloatHeader h = *((GpuFloatHeader*)p);
    h.checkMagicAndVersion();
    assert(FT == h.getFloatType());

    // Increment the pointer to past the floating point data
    return p + offsets_[batch] + sizeof(GpuFloatHeader) + sizeof(GpuFloatHeader2) + FTI::getUncompDataSize(h.size);
  }

  __device__ const void* getBatchStart(uint32_t batch) const {
    const uint8_t* p = (const uint8_t*)inProvider_.getBatchStart(batch);

    // This is the first place that touches the header
    GpuFloatHeader h = *((const GpuFloatHeader*)p);
    h.checkMagicAndVersion();
    assert(FT == h.getFloatType());

    // Increment the pointer to past the floating point data
    return p + offsets_[batch] + sizeof(GpuFloatHeader) + sizeof(GpuFloatHeader2) + FTI::getUncompDataSize(h.size);
  }

  InProvider inProvider_;
  uint32_t* offsets_;
};

// Inline version of FloatANSProvider
template <FloatType FT, int N>
struct FloatANSProviderInline {
  using FTI = FloatTypeInfo<FT>;

  __host__ FloatANSProviderInline(int num, const void** in) {
    CHECK_LE(num, N);
    for (int i = 0; i < num; ++i) {
      in_[i] = in[i];
    }
  }

  __device__ void* getBatchStart(uint32_t batch) {
    uint8_t* p = (uint8_t*)in_[batch];

    // This is the first place that touches the header
    GpuFloatHeader h = *((GpuFloatHeader*)p);
    h.checkMagicAndVersion();
    assert(FT == h.getFloatType());

    // Increment the pointer to past the floating point data
    return p + sizeof(GpuFloatHeader) + sizeof(GpuFloatHeader2) + FTI::getUncompDataSize(h.size);
  }

  __device__ const void* getBatchStart(uint32_t batch) const {
    const uint8_t* p = (const uint8_t*)in_[batch];

    // This is the first place that touches the header
    GpuFloatHeader h = *((const GpuFloatHeader*)p);
    h.checkMagicAndVersion();
    assert(FT == h.getFloatType());

    // Increment the pointer to past the floating point data
    return p + sizeof(GpuFloatHeader) + sizeof(GpuFloatHeader2) + FTI::getUncompDataSize(h.size);
  }

  const void* in_[N];
};

// For the fused kernel implementation: this writer for the ANS Output Batch
// Provider automatically joins the (decompressed) exponents with the
// (non-compressed) significands.
template <FloatType FT, uint32_t BlockSize>
struct JoinFloatWriter {
  using FTI = FloatTypeInfo<FT>;

  __host__ __device__ JoinFloatWriter(
      uint32_t size,
      typename FTI::WordT* out,
      const typename FTI::NonCompT* nonComp)
      : out_(out),
        nonComp_(nonComp),
        outBlock_(nullptr),
        nonCompBlock_(nullptr) {}

  __host__ __device__ void setBlock(uint32_t block) {
    outBlock_ = out_ + block * BlockSize;
    nonCompBlock_ = nonComp_ + block * BlockSize;
  }

  __device__ void write(uint32_t offset, uint8_t sym) {
    auto nonComp = nonCompBlock_[offset];
    outBlock_[offset] = FTI::join(&sym, nonComp);
  }

  // // The preload is an offset of a NonCompVec4
  // __device__ void preload(uint32_t offset) {
  //   // We can preload this before decompressing all of the ANS compressed
  //   data
  //   // to hide memory latency
  //   preload_ = ((typename FTI::NonCompVec4*)nonCompBlock_)[offset];
  // }

  //   __device__ void writeVec(uint32_t offset, ANSDecodedTx4 symV) {
  //     typename FTI::Vec4 outV;
  // #pragma unroll
  //     // We always receive 4 decoded values each iteration
  //     // FIXME: this is hacky
  //     for (int i = 0; i < 4; ++i) {
  //       outV.x[i] = JoinFloat<FT>::join(symV.x[i], preload_.x[i]);
  //     }

  //     ((typename FTI::Vec4*)outBlock_)[offset] = outV;
  //   }

  // typename FTI::NonCompVec4 preload_;
  typename FTI::WordT* out_;
  const typename FTI::NonCompT* nonComp_;
  typename FTI::WordT* outBlock_;
  const typename FTI::NonCompT* nonCompBlock_;
};

// Float32 specialization of the JoinFloatWriter
template <uint32_t BlockSize>
struct JoinFloatWriter<FloatType::kFloat32, BlockSize> {
  static constexpr bool kVectorize = false;
  using FTI = FloatTypeInfo<FloatType::kFloat32>;

  __host__ __device__ JoinFloatWriter(
      uint32_t size,
      typename FTI::WordT* out,
      const typename FTI::NonCompT* nonComp)
      : size_(size),
        out_(out),
        nonComp_(nonComp),
        outBlock_(nullptr),
        nonCompBlock2_(nullptr),
        nonCompBlock1_(nullptr) {}

  __host__ __device__ void setBlock(uint32_t block) {
    nonCompBlock2_ = (const uint16_t*)nonComp_ + block * BlockSize;
    nonCompBlock1_ =
        (const uint8_t*)((const uint16_t*)nonComp_ + roundUp(size_, 8U)) +
        block * BlockSize;
    outBlock_ = out_ + block * BlockSize;
  }

  __device__ void write(uint32_t offset, uint8_t sym) {
    uint32_t nc = uint32_t(nonCompBlock1_[offset]) * 65536U +
        uint32_t(nonCompBlock2_[offset]);

    outBlock_[offset] = FTI::join(&sym, nc);
  }

  // // This implementation does not preload
  // __device__ void preload(uint32_t offset) {
  // }

  // // This implementation does not vectorize
  // __device__ void writeVec(uint32_t offset, ANSDecodedTx4 symV) {
  // }

  uint32_t size_;
  typename FTI::WordT* out_;
  const typename FTI::NonCompT* nonComp_;
  typename FTI::WordT* outBlock_;
  const uint16_t* nonCompBlock2_;
  const uint8_t* nonCompBlock1_;
};

// BatchProvider that specifies where to write the ANS-decompressed exponents
template <
    typename InProvider,
    typename OutProvider,
    FloatType FT,
    uint32_t BlockSize>
struct FloatOutProvider {
  using Writer = JoinFloatWriter<FT, BlockSize>;
  using FTI = FloatTypeInfo<FT>;

  __host__ FloatOutProvider(InProvider& inProvider, OutProvider& outProvider)
      : inProvider_(inProvider), outProvider_(outProvider) {}

  __device__ void* getBatchStart(uint32_t batch) {
    return inProvider_.getBatchStart(batch);
  }

  __device__ uint32_t getBatchSize(uint32_t batch) {
    return outProvider_.getBatchSize(batch);
  }

  __device__ Writer getWriter(uint32_t batch) {
    // Get float header
    auto h = (const GpuFloatHeader*)getBatchStart(batch);

    return Writer(
        h->size,
        (typename FTI::WordT*)outProvider_.getBatchStart(batch),
        // advance past the header
        (const typename FTI::NonCompT*)(h + 2));
  }

  InProvider inProvider_;
  OutProvider outProvider_;
};

// Inline version of FloatOutProvider
template <int N, FloatType FT, uint32_t BlockSize>
struct FloatOutProviderInline {
  using FTI = FloatTypeInfo<FT>;
  using Writer = JoinFloatWriter<FT, BlockSize>;

  __host__ FloatOutProviderInline(
      int num,
      const void** in,
      void** out,
      const uint32_t* outCapacity) {
    CHECK_LE(num, N);
    for (int i = 0; i < num; ++i) {
      in_[i] = in[i];
      out_[i] = out[i];
      outCapacity_[i] = outCapacity[i];
    }
  }

  __device__ void* getBatchStart(uint32_t batch) {
    return in_[batch];
  }

  __device__ uint32_t getBatchSize(uint32_t batch) {
    return outCapacity_[batch];
  }

  __device__ Writer getWriter(uint32_t batch) {
    // Get float header
    auto h = (const GpuFloatHeader*)getBatchStart(batch);

    return Writer(
        h->size,
        (typename FTI::WordT*)out_[batch],
        // advance past the header
        (const typename FTI::NonCompT*)(h + 2));
  }

  const void* in_[N];
  void* out_[N];
  uint32_t outCapacity_[N];
};

// For Float64 compression, this populates an array with the number of bytes in
// the first ANS-compressed section. This allows us to find where the start of
// the second ANS-compressed section is.
template <typename InProvider, FloatType FT>
__global__ void getANSOutOffset(InProvider inProvider, 
      uint32_t* ansOutOffset, uint32_t numInBatch) {
  uint32_t batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch < numInBatch) {
    auto headerIn = ((GpuFloatHeader2*) inProvider.getBatchStart(batch)) + 1;
    ansOutOffset[batch] = headerIn->getFirstCompSegmentBytes();
  }
}

// Main method, called by GpuFloatDeompress.cu
template <typename InProvider, typename OutProvider>
FloatDecompressStatus floatDecompressDevice(
    // used for allocating all GPU memory
    StackDeviceMemory& res,

    // Config for float decompression. See GpuFloatCodec.h
    const FloatDecompressConfig& config,

    // Number of input batches
    uint32_t numInBatch,

    // BatchProvider containing compressed data for each batch
    InProvider& inProvider,

    // OutProvider that determines where we will write the decompressed floats
    OutProvider& outProvider,

    // Maximum number of floats we can write to any given batch output
    uint32_t maxCapacity,

    // This array will be populated with whether decompression was successful
    uint8_t* outSuccess_dev,

    // This array will be populated with the number of decompressed floats in
    // each batch
    uint32_t* outSize_dev,

    // CUDA execution stream
    cudaStream_t stream) {
  // not allowed in float mode
  assert(!config.ansConfig.useChecksum);

  // We can perform decoding in a single pass if all input data is 16 byte
  // aligned. A fused kernel implementation is not possible for Float64,
  // as there are two rounds of ANS decompression.
  if (config.is16ByteAligned && config.floatType != FloatType::kFloat64) {
    //
    // Fused kernel: perform decompression in a single pass
    //
#define RUN_FUSED(FT)                                                     \
  do {                                                                    \
    auto inProviderANS = FloatANSProvider<FT, InProvider>(inProvider);    \
    auto outProviderANS =                                                 \
        FloatOutProvider<InProvider, OutProvider, FT, kDefaultBlockSize>( \
            inProvider, outProvider);                                     \
                                                                          \
    ansDecodeBatch(                                                       \
        res,                                                              \
        config.ansConfig,                                                 \
        numInBatch,                                                       \
        inProviderANS,                                                    \
        outProviderANS,                                                   \
        outSuccess_dev,                                                   \
        outSize_dev,                                                      \
        stream);                                                          \
  } while (false)

    switch (config.floatType) {
      case FloatType::kFloat16:
        RUN_FUSED(FloatType::kFloat16);
        break;
      case FloatType::kBFloat16:
        RUN_FUSED(FloatType::kBFloat16);
        break;
      case FloatType::kFloat32:
        RUN_FUSED(FloatType::kFloat32);
        break;
      default:
        CHECK(false);
        break;
    }

#undef RUN_FUSED
  }

  else {
    //
    // Two pass kernel: decompress the ANS compressed data, then rejoin with
    // uncompressed data
    //

    // Temporary space for the decompressed exponents
    // We need to ensure 16 byte alignment for the decompressed data due to
    // vectorization
    uint32_t maxCapacityAligned = roundUp(maxCapacity, sizeof(uint4));

    auto exp_dev = res.alloc<uint8_t>(stream, numInBatch * maxCapacityAligned * MAX_NUM_COMP_OUTS);

    // This is the "offset" array passed into the FloatANSProviderOffset. This will
    // be all zero, except for the second round of ANS decompression for Float64,
    // where it will contain the size of the first ANS-compressed output.
    auto ansOutOffset_dev = res.alloc<uint32_t>(stream, numInBatch);
    CUDA_VERIFY(cudaMemsetAsync(
      ansOutOffset_dev.data(),
      0,
      sizeof(uint32_t) * numInBatch,
      stream));

uint32_t compSegment = 0;
#define RUN_DECODE(FT, nCompSegments)                                       \
compSegment = 0;                                                            \
  do {                                                                      \
      using InProviderANS = FloatANSProviderOffset<FT, InProvider>;         \
      auto inProviderANS = InProviderANS(inProvider,                        \
                                         ansOutOffset_dev.data());          \
                                                                            \
      using OutProviderANS = BatchProviderStride;                           \
      auto outProviderANS = OutProviderANS(                                 \
          exp_dev.data() + compSegment * maxCapacityAligned * numInBatch,   \
          maxCapacityAligned, maxCapacityAligned);                          \
      auto outProviderANSFirstSegment = OutProviderANS(exp_dev.data(),      \
                                   maxCapacityAligned, maxCapacityAligned); \
      ansDecodeBatch(                                                       \
          res,                                                              \
          config.ansConfig,                                                 \
          numInBatch,                                                       \
          inProviderANS,                                                    \
          outProviderANS,                                                   \
          outSuccess_dev,                                                   \
          outSize_dev,                                                      \
          stream);                                                          \
                                                                            \
      if(compSegment==0 && nCompSegments == 2){                             \
      getANSOutOffset<InProvider, FT><<<divUp(numInBatch, 128), 128, 0,     \
                                    stream>>> (inProvider,                  \
                                    ansOutOffset_dev.data(), numInBatch);   \
      }                                                                     \
    if (compSegment == nCompSegments-1){                                    \
      constexpr int kThreads = 256;                                         \
      auto& props = getCurrentDeviceProperties();                           \
      int maxBlocksPerSM = 0;                                               \
      CUDA_VERIFY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(            \
          &maxBlocksPerSM,                                                  \
          joinFloat<OutProviderANS, InProvider, OutProvider, FT, kThreads>, \
          kThreads,                                                         \
          0));                                                              \
      uint32_t maxGrid = maxBlocksPerSM * props.multiProcessorCount;        \
      uint32_t perBatchGrid = divUp(maxGrid, numInBatch);                   \
      if ((perBatchGrid * numInBatch > maxGrid) && perBatchGrid > 1) {      \
        perBatchGrid -= 1;                                                  \
      }                                                                     \
      auto grid = dim3(perBatchGrid, numInBatch);                           \
                                                                            \
      joinFloat<OutProviderANS, InProvider, OutProvider, FT, kThreads>      \
          <<<grid, kThreads, 0, stream>>>(                                  \
              outProviderANS,                                               \
              outProviderANSFirstSegment,                                   \
              inProvider,                                                   \
              outProvider,                                                  \
              outSuccess_dev,                                               \
              outSize_dev);                                                 \
    }                                                                       \
  } while(++compSegment < nCompSegments);                                   \

    switch (config.floatType) {
      case FloatType::kFloat16:
        RUN_DECODE(FloatType::kFloat16, 1);
        break;
      case FloatType::kBFloat16:
        RUN_DECODE(FloatType::kBFloat16, 1);
        break;
      case FloatType::kFloat32:
        RUN_DECODE(FloatType::kFloat32, 1);
        break;
      case FloatType::kFloat64:
        RUN_DECODE(FloatType::kFloat64, 2);
        break;
      default:
        CHECK(false);
        break;
    }

#undef RUN_DECODE
  }

  FloatDecompressStatus status;

  // Perform optional checksum, if desired
  if (config.useChecksum) {
    auto checksum_dev = res.alloc<uint32_t>(stream, numInBatch);
    auto sizes_dev = res.alloc<uint32_t>(stream, numInBatch);
    auto archiveChecksum_dev = res.alloc<uint32_t>(stream, numInBatch);

    // Checksum the output data
    checksumBatch(numInBatch, outProvider, checksum_dev.data(), stream);

    // Get prior checksum from the float headers
    floatGetCompressedInfo(
        inProvider,
        numInBatch,
        sizes_dev.data(),
        nullptr,
        archiveChecksum_dev.data(),
        stream);

    // Compare against previously seen checksums on the host
    auto sizes = sizes_dev.copyToHost(stream);
    auto newChecksums = checksum_dev.copyToHost(stream);
    auto oldChecksums = archiveChecksum_dev.copyToHost(stream);

    std::stringstream errStr;

    for (int i = 0; i < numInBatch; ++i) {
      if (oldChecksums[i] != newChecksums[i]) {
        status.error = FloatDecompressError::ChecksumMismatch;

        errStr << "Checksum mismatch in batch member " << i
               << ": expected checksum " << std::hex << oldChecksums[i]
               << " got " << newChecksums[i] << "\n";
        status.errorInfo.push_back(std::make_pair(i, errStr.str()));
      }
    }
  }

  CUDA_TEST_ERROR();

  return status;
}

} // namespace dietgpu