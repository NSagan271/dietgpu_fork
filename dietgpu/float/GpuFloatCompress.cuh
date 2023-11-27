/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

#define MAX_NUM_COMP_OUTS 2

namespace dietgpu {

template <FloatType FT>
struct UpdateCompAndHist {
  static __device__ void update(
      const typename FloatTypeInfo<FT>::WordT inWord,
      typename FloatTypeInfo<FT>::CompT* compOuts,
      typename FloatTypeInfo<FT>::NonCompSplit1T& nonComp1Out,
      typename FloatTypeInfo<FT>::NonCompSplit2T& nonComp2Out, // Only used for F32 and F64
      uint32_t** warpHistograms) {
    using FTI = FloatTypeInfo<FT>;
    using CompT = typename FTI::CompT;
    using NonCompT = typename FTI::NonCompT;
    using WordT = typename FTI::WordT;

    NonCompT nonComp;
    FTI::split(inWord, compOuts, nonComp);

    nonComp1Out = nonComp;

    atomicAdd(&warpHistograms[0][compOuts[0]], 1);
  }
};

template<>
struct UpdateCompAndHist<FloatType::kFloat32> {
  static __device__ void update(
      const typename FloatTypeInfo<FloatType::kFloat32>::WordT inWord,
      typename FloatTypeInfo<FloatType::kFloat32>::CompT* compOuts,
      typename FloatTypeInfo<FloatType::kFloat32>::NonCompSplit1T& nonComp1Out,
      typename FloatTypeInfo<FloatType::kFloat32>::NonCompSplit2T& nonComp2Out, // Only used for F32 and F64
      uint32_t** warpHistograms) {
    using FTI = FloatTypeInfo<FloatType::kFloat32>;
    using CompT = typename FTI::CompT;
    using NonCompT = typename FTI::NonCompT;
    using WordT = typename FTI::WordT;

    NonCompT nonComp;
    FTI::split(inWord, compOuts, nonComp);

    nonComp1Out = nonComp & 0xffffU;
    nonComp2Out = nonComp >> 16;

    atomicAdd(&warpHistograms[0][compOuts[0]], 1);
  }
};

template<>
struct UpdateCompAndHist<FloatType::kFloat64> {
  static __device__ void update(
      const typename FloatTypeInfo<FloatType::kFloat64>::WordT inWord,
      typename FloatTypeInfo<FloatType::kFloat64>::CompT* compOuts,
      typename FloatTypeInfo<FloatType::kFloat64>::NonCompSplit1T& nonComp1Out,
      typename FloatTypeInfo<FloatType::kFloat64>::NonCompSplit2T& nonComp2Out, // Only used for F32 and F64
      uint32_t** warpHistograms) {
    using FTI = FloatTypeInfo<FloatType::kFloat64>;
    using CompT = typename FTI::CompT;
    using NonCompT = typename FTI::NonCompT;
    using WordT = typename FTI::WordT;

    NonCompT nonComp;
    FTI::split(inWord, compOuts, nonComp);

    nonComp1Out = nonComp & 0xffffffffU;
    nonComp2Out = nonComp >> 32;

    atomicAdd(&warpHistograms[0][compOuts[0]], 1);
    atomicAdd(&warpHistograms[1][compOuts[1]], 1);
  }
};

  template <FloatType FT, int Threads>
  struct SplitFloatNonAligned {
    static __device__ void split(
        const typename FloatTypeInfo<FT>::WordT* in,
        uint32_t size,
        typename FloatTypeInfo<FT>::CompT* compOuts,
        typename FloatTypeInfo<FT>::NonCompT* nonCompOut,
        uint32_t* warpHistograms,
        uint32_t compDatasetStride,
        uint32_t histDatasetStride) {
      using FTI = FloatTypeInfo<FT>;
      using CompT = typename FTI::CompT;
      using NonCompT = typename FTI::NonCompT;
      using NonCompSplit1T = typename FTI::NonCompSplit1T;
      using NonCompSplit2T = typename FTI::NonCompSplit2T;

      uint32_t* warpHistogram2DArr[MAX_NUM_COMP_OUTS] = {warpHistograms, warpHistograms + histDatasetStride};

      NonCompSplit1T* nonCompOut1 = (NonCompSplit1T*) nonCompOut;
      NonCompSplit2T* nonCompOut2 = (NonCompSplit2T*) nonCompOut;
      if (FTI::getIfNonCompSplit())
        nonCompOut2 = (NonCompSplit2T*) (nonCompOut1 + roundUp(size, 16 / sizeof(NonCompSplit1T)));

      for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
          i += gridDim.x * blockDim.x) {
        CompT comps[MAX_NUM_COMP_OUTS];
        NonCompSplit1T nonComp1;
        NonCompSplit2T nonComp2;

        UpdateCompAndHist<FT>::update(in[i], comps, nonComp1, nonComp2, warpHistogram2DArr);
        nonCompOut1[i] = nonComp1;
        if (FTI::getIfNonCompSplit())
          nonCompOut2[i] = nonComp2;
        
        for (int k = 0; k < FTI::getNumCompSegments(); k++) {
          compOuts[i + k*compDatasetStride] = comps[k];
        }
      }
    }
  };

template <FloatType FT, int Threads>
struct SplitFloatAligned16 {
  static __device__ void split(
      const typename FloatTypeInfo<FT>::WordT* __restrict__ in,
      uint32_t size,
      typename FloatTypeInfo<FT>::CompT* __restrict__ compOuts,
      typename FloatTypeInfo<FT>::NonCompT* __restrict__ nonCompOut,
      uint32_t* warpHistograms,
      uint32_t compDatasetStride,
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

    constexpr int kOuterUnroll = 2;
    constexpr int kInnerUnroll = sizeof(VecT) / sizeof(WordT);

    int numCompSegments = FTI::getNumCompSegments();

    uint32_t *warpHistogram2DArr[MAX_NUM_COMP_OUTS] = {warpHistograms, warpHistograms+histDatasetStride};

    NonCompSplit1T* nonCompOut1 = (NonCompSplit1T*)nonCompOut;
    NonCompSplit2T* nonCompOut2 = (NonCompSplit2T*)nonCompOut;
    if (FTI::getIfNonCompSplit()) {
      nonCompOut2 = (NonCompSplit2T*) (nonCompOut1 + roundUp(size, 16 / sizeof(NonCompSplit1T)));
    }

    const VecT* inV = (const VecT*)in;
    CompVecT* compOutsV = (CompVecT*)compOuts;
    NonCompVecSplit1T* nonCompOutV1 = (NonCompVecSplit1T*)nonCompOut1;
    NonCompVecSplit2T* nonCompOutV2 = (NonCompVecSplit2T*)nonCompOut2;

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

          UpdateCompAndHist<FT>::update(v[i].x[j], comps, nonComp1, nonComp2, warpHistogram2DArr);
          nonCompV1[i].x[j] = nonComp1;
          if (FTI::getIfNonCompSplit())
            nonCompV2[i].x[j] = nonComp2;

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

template <
    typename InProvider,
    typename NonCompProvider,
    FloatType FT,
    int Threads>
__global__ void splitFloat(
    InProvider inProvider,
    bool useChecksum,
    const uint32_t* __restrict__ checksum,
    void* __restrict__ compOuts,
    uint32_t compOutStride,
    uint32_t compDatasetStride, // for F64: where the second dataset to ANS compress starts
    uint32_t histDatasetStride,
    NonCompProvider nonCompProvider,
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

  // printf("here here here 1\n");
  checksum += batch;

  // +1 in order to force very common symbols that could overlap into different
  // banks between different warps
  __shared__ uint32_t histogram[kWarps][MAX_NUM_COMP_OUTS * roundUp(kNumSymbols + 1, 4)];
// printf("here here here 1.5\n");
#pragma unroll
  for (int i = 0; i < kWarps; ++i) {
    for (int k = 0; k < numHists; k++) {
        histogram[i][threadIdx.x + k*roundUp(kNumSymbols + 1, 4)] = 0;
    }
  }

  __syncthreads();
  // printf("here here here 2\n");

  uint32_t* warpHistograms = histogram[warpId];
  // printf("here here here 3\n");

  auto curIn = (const WordT*)inProvider.getBatchStart(batch);
  auto headerOut = (GpuFloatHeader*)nonCompProvider.getBatchStart(batch);
  // printf("here here here 4\n");

  CompT* curCompOuts = (CompT*) compOuts + compOutStride * batch;
  // printf("here here here 5\n");
  auto curSize = inProvider.getBatchSize(batch);
  // printf("here here here 6\n");

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

  // printf("here here here 7\n");

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
// uncompressed and compressed portions
template <FloatType FT, typename InProvider>
__global__ void
incOutputSizes(InProvider inProvider, uint32_t* outSize, uint32_t numInBatch) {
  uint32_t batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch < numInBatch) {
    outSize[batch] += sizeof(GpuFloatHeader) + sizeof(GpuFloatHeader2) +
        FloatTypeInfo<FT>::getUncompDataSize(inProvider.getBatchSize(batch));
  }
}

__global__ void
incOutputSizes2(uint32_t* outSize, uint32_t* increments, uint32_t numInBatch) {
  uint32_t batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch < numInBatch) {
    outSize[batch] += increments[batch];
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

template <typename InProvider, typename OutProvider>
void floatCompressDevice(
    StackDeviceMemory& res,
    const FloatCompressConfig& config,
    uint32_t numInBatch,
    InProvider& inProvider,
    uint32_t maxSize,
    OutProvider& outProvider,
    uint32_t* outSize_dev,
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
  // F64: [ ... first dataset (size numInBatch * compRowStride) ..., ... second dataset (same size) ... ]
  auto toComp_dev = res.alloc<uint8_t>(stream, roundUp(numInBatch * compRowStride, 16) * MAX_NUM_COMP_OUTS);

  auto tempOutSize_dev = res.alloc<uint32_t>(stream, numInBatch);
  auto ansOutOffset_dev = res.alloc<uint32_t>(stream, numInBatch);

  // We calculate a histogram of the symbols to be compressed as part of
  // extracting the compressible symbol from the float
  auto histograms_dev = res.alloc<uint32_t>(stream, roundUp(numInBatch * kNumSymbols, 4) * MAX_NUM_COMP_OUTS);

  // zero out buckets before proceeding, as we aggregate with atomic adds
  CUDA_VERIFY(cudaMemsetAsync(
      histograms_dev.data(),
      0,
      sizeof(uint32_t) * roundUp(numInBatch * kNumSymbols, 4) * MAX_NUM_COMP_OUTS,
      stream));

  CUDA_VERIFY(cudaMemsetAsync(
      ansOutOffset_dev.data(),
      0,
      sizeof(uint32_t) * numInBatch,
      stream));

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
            roundUp(numInBatch * compRowStride, 16),               \
            roundUp(numInBatch * kNumSymbols, 4),                  \
            outProvider,                                           \
            histograms_dev.data());                                \
  /*printf("splitFloat done\n");*/\
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

    // outSize as reported by ansEncode is just the ANS-encoded portion of the
    // data.
    // We need to increment the sizes by the uncompressed portion (header plus
    // uncompressed float data) with incOutputSizes
uint32_t compSegment = 0; 
#define RUN_ANS(FT, nCompSegments)                                          \
  compSegment = 0;                                                          \
  do {                                                                      \
  /*printf("run FloatANSInProvider\n");*/ \
    auto inProviderANS = FloatANSInProvider<InProvider>(                    \
        toComp_dev.data() + compSegment *                                   \
                    roundUp(numInBatch * compRowStride, 16),                \
        compRowStride, inProvider);                                         \
                                                                            \
   /*printf("run FloatANSOutProvider\n");*/ \
    auto outProviderANS = FloatANSOutProvider<FT, OutProvider, InProvider>( \
      outProvider, inProvider, ansOutOffset_dev.data());                    \
                                                                            \
    uint32_t* outSizes = (compSegment == 0) ? outSize_dev :                 \
                              tempOutSize_dev.data();                       \
                                                                            \
    /*printf("run ansEncodeBatchDevice\n");*/ \
    ansEncodeBatchDevice(                                                   \
        res,                                                                \
        config.ansConfig,                                                   \
        numInBatch,                                                         \
        inProviderANS,                                                      \
        histograms_dev.data() + compSegment *                               \
                    roundUp(numInBatch * kNumSymbols, 4),                   \
        maxSize,                                                            \
        outProviderANS,                                                     \
        outSizes,                                                           \
        stream);                                                            \
                                                                            \
    if (compSegment == 0) {                                                 \
        incOutputSizes<FT><<<divUp(numInBatch, 128), 128, 0,                \
            stream>>>(inProvider, outSize_dev, numInBatch);                 \
        setHeaderAndANSOutOffset<InProvider, OutProvider, FT>               \
                                <<<divUp(numInBatch, 128), 128, 0,          \
                                    stream>>> ( outProvider, outProviderANS,\
                                    ansOutOffset_dev.data(), numInBatch);   \
    }                                                                       \
    else                                                                    \
        incOutputSizes2<<<divUp(numInBatch, 128), 128, 0,                   \
            stream>>>(outSize_dev, tempOutSize_dev.data(), numInBatch);     \
                                                                            \
       /*printf("done ANS\n");*/ \
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
