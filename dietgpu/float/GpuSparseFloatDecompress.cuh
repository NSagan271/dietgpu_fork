#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/ans/GpuANSDecode.cuh"
#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuFloatInfo.cuh"
#include "dietgpu/float/GpuFloatUtils.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"
#include "dietgpu/float/GpuFloatDecompress.cuh"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

#include <glog/logging.h>
#include <cmath>
#include <sstream>
#include <vector>
#include <memory>


namespace dietgpu {

template <typename InProvider, 
    typename OutProvider, 
    typename HeaderProvider,
    FloatType FT>
__global__ void populate_dense(
    InProvider inProvider,
    OutProvider outProvider,
    HeaderProvider headerProvider,
    uint8_t* bitmaps,
    uint32_t* sparseIdx,
    uint32_t* outSize,
    uint32_t rowStride
) {
    using FTI = FloatTypeInfo<FT>;
    using WordT = typename FloatTypeInfo<FT>::WordT;
    int batch = blockIdx.y;

    auto curIn = (const WordT*) inProvider.getBatchStart(batch);
    auto curOut = (WordT*) outProvider.getBatchStart(batch);
    auto curHeader = (const GpuSparseFloatHeader*) headerProvider.getBatchStart(batch);
    uint32_t curSize = curHeader->size;

    auto curBitmap = bitmaps + batch * rowStride;

    uint32_t* curSparseIdx = sparseIdx + batch * rowStride;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        outSize[batch] = curSize;
    }

    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < curSize-1;
            i += gridDim.x * blockDim.x) {
        if (curBitmap[i] == 1) {
            curOut[i] = curIn[curSparseIdx[i]];
        } else {
            curOut[i] = 0;
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (curBitmap[curSize-1] == 1) {
            curOut[curSize-1] = curIn[curSparseIdx[curSize-2] + 1];
        } else{
            curOut[curSize-1] = 0;
        }
    }
}

template<typename InProvider>
__global__ void bitmap_bits_to_bytes(
    uint8_t* bitmap_bytes,
    InProvider inProvider,
    uint32_t rowStride
) {
    int batch = blockIdx.y;

    auto curOut = bitmap_bytes + batch * rowStride;
    auto curHeader = (const GpuSparseFloatHeader*) inProvider.getBatchStart(batch);
    uint32_t curSize = curHeader->size;

    auto curIn = (const uint8_t*) (curHeader + 1);

    auto curOutV = (uint8x8*) curOut;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (curSize + 7) / 8; 
            i+=gridDim.x * blockDim.x) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            curOutV[i].x[j] = (curIn[i] & (1 << (7 - j))) >> (7 - j);
        }
    }
}

template<typename InProvider, FloatType FT>
struct  SparseFloatInProvider{
    using FTI = FloatTypeInfo<FT>;
    
    __host__ SparseFloatInProvider(InProvider& inProvider)
        : inProvider_(inProvider) {}

    __device__ void* getBatchStart(uint32_t batch) {
        uint8_t* p = (uint8_t*)inProvider_.getBatchStart(batch);
        // This is the first place that touches the header
        GpuSparseFloatHeader h = *((GpuSparseFloatHeader*)p);

        // Increment the pointer to past the bitmap
        return p + sizeof(GpuSparseFloatHeader) + roundUp((h.size + 7) / 8, 16);
    }

    __device__ const void* getBatchStart(uint32_t batch) const {
        uint8_t* p = (uint8_t*)inProvider_.getBatchStart(batch);

        // This is the first place that touches the header
        GpuSparseFloatHeader h = *((GpuSparseFloatHeader*)p);

        // Increment the pointer to past the bitmap
        return p + sizeof(GpuSparseFloatHeader) + roundUp((h.size + 7) / 8, 16);
    }
    InProvider inProvider_;
};

template <typename InProvider, typename OutProvider>
FloatDecompressStatus floatDecompressSparseDevice(
        StackDeviceMemory& res,
        const FloatDecompressConfig& config,
        uint32_t numInBatch,
        InProvider& inProvider,
        OutProvider& outProvider,
        uint32_t maxCapacity,
        uint8_t* outSuccess_dev,
        uint32_t* outSize_dev,
        cudaStream_t stream) {
    // not allowed in float mode
    assert(!config.ansConfig.useChecksum);

    uint32_t maxCapacityAligned = roundUp(maxCapacity, sizeof(uint4));
    auto sparseDecompSizes = res.alloc<uint32_t>(stream, numInBatch);

    auto bitmaps = res.alloc<uint8_t>(stream, numInBatch*maxCapacityAligned);
    CUDA_VERIFY(cudaMemsetAsync(
      bitmaps.data(),
      0,
      sizeof(uint8_t) * maxCapacityAligned * numInBatch,
      stream));
    
    constexpr int kThreads = 256; 

    FloatDecompressStatus status;

    #define RUN_DECOMPRESS(FT)                                                \
    do {                                                                      \
        auto& props = getCurrentDeviceProperties();                           \
        int maxBlocksPerSM = 0;                                               \
        CUDA_VERIFY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(            \
            &maxBlocksPerSM,                                                  \
            bitmap_bits_to_bytes<InProvider>,                                 \
            kThreads,                                                         \
            0));                                                              \
        uint32_t maxGrid = maxBlocksPerSM * props.multiProcessorCount;        \
        uint32_t perBatchGrid = divUp(maxGrid, numInBatch);                   \
        if ((perBatchGrid * numInBatch > maxGrid) && perBatchGrid > 1) {      \
            perBatchGrid -= 1;                                                \
        }                                                                     \
        auto grid = dim3(perBatchGrid, numInBatch);                           \
                                                                              \
        auto sparseData = res.alloc<typename FloatTypeInfo<FT>::WordT>(       \
            stream, numInBatch * maxCapacityAligned);                         \
        auto sparseFloatInProvider = SparseFloatInProvider                    \
            <InProvider, FT>(inProvider);                                     \
        BatchProviderStride sparseFloatOutProvider = BatchProviderStride(     \
            sparseData.data(),                                                \
            maxCapacityAligned*sizeof(typename FloatTypeInfo<FT>::WordT),     \
            maxCapacity                                                       \
        );                                                                    \
                                                                              \
        status = floatDecompressDevice(                                       \
            res, config, numInBatch, sparseFloatInProvider,                   \
            sparseFloatOutProvider, maxCapacity, outSuccess_dev,              \
            sparseDecompSizes.data(), stream                                  \
        );                                                                    \
                                                                              \
        bitmap_bits_to_bytes<InProvider><<<grid, kThreads, 0, stream>>>(      \
            bitmaps.data(), inProvider, maxCapacityAligned                    \
        );                                                                    \
                                                                              \
        auto sparseIdx = res.alloc<uint32_t>(stream,                          \
            numInBatch * maxCapacityAligned);                                 \
                                                                              \
        cudaDeviceSynchronize();                                              \
        for (int batch = 0; batch < numInBatch; ++batch){                     \
            thrust::exclusive_scan(                                           \
                thrust::device,                                               \
                bitmaps.data() + batch * maxCapacityAligned,                  \
                bitmaps.data() + batch * maxCapacityAligned +                 \
                                            maxCapacityAligned - 1,           \
                (uint32_t *) sparseIdx.data() + batch*maxCapacityAligned, 0); \
        }                                                                     \
        cudaDeviceSynchronize();                                              \
                                                                              \
        maxBlocksPerSM = 0;                                                   \
        CUDA_VERIFY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(            \
            &maxBlocksPerSM,                                                  \
            populate_dense<BatchProviderStride, OutProvider, InProvider, FT>, \
            kThreads,                                                         \
            0));                                                              \
        maxGrid = maxBlocksPerSM * props.multiProcessorCount;                 \
        perBatchGrid = divUp(maxGrid, numInBatch);                            \
        if ((perBatchGrid * numInBatch > maxGrid) && perBatchGrid > 1) {      \
            perBatchGrid -= 1;                                                \
        }                                                                     \
        grid = dim3(perBatchGrid, numInBatch);                                \
        populate_dense<BatchProviderStride, OutProvider, InProvider, FT>      \
            <<<grid, kThreads, 0, stream>>>(                                  \
            sparseFloatOutProvider, outProvider, inProvider,                  \
            bitmaps.data(), sparseIdx.data(), outSize_dev, maxCapacityAligned \
        );                                                                    \
    } while(false);

    switch (config.floatType) {
        case FloatType::kFloat16:
            RUN_DECOMPRESS(FloatType::kFloat16);
            break;
        case FloatType::kBFloat16:
            RUN_DECOMPRESS(FloatType::kBFloat16);
            break;
        case FloatType::kFloat32:
            RUN_DECOMPRESS(FloatType::kFloat32);
            break;
        case FloatType::kFloat64:
            RUN_DECOMPRESS(FloatType::kFloat64);
            break;
        default:
            assert(false);
        break;
    }

    return status;
}
}