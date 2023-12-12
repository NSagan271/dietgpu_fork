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

template<typename T>
__global__ void printarr(T *idxs, uint32_t count) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint32_t i = 0; i < count; ++i) {
            printf("%lu\t", idxs[i]);
            if (i % 10 == 9)
                printf("\n");
        }
        printf("\n");
    }
}

template <typename InProvider>
__global__ void get_bitmap_idxs(
    InProvider inProvider,
    uint8_t** bitmapStarts,
    uint8_t** bitmapEnds
) {
    int batch = blockIdx.y;

    auto header = (GpuSparseFloatHeader*) inProvider.getBatchStart(batch);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        bitmapStarts[batch] = (uint8_t *) (header + 1);
        bitmapEnds[batch] = ((uint8_t *) (header + 1)) + header->size - 1;
    }
}

template <typename InProvider, 
    typename OutProvider, 
    typename HeaderProvider,
    FloatType FT>
__global__ void populate_dense(
    InProvider inProvider,
    OutProvider outProvider,
    HeaderProvider headerProvider,
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

    auto curBitmap = (const uint8_t*) (curHeader + 1);

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
        return p + sizeof(GpuSparseFloatHeader) + roundUp(h.size, 16);
    }

    __device__ const void* getBatchStart(uint32_t batch) const {
        uint8_t* p = (uint8_t*)inProvider_.getBatchStart(batch);

        // This is the first place that touches the header
        GpuSparseFloatHeader h = *((GpuSparseFloatHeader*)p);

        // Increment the pointer to past the bitmap
        return p + sizeof(GpuSparseFloatHeader) + roundUp(h.size, 16);
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

    auto bitmapStarts = res.alloc<uintptr_t>(stream, 2*numInBatch);
    auto bitmapEnds = bitmapStarts.data() + numInBatch;
    

    constexpr int kThreads = 256; 

    FloatDecompressStatus status;

    #define RUN_DECOMPRESS(FT) \
    do { \
        auto& props = getCurrentDeviceProperties();                           \
        int maxBlocksPerSM = 0;                                               \
        CUDA_VERIFY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(            \
            &maxBlocksPerSM,                                                  \
            get_bitmap_idxs<InProvider>, \
            kThreads,                                                         \
            0));                                                              \
        uint32_t maxGrid = maxBlocksPerSM * props.multiProcessorCount;        \
        uint32_t perBatchGrid = divUp(maxGrid, numInBatch);                   \
        if ((perBatchGrid * numInBatch > maxGrid) && perBatchGrid > 1) {      \
            perBatchGrid -= 1;                                                 \
        }                                                                     \
        auto grid = dim3(perBatchGrid, numInBatch);    \
        \
        auto sparseData = res.alloc<typename FloatTypeInfo<FT>::WordT>(\
            stream, numInBatch * maxCapacityAligned); \
        auto sparseFloatInProvider = SparseFloatInProvider<InProvider, FT>(inProvider); \
        BatchProviderStride sparseFloatOutProvider = BatchProviderStride( \
            sparseData.data(), maxCapacityAligned, maxCapacity \
        ); \
        \
        status = floatDecompressDevice( \
            res, config, numInBatch, sparseFloatInProvider, sparseFloatOutProvider, \
            maxCapacity, outSuccess_dev, sparseDecompSizes.data(), stream \
        ); \
        \
        cudaDeviceSynchronize(); \
        get_bitmap_idxs<InProvider><<<grid, kThreads, 0, stream>>>( \
            inProvider, (uint8_t**)bitmapStarts.data(), (uint8_t**)bitmapEnds \
        ); \
        /* [bitmap start for batch 1, start for batch 2, ..., end for batch 1, ...]*/ \
        std::vector<uintptr_t> bitmapStarts_host(numInBatch * 2);  \
        uintptr_t* bitmapEnds_host = bitmapStarts_host.data() + numInBatch; \
        \
        CUDA_VERIFY(cudaMemcpyAsync( \
            bitmapStarts_host.data(), \
            bitmapStarts.data(), \
            sizeof(uintptr_t) * numInBatch * 2, \
            cudaMemcpyDeviceToHost, \
            stream)); \
        cudaDeviceSynchronize(); \
        uint32_t maxSize = 0; \
        for (int batch = 0; batch < numInBatch; ++batch){ \
            maxSize = std::max(maxSize, (uint32_t) (bitmapEnds_host[batch] - bitmapStarts_host[batch]) + 1); \
        } \
        \
        uint32_t rowStride = roundUp(maxSize, sizeof(uint4)); \
        \
        auto sparseIdx = res.alloc<uint32_t>(stream, numInBatch * rowStride); \
        \
        for (int batch = 0; batch < numInBatch; ++batch){ \
            thrust::exclusive_scan( \
                thrust::device, \
                (uint8_t *)bitmapStarts_host[batch], \
                (uint8_t *)bitmapEnds_host[batch], \
                (uint32_t *) sparseIdx.data() + batch*rowStride, 0); \
        } \
        cudaDeviceSynchronize(); \
        \
        maxBlocksPerSM = 0;                                               \
        CUDA_VERIFY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(            \
            &maxBlocksPerSM,                                                  \
            populate_dense<BatchProviderStride, OutProvider, InProvider, FT>, \
            kThreads,                                                         \
            0));                                                              \
        maxGrid = maxBlocksPerSM * props.multiProcessorCount;        \
        perBatchGrid = divUp(maxGrid, numInBatch);                   \
        if ((perBatchGrid * numInBatch > maxGrid) && perBatchGrid > 1) {      \
            perBatchGrid -= 1;                                                 \
        }                                                                     \
        grid = dim3(perBatchGrid, numInBatch);    \
        populate_dense<BatchProviderStride, OutProvider, InProvider, FT>\
            <<<grid, kThreads, 0, stream>>>( \
            sparseFloatOutProvider, outProvider, inProvider, \
            sparseIdx.data(), outSize_dev, rowStride \
        ); \
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