
#include "dietgpu/ans/BatchProvider.cuh"
#include "dietgpu/ans/GpuANSEncode.cuh"
#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/ans/GpuChecksum.cuh"
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuFloatCompress.cuh"
#include "dietgpu/float/GpuFloatUtils.cuh"
#include "dietgpu/utils/DeviceDefs.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include <glog/logging.h>
#include <cmath>
#include <memory>
#include <vector>

namespace dietgpu {

template<typename T>
__global__ void printarr2(T *idxs, int count) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < count; ++i) {
            printf("%lu\t", idxs[i]);
            if (i % 10 == 9)
                printf("\n");
        }
        printf("\n");
    }
}

template <
    typename InProvider,
    FloatType FT>
__global__ void generate_bitmap(
    InProvider inProvider,
    uint8_t* bitmaps,
    uint32_t rowStride
) {
    using FTI = FloatTypeInfo<FT>;
    using WordT = typename FloatTypeInfo<FT>::WordT;

    int batch = blockIdx.y;

    auto curIn = (const WordT*) inProvider.getBatchStart(batch);
    auto curSize = inProvider.getBatchSize(batch);

    auto curOut = bitmaps + batch * rowStride;

    // TODO: vectorize!
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < curSize;
            i += gridDim.x * blockDim.x) {
        curOut[i] = (curIn[i] == 0) ? 0 : 1;
    }
}

template<typename OutProvider, typename SizeProvider>
__global__ void bitmap_bytes_to_bits(
    uint8_t* bitmap_bytes,
    OutProvider outProvider,
    SizeProvider sizeProvider,
    uint32_t rowStride
) {
    int batch = blockIdx.y;

    auto curIn = bitmap_bytes + batch * rowStride;
    auto curSize = sizeProvider.getBatchSize(batch);

    auto headerOut = (GpuSparseFloatHeader*) outProvider.getBatchStart(batch);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        GpuSparseFloatHeader h;
        h.setSize(curSize);
        *headerOut = h;
    }
    auto curOut = (uint8_t *) (headerOut + 1);

    auto curInV = (uint8x8*) curIn;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (curSize + 7) / 8; 
            i+=gridDim.x * blockDim.x) {
        uint8x8 v = curInV[i];
        curOut[i] = v.x[7] | (v.x[6] << 1) | (v.x[5] << 2) |
                    (v.x[4] << 3) | (v.x[3] << 4) | (v.x[2] << 5) |
                    (v.x[1] << 6) | (v.x[0] << 7);

    }
}

template <
    typename InProvider,
    FloatType FT>
__global__ void fill_comp_input(
    InProvider inProvider,
    typename FloatTypeInfo<FT>::WordT* sparseData,
    uint8_t* bitmaps,
    uint32_t* sparseIdx,
    uint32_t* sparseBatchSizes, // size numInBatch
    void** batchPointers, // for the BatchProviderPointer
    uint32_t rowStride // the start of the second batch in sparseIdx
                        // and sparseData
) {
    using FTI = FloatTypeInfo<FT>;
    using WordT = typename FloatTypeInfo<FT>::WordT;

    int batch = blockIdx.y;

    auto curIn = (const WordT*) inProvider.getBatchStart(batch);
    auto curBitmap = bitmaps + batch * rowStride;
    auto curOut = sparseData + batch * rowStride;
    auto curSize = inProvider.getBatchSize(batch);
    uint32_t* curSparseIdx = sparseIdx + batch * rowStride;

    // TODO: vectorize?
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < curSize-1;
            i += gridDim.x * blockDim.x) {
        if (curBitmap[i] == 1) {
            curOut[curSparseIdx[i]] = curIn[i];
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        batchPointers[batch] = (void*) curOut;
        if (curBitmap[curSize-1] == 1) {
            curOut[curSparseIdx[curSize-2] + 1] = curIn[curSize-1];
        }
        sparseBatchSizes[batch] = curSparseIdx[curSize-2] + curBitmap[curSize-1] + 1;
    }
}

template <FloatType FT, typename OutProvider, typename SizeProvider>
struct SparseFloatOutProvider {
    using Writer = BatchWriter;
    using FTI = FloatTypeInfo<FT>;

    __host__ SparseFloatOutProvider(
        OutProvider& outProvider,
        SizeProvider& sizeProvider)
        : outProvider_(outProvider), sizeProvider_(sizeProvider) {
        }

    __device__ void* getBatchStart(uint32_t batch) {
        uint8_t* p = (uint8_t*)outProvider_.getBatchStart(batch);

        // Increment the pointer to past the bitmap data
        return p + sizeof(GpuSparseFloatHeader) +
            roundUp((sizeProvider_.getBatchSize(batch) + 7) / 8, 16);
    }

    __device__ const void* getBatchStart(uint32_t batch) const {
        const uint8_t* p = (const uint8_t*)outProvider_.getBatchStart(batch);

        // Increment the pointer to past the bitmap data
        return p + sizeof(GpuSparseFloatHeader) +
            roundUp((sizeProvider_.getBatchSize(batch) + 7) / 8, 16);
    }

    __device__ BatchWriter getWriter(uint32_t batch) {
        return BatchWriter(getBatchStart(batch));
    }

    OutProvider outProvider_;
    SizeProvider sizeProvider_;
};

template<typename SizeProvider>
__global__ void addBitmapToOutSizes(SizeProvider sizeProvider, uint32_t* outSize_dev) {
    int batch = blockIdx.y;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        outSize_dev[batch] += roundUp((sizeProvider.getBatchSize(batch) + 7) / 8, 16) +
                             sizeof(GpuSparseFloatHeader);
    }
}

template <typename InProvider, typename OutProvider>
void floatCompressSparseDevice(
    StackDeviceMemory& res,
    const FloatCompressConfig& config,
    uint32_t numInBatch,
    InProvider& inProvider,
    uint32_t maxSize,
    OutProvider& outProvider,
    uint32_t* outSize_dev,
    void** out_host,
    const uint32_t* inSize_host,
    cudaStream_t stream) {

    // not allowed in float mode
    assert(!config.ansConfig.useChecksum);

    // CUDA kernel parameters
    constexpr int kBlock = 256; 

    uint32_t rowStride = roundUp(maxSize, sizeof(uint4));
    auto bitmaps = res.alloc<uint8_t>(stream, numInBatch * rowStride);
    CUDA_VERIFY(cudaMemsetAsync(
      bitmaps.data(),
      0,
      sizeof(uint8_t) * rowStride * numInBatch,
      stream));

    #define RUN_BITMAP(FT) \
    do { \
        auto& props = getCurrentDeviceProperties();                    \
        int maxBlocksPerSM = 0;                                        \
        CUDA_VERIFY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(     \
            &maxBlocksPerSM,                                           \
            generate_bitmap<InProvider, FT>,   \
            kBlock,                                                    \
            0));                                                       \
        uint32_t maxGrid = maxBlocksPerSM * props.multiProcessorCount; \
        uint32_t perBatchGrid = 4 * divUp(maxGrid, numInBatch);        \
        auto grid = dim3(perBatchGrid, numInBatch);                     \
        generate_bitmap<InProvider, FT><<<grid, kBlock, 0, stream>>>( \
            inProvider, bitmaps.data(), rowStride \
        ); \
        bitmap_bytes_to_bits<OutProvider, InProvider><<<grid, kBlock, 0, stream>>>( \
            bitmaps.data(), outProvider, inProvider, rowStride \
        ); \
    } while(false);

    switch (config.floatType) {
        case FloatType::kFloat16:
            RUN_BITMAP(FloatType::kFloat16);
            break;
        case FloatType::kBFloat16:
            RUN_BITMAP(FloatType::kBFloat16);
            break;
        case FloatType::kFloat32:
            RUN_BITMAP(FloatType::kFloat32);
            break;
        case FloatType::kFloat64:
            RUN_BITMAP(FloatType::kFloat64);
            break;
        default:
            assert(false);
        break;
    }

    cudaDeviceSynchronize();

    auto sparseIdx = res.alloc<uint32_t>(stream, numInBatch * rowStride);
    auto sparseBatchSizes = res.alloc<uint32_t>(stream, numInBatch);
    auto batchPointers = res.alloc<uintptr_t>(stream, numInBatch);
    
    for (int batch = 0; batch < numInBatch; ++batch){
        thrust::exclusive_scan(
            thrust::device,
            bitmaps.data() + batch * rowStride,
            bitmaps.data() + batch * rowStride + rowStride - 1,
            sparseIdx.data() + batch*rowStride, 0);
    }

    cudaDeviceSynchronize();

    #define RUN_COMPRESS(FT) \
    do { \
        auto& props = getCurrentDeviceProperties();                    \
        int maxBlocksPerSM = 0;                                        \
        CUDA_VERIFY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(     \
            &maxBlocksPerSM,                                           \
            fill_comp_input<InProvider, FT>,          \
            kBlock,                                                    \
            0));                                                       \
        uint32_t maxGrid = maxBlocksPerSM * props.multiProcessorCount; \
        uint32_t perBatchGrid = 4 * divUp(maxGrid, numInBatch);        \
        auto grid = dim3(perBatchGrid, numInBatch);                     \
        auto sparseData = res.alloc<typename FloatTypeInfo<FT>::WordT>(\
            stream, numInBatch * rowStride); \
        fill_comp_input<InProvider, FT><<<grid, kBlock, 0, stream>>> ( \
            inProvider, sparseData.data(), bitmaps.data(), sparseIdx.data(), \
            sparseBatchSizes.data(), (void**) batchPointers.data(), rowStride \
        ); \
        \
        auto sparseFloatOutProvider = SparseFloatOutProvider<FT, OutProvider, InProvider>(\
            outProvider, inProvider \
        ); \
        auto sparseFloatInProvider = BatchProviderPointer((void**) batchPointers.data(), \
            sparseBatchSizes.data()); \
        floatCompressDevice( \
            res, config, numInBatch, sparseFloatInProvider, maxSize, \
            sparseFloatOutProvider, outSize_dev, stream \
        ); \
        \
        addBitmapToOutSizes<<<grid, kBlock, 0, stream>>>(inProvider, outSize_dev);\
    } while(false);

    switch (config.floatType) {
        case FloatType::kFloat16:
            RUN_COMPRESS(FloatType::kFloat16);
            break;
        case FloatType::kBFloat16:
            RUN_COMPRESS(FloatType::kBFloat16);
            break;
        case FloatType::kFloat32:
            RUN_COMPRESS(FloatType::kFloat32);
            break;
        case FloatType::kFloat64:
            RUN_COMPRESS(FloatType::kFloat64);
            break;
        default:
            assert(false);
        break;
    }
    
    CUDA_TEST_ERROR();
}
}