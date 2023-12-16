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

template<typename InProvider>
__global__ void bitmap_bits_to_bytes(
    // output bitmaps, as a byte array. Its length must be rounded up to the
    // nearest multiple of 8.
    uint8_t* bitmap_bytes,

    // BatchProvider input to floatDecompressSparseDevice. This will provide
    // the sparse float header, which lets us calculate the length of the
    // bitmap. The bitmap (packed into bits) is located directly after the header.
    InProvider inProvider,

    // rowStride is the distance, in array elements, between adjacent batches
    // in the output "bitmaps" array.
    uint32_t rowStride
) {
    int batch = blockIdx.y;

    auto curOut = bitmap_bytes + batch * rowStride;
    auto curHeader = (const GpuSparseFloatHeader*) inProvider.getBatchStart(batch);

    // This, rounded up to 8, is the number of bits in the bitmap.
    uint32_t curSize = curHeader->size;

    // Input data, where each bit will be a different element of
    // bitmap_bytes.
    auto curIn = (const uint8_t*) (curHeader + 1);

    // For each byte of curIn, we will write an 8-byte section of bitmap_bytes.
    auto curOutV = (uint8x8*) curOut;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (curSize + 7) / 8; 
            i+=gridDim.x * blockDim.x) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            // Each bit of curIn becomes an element of the output byte array
            curOutV[i].x[j] = (curIn[i] & (1 << (7 - j))) >> (7 - j);
        }
    }
}

/* Given the decompression output of floatDecompressDevice, which gives us
 * all of the nonzero floats, the bitmaps produced by bitmap_bits_to_bytes,
 * and the exclusive scan of each batch's bitmap, populate the final output
 * array of floatDecompressSparseDevice.
 */
template <typename InProvider, 
    typename OutProvider, 
    typename HeaderProvider,
    FloatType FT>
__global__ void fill_in_nonzeros(
    // BatchProvider with output data from floatDecompressDevice (called
    // inProvider because it is the input to this subroutine)
    InProvider inProvider,

    // BatchProvider that tells us where to write the final output data
    OutProvider outProvider,

    // BatchProvider originally input to floatDecompressSparseDevice. This
    // gives us access to the sparse float header, which tells us how many
    // floats (zero and nonzero combined) were in the originally-compressed
    // dataset.
    HeaderProvider headerProvider,

    // Bitmap, as a byte array, denoting whether each index of the originally-
    // compressed float data was zero or nonzero. This array should be populated
    // by bitmap_bits_to_bytes.
    uint8_t* bitmaps,

    // result of performing an exclusive scan on each batch's bitmap.
    // Where bitmask is 1, nonzerosIdx gives the corresponding index of the
    // nonzeros dataset, aka the data from inProvider
    uint32_t* nonzerosIdx,

    // This kernel will set each element of outSize to the corresponding number
    // of floats in the originally-compressed dataset for this batch.
    uint32_t* outSize,

    // rowStride is the distance, in array elements, between adjacent batches
    // in the "bitmaps" and "nonzerosIdx" arrays.
    uint32_t rowStride
) {
    using FTI = FloatTypeInfo<FT>;
    using WordT = typename FloatTypeInfo<FT>::WordT;
    int batch = blockIdx.y;

    // Get input, output, header, bitmap, and nonzerosIdx for this batch
    auto curIn = (const WordT*) inProvider.getBatchStart(batch);
    auto curOut = (WordT*) outProvider.getBatchStart(batch);
    auto curHeader = (const GpuSparseFloatHeader*) headerProvider.getBatchStart(batch);
    uint32_t curSize = curHeader->size;

    auto curBitmap = bitmaps + batch * rowStride;
    uint32_t* curNonzerosIdx = nonzerosIdx + batch * rowStride;

    // The first thread sets the outSize array for the appropriate batch
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        outSize[batch] = curSize;
    }

    // For each element of the output, set it to the correct index of the
    // "nonzeros dataset" (curIn) if the bitmap value is 1 and 0 otherwise.
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < curSize-1;
            i += gridDim.x * blockDim.x) {
        if (curBitmap[i] == 1) {
            curOut[i] = curIn[curNonzerosIdx[i]];
        } else {
            curOut[i] = 0;
        }
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Special handling is required for the last input float, as the
        // exclusive scan does not properly populate the last element of
        // curNonzerosIdx
        if (curBitmap[curSize-1] == 1) {
            curOut[curSize-1] = curIn[curNonzerosIdx[curSize-2] + 1];
        } else{
            curOut[curSize-1] = 0;
        }
    }
}

/* This is a version of a BatchProvider, and it tells the float decompressor where
 * to find the compressed input. It takes the InProvider passed into
 * floatDecompressSparseDevice, and it finds the first 16-byte aligned address
 * after the sparse float header and the bitmap.
 */
template<typename InProvider, FloatType FT>
struct  SparseFloatInProvider{
    using FTI = FloatTypeInfo<FT>;
    
    __host__ SparseFloatInProvider(InProvider& inProvider)
        : inProvider_(inProvider) {}

    __device__ void* getBatchStart(uint32_t batch) {
        uint8_t* p = (uint8_t*)inProvider_.getBatchStart(batch);
        // This is the first place that touches the header
        GpuSparseFloatHeader h = *((GpuSparseFloatHeader*)p);

        // Increment the pointer to past the header and bitmap
        return p + sizeof(GpuSparseFloatHeader) + roundUp((h.size + 7) / 8, 16);
    }

    __device__ const void* getBatchStart(uint32_t batch) const {
        uint8_t* p = (uint8_t*)inProvider_.getBatchStart(batch);

        // This is the first place that touches the header
        GpuSparseFloatHeader h = *((GpuSparseFloatHeader*)p);

        // Increment the pointer to past the header and bitmap
        return p + sizeof(GpuSparseFloatHeader) + roundUp((h.size + 7) / 8, 16);
    }
    InProvider inProvider_;
};

/* Perform sparse float decompression. Called by floatDeompressSparse, in
 * GpuSparseFloatDeompress.cu. This has the same API as floatDeompressDevice.
 */
template <typename InProvider, typename OutProvider>
FloatDecompressStatus floatDecompressSparseDevice(
        // used for allocating all GPU memory
        StackDeviceMemory& res,

        // Config for float decompression. See GpuFloatCodec.h
        const FloatDecompressConfig& config,

        // number of input batches
        uint32_t numInBatch,

        // BatchProvider that tells us where the compressed data for each
        // batch is.
        InProvider& inProvider,

        // BatchProvider that tells us where to write the output (decompressed)
        // data.
        OutProvider& outProvider,

        // Maximum number of output floats
        uint32_t maxCapacity,

        // This will be populated with whether float decompression was successful
        uint8_t* outSuccess_dev,

        // This will be populated with the total number of output floats
        uint32_t* outSize_dev,

        // CUDA execution stream
        cudaStream_t stream) {

    // checksum not allowed in float mode
    assert(!config.ansConfig.useChecksum);

    // Round the maximum capacity to the next multiple of 16
    uint32_t maxCapacityAligned = roundUp(maxCapacity, sizeof(uint4));

    // Bitmap, as a byte array where each element is 1 if the corresponding
    // element of the decompressed data should be nonzero and 0 otherwise.
    // Its size is a multiple of 8, as required by the kernel function
    // bitmap_bits_to_bytes.
    auto bitmaps = res.alloc<uint8_t>(stream, numInBatch*maxCapacityAligned);
    CUDA_VERIFY(cudaMemsetAsync(
      bitmaps.data(),
      0,
      sizeof(uint8_t) * maxCapacityAligned * numInBatch,
      stream));
    
    // The number of threads in a CUDA kernel block is fixed at 256
    constexpr int kThreads = 256; 

    // This function returns the decompression status (i.e., whether it was 
    // successful, as well as any errors that occured)
    FloatDecompressStatus status;

    // Any code that requires the float type (i.e., the element of the enum
    // FloatType corresponding the the datatype we're decompressing) to be a
    // constant, i.e., when the float type is passed in as a template argument,
    // must be in this sort of macro. For decompression, everything needs to be
    // in one big macro.
    #define RUN_DECOMPRESS(FT)                                                \
    do {                                                                      \
        /* Grid size and number of threads for running the generate_bitmap */ \
        /* kernel. */                                                         \
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
                                                                              \
        /* Allocate memory to store the output of the float decompressor, */  \
        /* which contains all of the nonzero floats that were compressed. */  \
        auto nonzerosData = res.alloc<typename FloatTypeInfo<FT>::WordT>(     \
            stream, numInBatch * maxCapacityAligned);                         \
                                                                              \
        /* Input and output BatchProviders for running */                     \
        /* floatDecompressDevice. */                                          \
        auto sparseFloatInProvider = SparseFloatInProvider                    \
            <InProvider, FT>(inProvider);                                     \
        BatchProviderStride sparseFloatOutProvider = BatchProviderStride(     \
            nonzerosData.data(),                                              \
            maxCapacityAligned*sizeof(typename FloatTypeInfo<FT>::WordT),     \
            maxCapacity                                                       \
        );                                                                    \
                                                                              \
        /* Run decompression to get the nonzeros. */                          \
        status = floatDecompressDevice(                                       \
            res, config, numInBatch, sparseFloatInProvider,                   \
            sparseFloatOutProvider, maxCapacity, outSuccess_dev,              \
            outSize_dev, stream                                               \
        );                                                                    \
                                                                              \
        /* The bitmap in the compressed data is packed into bits. In order */ \
        /* for the bitmap to be useful, we need to convert it to a byte */    \
        /* array. */                                                          \
        bitmap_bits_to_bytes<InProvider><<<grid, kThreads, 0, stream>>>(      \
            bitmaps.data(), inProvider, maxCapacityAligned                    \
        );                                                                    \
                                                                              \
        /* Run an exclusive scan on the bitmaps, for each batch. This */      \
        /* gives us the indices of the floatDecompressDevice output that */   \
        /* correspond to each nonzero element of the final decompressed */    \
        /* output. */                                                         \
        auto nonzerosIdx = res.alloc<uint32_t>(stream,                        \
            numInBatch * maxCapacityAligned);                                 \
        cudaDeviceSynchronize();                                              \
        for (int batch = 0; batch < numInBatch; ++batch){                     \
            thrust::exclusive_scan(                                           \
                thrust::device,                                               \
                bitmaps.data() + batch * maxCapacityAligned,                  \
                bitmaps.data() + batch * maxCapacityAligned +                 \
                                            maxCapacityAligned - 1,           \
                (uint32_t*) nonzerosIdx.data() + batch*maxCapacityAligned, 0);\
        }                                                                     \
        cudaDeviceSynchronize();                                              \
                                                                              \
        /* Grid size and number of threads for running the */                 \
        /* fill_in_nonzeros kernel. */                                        \
        maxBlocksPerSM = 0;                                                   \
        CUDA_VERIFY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(            \
            &maxBlocksPerSM,                                                  \
            fill_in_nonzeros<BatchProviderStride,                             \
                    OutProvider, InProvider, FT> ,                            \
            kThreads,                                                         \
            0));                                                              \
        maxGrid = maxBlocksPerSM * props.multiProcessorCount;                 \
        perBatchGrid = divUp(maxGrid, numInBatch);                            \
        if ((perBatchGrid * numInBatch > maxGrid) && perBatchGrid > 1) {      \
            perBatchGrid -= 1;                                                \
        }                                                                     \
        grid = dim3(perBatchGrid, numInBatch);                                \
                                                                              \
        /* Run the fill_in_nonzeros kernel. This produces the final output. */\
        fill_in_nonzeros<BatchProviderStride, OutProvider, InProvider, FT>    \
            <<<grid, kThreads, 0, stream>>>(                                  \
            sparseFloatOutProvider, outProvider, inProvider,                  \
            bitmaps.data(), nonzerosIdx.data(), outSize_dev,                  \
            maxCapacityAligned                                                \
        );                                                                    \
    } while(false);
    
    // Run the above macro, passing in the appropriate float type
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