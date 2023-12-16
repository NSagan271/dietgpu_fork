
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

/* Given input float data, generate one bitmap per batch storing whether each
 * float is zero or nonzero. In the context of this function, a bitmap is a
 * byte array, where each element is either 0 or 1. bitmap_bytes_to_bits is
 * used to convert such an array to actual bits.
*/
template <
    typename InProvider,
    FloatType FT>
__global__ void generate_bitmap(
    // BatchProvider for the input floats
    InProvider inProvider,

    // Byte array output
    uint8_t* bitmaps,

    // rowStride is the distance, in array elements, between adjacent batches
    // in the output "bitmaps" array. This should be 16-byte aligned.
    uint32_t rowStride
) {
    using FTI = FloatTypeInfo<FT>;
    using WordT = typename FloatTypeInfo<FT>::WordT;

    int batch = blockIdx.y;

    // Inputs and outputs for the current batch
    auto curIn = (const WordT*) inProvider.getBatchStart(batch);
    auto curSize = inProvider.getBatchSize(batch);
    auto curOut = bitmaps + batch * rowStride;

    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < curSize;
            i += gridDim.x * blockDim.x) {
        curOut[i] = (curIn[i] == 0) ? 0 : 1;
    }
}

/* Given a series of bitmaps (one per each batch), as byte arrays where each
 * element is either 0 or 1, pack each into a series of bits, written to
 * the appropriate batch in BatchProvider outProvider.
 */
template<typename OutProvider, typename SizeProvider>
__global__ void bitmap_bytes_to_bits(
    // input bitmaps, as a byte array. Its length must be a multiple of 8;
    // if the number of input floats is not a mutliple of 8, the bitmaps should
    // be zero-padded at the end.
    uint8_t* bitmap_bytes,

    // BatchProvider for writing the outputs
    OutProvider outProvider,

    // any BatchProvider such that sizeProvider.getBatchSize(batch) produces
    // the number of floats in the corresponding batch. e.g., the InProvider
    // to floatCompressSparseDevice
    SizeProvider sizeProvider,

    // rowStride is the distance, in array elements, between adjacent batches
    // in the input "bitmaps" array. This should be 16-byte aligned.
    uint32_t rowStride
) {
    int batch = blockIdx.y;

    auto curIn = bitmap_bytes + batch * rowStride;
    auto curSize = sizeProvider.getBatchSize(batch);

    // Write the GpuSparseFloatHeader to each batch's output, specifying
    // the total number of floats (both zero and nonzero) in the batch
    auto headerOut = (GpuSparseFloatHeader*) outProvider.getBatchStart(batch);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        GpuSparseFloatHeader h;
        h.setSize(curSize);
        *headerOut = h;
    }

    // The output bitmap will be written right after the header.
    auto curOut = (uint8_t *) (headerOut + 1);

    // Read in batches of 8 floats in the bitmap
    auto curInV = (uint8x8*) curIn;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (curSize + 7) / 8; 
            i+=gridDim.x * blockDim.x) {
        uint8x8 v = curInV[i];

        // convert each group of 8 elements of the bitmap to one
        // 8-bit integer.
        curOut[i] = v.x[7] | (v.x[6] << 1) | (v.x[5] << 2) |
                    (v.x[4] << 3) | (v.x[3] << 4) | (v.x[2] << 5) |
                    (v.x[1] << 6) | (v.x[0] << 7);

    }
}

/* Given the bitmap output of generate_bitmap, and an exclusive scan performed
 * on it, compute the input to the float compressor: i.e., a dataset that only
 * contains the nonzero elements of the input data.
 */
template <
    typename InProvider,
    FloatType FT>
__global__ void fill_comp_input(
    // BatchProvider for input floats
    InProvider inProvider,

    // Output array that only contains the nonzero floats from each batch
    typename FloatTypeInfo<FT>::WordT* nonzerosData,

    // result of generate_bitmap
    uint8_t* bitmaps,

    // result of performing an exclusive scan on each batch's bitmap.
    // Where bitmask is 1, nonzerosIdx gives the corresponding index of the
    // nonzeros dataset, aka nonzerosData
    uint32_t* nonzerosIdx,

    // Array of length equivalent to the number of batches. It will be populated
    // with the number of nonzero floats in each batch
    uint32_t* nonzerosBatchSizes,

    // This array will be populated with the start of each batch in nonzerosData.
    // It is required in the construction of the BatchProvider that will be
    // eventually passed into floatCompressDevice.
    void** batchPointers,

    // rowStride is the distance, in array elements, between adjacent batches
    // in the input "bitmaps," "nonzerosIdx," and "nonzerosData" arrays.
    uint32_t rowStride
) {
    using FTI = FloatTypeInfo<FT>;
    using WordT = typename FloatTypeInfo<FT>::WordT;

    int batch = blockIdx.y;

    // Input, output, bitmap, and nonzerosIdx for this batch
    auto curIn = (const WordT*) inProvider.getBatchStart(batch);
    auto curBitmap = bitmaps + batch * rowStride;
    auto curOut = nonzerosData + batch * rowStride;
    auto curSize = inProvider.getBatchSize(batch); // total number of floats (zero and nonzero)
    uint32_t* curNonzerosIdx = nonzerosIdx + batch * rowStride;

    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < curSize-1;
            i += gridDim.x * blockDim.x) {
        if (curBitmap[i] == 1) {
            curOut[curNonzerosIdx[i]] = curIn[i];
        }
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // set the batchPointers array for this batch
        batchPointers[batch] = (void*) curOut;

        // Special handling is required for the last input float, as the
        // exclusive scan does not properly populate the last element of
        // curNonzerosIdx
        if (curBitmap[curSize-1] == 1) {
            curOut[curNonzerosIdx[curSize-2] + 1] = curIn[curSize-1];
        }

        // set the number of nonzero floats for this batch, aka the number of
        // floats in curOut
        nonzerosBatchSizes[batch] = curNonzerosIdx[curSize-2] + curBitmap[curSize-1] + 1;
    }
}

/* This is a version of a BatchProvider, and it tells the float compressor where
 * to write the compressed output. It takes the OutProvider passed into
 * floatCompressSparseDevice, and it finds the first 16-byte aligned address
 * after the sparse float header and the bitmap.
 */
template <FloatType FT, typename OutProvider, typename SizeProvider>
struct SparseFloatOutProvider {
    using Writer = BatchWriter;
    using FTI = FloatTypeInfo<FT>;

    __host__ SparseFloatOutProvider(
        // OutProvider from floatCompressSparseDevice
        OutProvider& outProvider,

        // any BatchProvider such that sizeProvider.getBatchSize(batch) produces
        // the number of floats in the corresponding batch. e.g., the InProvider
        // to floatCompressSparseDevice
        SizeProvider& sizeProvider)
        : outProvider_(outProvider), sizeProvider_(sizeProvider) {
        }

    __device__ void* getBatchStart(uint32_t batch) {
        uint8_t* p = (uint8_t*)outProvider_.getBatchStart(batch);

        // Increment the pointer to past the header and bitmap data
        return p + sizeof(GpuSparseFloatHeader) +
            roundUp((sizeProvider_.getBatchSize(batch) + 7) / 8, 16);
    }

    __device__ const void* getBatchStart(uint32_t batch) const {
        const uint8_t* p = (const uint8_t*)outProvider_.getBatchStart(batch);

        // Increment the pointer to past the header bitmap data
        return p + sizeof(GpuSparseFloatHeader) +
            roundUp((sizeProvider_.getBatchSize(batch) + 7) / 8, 16);
    }

    __device__ BatchWriter getWriter(uint32_t batch) {
        return BatchWriter(getBatchStart(batch));
    }

    OutProvider outProvider_;
    SizeProvider sizeProvider_;
};

/* After float compression is complete, the outSize_dev array will be populated
 * with the number of bytes of compressed data. This, however, does not include
 * the sparse float header and the bitmap. So, we need to the bytes from those
 * to outSize_dev.
 */
template<typename SizeProvider>
__global__ void addBitmapToOutSizes(SizeProvider sizeProvider, uint32_t* outSize_dev) {
    int batch = blockIdx.y;

    // Only the first thread from this batch need do anything, as there is only
    // one computation to do per batch
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        outSize_dev[batch] += roundUp((sizeProvider.getBatchSize(batch) + 7) / 8, 16) +
                             sizeof(GpuSparseFloatHeader);
    }
}

/* Perform sparse float compression. Called by floatCompressSparse, in
 * GpuSparseFloatCompress.cu. This has the same API as floatCompressDevice,
 * but the algorithm is specialized for sparse floating point data.
 */
template <typename InProvider, typename OutProvider>
void floatCompressSparseDevice(
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

        // This will be populated with the number of compressed bytes for each batch
        uint32_t* outSize_dev,

        // CUDA execution stream
        cudaStream_t stream) {

    // checksum is not allowed in float mode
    assert(!config.ansConfig.useChecksum);

    // The number of threads in a CUDA kernel block is fixed at 256
    constexpr int kBlock = 256; 

    // Number of array elements between adjacent batches in any array
    // that contains data for multiple batches. This will always be a
    // multiple of 16 for memory alignment reasons.
    uint32_t rowStride = roundUp(maxSize, sizeof(uint4));

    // Bitmap, as a byte array where each element is 1 if the corresponding
    // input float is nonzero and 0 otherwise. Its size is a multiple of 8,
    // as required by the kernel function bitmap_bytes_to_bits.
    auto bitmaps = res.alloc<uint8_t>(stream, numInBatch * rowStride);

    // set the bitmap to zero
    CUDA_VERIFY(cudaMemsetAsync(
      bitmaps.data(),
      0,
      sizeof(uint8_t) * rowStride * numInBatch,
      stream));

    // Any code that requires the float type (i.e., the element of the enum
    // FloatType corresponding the the datatype we're compressing) to be a
    // constant, i.e., when the float type is passed in as a template argument,
    // must be in this sort of macro.
    #define RUN_BITMAP(FT)                                             \
    do {                                                               \
        /* Grid size and number of threads for running the */          \
        /* generate_bitmap kernel. */                                  \
        auto& props = getCurrentDeviceProperties();                    \
        int maxBlocksPerSM = 0;                                        \
        CUDA_VERIFY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(     \
            &maxBlocksPerSM,                                           \
            generate_bitmap<InProvider, FT>,                           \
            kBlock,                                                    \
            0));                                                       \
        uint32_t maxGrid = maxBlocksPerSM * props.multiProcessorCount; \
        uint32_t perBatchGrid = 4 * divUp(maxGrid, numInBatch);        \
        auto grid = dim3(perBatchGrid, numInBatch);                    \
                                                                       \
        /* run the generate_bitmap kernel */                           \
        generate_bitmap<InProvider, FT><<<grid, kBlock, 0, stream>>>(  \
            inProvider, bitmaps.data(), rowStride                      \
        );                                                             \
        /* Pack the bitmap byte array into bits and write it to the */ \
        /* compression output */                                       \
        bitmap_bytes_to_bits<OutProvider, InProvider>                  \
            <<<grid, kBlock, 0, stream>>>(                             \
                bitmaps.data(), outProvider, inProvider, rowStride     \
        );                                                             \
    } while(false);

    // Run the above macro, passing in the appropriate float type
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

    auto nonzerosIdx = res.alloc<uint32_t>(stream, numInBatch * rowStride);
    auto nonzerosBatchSizes = res.alloc<uint32_t>(stream, numInBatch);
    auto batchPointers = res.alloc<uintptr_t>(stream, numInBatch);
    
    // Run an exclusive scan on the bitmaps, for each batch. This lets us
    // create a dataset with only the nonzero elements of the input data.
    // See the fill_comp_input kernel for more details.
    for (int batch = 0; batch < numInBatch; ++batch){
        thrust::exclusive_scan(
            thrust::device,
            bitmaps.data() + batch * rowStride,
            bitmaps.data() + batch * rowStride + rowStride - 1,
            nonzerosIdx.data() + batch*rowStride, 0);
    }

    cudaDeviceSynchronize();

    // Again, we need the code that calls fill_comp_input and floatCompressDevice
    // to be in a macro so that the float type can be kept as a constant
    #define RUN_COMPRESS(FT)                                           \
    do {                                                               \
        /* Grid size and number of threads for running the */          \
        /* fill_comp_input kernel. */                                  \
        auto& props = getCurrentDeviceProperties();                    \
        int maxBlocksPerSM = 0;                                        \
        CUDA_VERIFY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(     \
            &maxBlocksPerSM,                                           \
            fill_comp_input<InProvider, FT>,                           \
            kBlock,                                                    \
            0));                                                       \
        uint32_t maxGrid = maxBlocksPerSM * props.multiProcessorCount; \
        uint32_t perBatchGrid = 4 * divUp(maxGrid, numInBatch);        \
        auto grid = dim3(perBatchGrid, numInBatch);                    \
                                                                       \
        /* Allocate memory for storing the dataset containing all*/    \
        /* nonzero elements of the input data. */                      \
        auto nonzerosData = res.alloc                                  \
                <typename FloatTypeInfo<FT>::WordT>(                   \
            stream, numInBatch * rowStride);                           \
                                                                       \
        /* Call the fill_comp_input kernel */                          \
        fill_comp_input<InProvider, FT><<<grid, kBlock, 0, stream>>> ( \
            inProvider, nonzerosData.data(), bitmaps.data(),           \
            nonzerosIdx.data(), nonzerosBatchSizes.data(),             \
            (void**) batchPointers.data(), rowStride                   \
        );                                                             \
                                                                       \
        /* Set up the inputs and outputs to floatCompressDevice,*/     \
        /* which performs float compression on the nonzero elements */ \
        /* of the original input data. */                              \
        auto sparseFloatOutProvider = SparseFloatOutProvider           \
            <FT, OutProvider, InProvider>(                             \
            outProvider, inProvider                                    \
        );                                                             \
        auto sparseFloatInProvider = BatchProviderPointer(             \
            (void**) batchPointers.data(),                             \
            nonzerosBatchSizes.data());                                \
                                                                       \
        /* Run float compression */                                    \
        floatCompressDevice(                                           \
            res, config, numInBatch, sparseFloatInProvider, maxSize,   \
            sparseFloatOutProvider, outSize_dev, stream                \
        );                                                             \
                                                                       \
        /* The output sizes set by floatCompressDevice do not */       \
        /* include the bitmap and sparse float header. So, the */      \
        /* bytes used for those need to be added to outSize_dev */     \
        addBitmapToOutSizes<<<grid, kBlock, 0, stream>>>(              \
            inProvider, outSize_dev                                    \
        );                                                             \
    } while(false);

    // Run the above macro, passing in the appropriate float type
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
} //namespace dietgpu