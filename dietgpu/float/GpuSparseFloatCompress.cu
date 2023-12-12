#include "dietgpu/ans/BatchProvider.cuh"
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuSparseFloatCompress.cuh"
#include "dietgpu/float/GpuFloatUtils.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"

#include <glog/logging.h>
#include <cmath>
#include <memory>
#include <vector>

namespace dietgpu {

uint32_t getMaxSparseFloatCompressedSize(FloatType floatType, uint32_t size) {
  // kNotCompressed bytes per float are simply stored uncompressed
  // rounded up to 16 bytes to ensure alignment of the following ANS data
  // portion
  uint32_t baseSize = sizeof(GpuFloatHeader) + sizeof(GpuFloatHeader2) + 
    sizeof(GpuSparseFloatHeader) + getMaxCompressedSize(size) + roundUp(size, 16);

  switch (floatType) {
    case FloatType::kFloat16:
      baseSize += FloatTypeInfo<FloatType::kFloat16>::getUncompDataSize(size);
      break;
    case FloatType::kBFloat16:
      baseSize += FloatTypeInfo<FloatType::kBFloat16>::getUncompDataSize(size);
      break;
    case FloatType::kFloat32:
      baseSize += FloatTypeInfo<FloatType::kFloat32>::getUncompDataSize(size);
      break;
    case FloatType::kFloat64:
      baseSize += FloatTypeInfo<FloatType::kFloat64>::getUncompDataSize(size) + getMaxCompressedSize(size);
      break;
    default:
      CHECK(false);
      break;
  }

  return baseSize;
}

void floatCompressSparse(
    StackDeviceMemory& res,
    const FloatCompressConfig& config,
    uint32_t numInBatch,
    const void** in,
    const uint32_t* inSize,
    void** out,
    uint32_t* outSize_dev,
    cudaStream_t stream) {
  // Get the total and maximum input size
  uint32_t maxSize = 0;

  for (uint32_t i = 0; i < numInBatch; ++i) {
    maxSize = std::max(maxSize, inSize[i]);
  }

  // Copy data to device
  // To reduce latency, we prefer to coalesce all data together and copy as one
  // contiguous chunk
  static_assert(sizeof(void*) == sizeof(uintptr_t), "");
  static_assert(sizeof(uint32_t) <= sizeof(uintptr_t), "");

  // in, inSize, out
  auto params_dev = res.alloc<uintptr_t>(stream, numInBatch * 3);
  auto params_host =
      std::unique_ptr<uintptr_t[]>(new uintptr_t[3 * numInBatch]);

  std::memcpy(&params_host[0], in, numInBatch * sizeof(void*));
  std::memcpy(&params_host[numInBatch], inSize, numInBatch * sizeof(uint32_t));
  std::memcpy(&params_host[2 * numInBatch], out, numInBatch * sizeof(void*));

  CUDA_VERIFY(cudaMemcpyAsync(
      params_dev.data(),
      params_host.get(),
      3 * numInBatch * sizeof(uintptr_t),
      cudaMemcpyHostToDevice,
      stream));

  auto in_dev = (const void**)params_dev.data();
  auto inSize_dev = (const uint32_t*)(params_dev.data() + numInBatch);
  auto out_dev = (void**)(params_dev.data() + 2 * numInBatch);

  auto inProvider = BatchProviderPointer((void**)in_dev, inSize_dev);
  auto outProvider = BatchProviderPointer(out_dev);

  floatCompressSparseDevice(
      res,
      config,
      numInBatch,
      inProvider,
      maxSize,
      outProvider,
      outSize_dev,
      stream);
}

} // namespace dietgpu
