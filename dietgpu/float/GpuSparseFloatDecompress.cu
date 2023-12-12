#include "dietgpu/ans/BatchProvider.cuh"
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuSparseFloatDecompress.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/StackDeviceMemory.h"

#include <glog/logging.h>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace dietgpu {

FloatDecompressStatus floatDecompressSparse(
    StackDeviceMemory& res,
    const FloatDecompressConfig& config,
    uint32_t numInBatch,
    const void** in,
    void** out,
    const uint32_t* outCapacity,
    uint8_t* outSuccess_dev,
    uint32_t* outSize_dev,
    cudaStream_t stream) {
  // If the batch size is <= kBSLimit, we avoid cudaMemcpy and send all data at
  // kernel launch
  constexpr int kLimit = 128;

  bool is16ByteAligned = true;
  for (int i = 0; i < numInBatch; ++i) {
    if (reinterpret_cast<uintptr_t>(out[i]) % 16 != 0) {
      is16ByteAligned = false;
      break;
    }
  }

  auto updatedConfig = config;
  updatedConfig.is16ByteAligned = is16ByteAligned;

  // We need a max capacity estimate before proceeding, for temporary memory
  // allocations
  uint32_t maxCapacity = 0;
  for (uint32_t i = 0; i < numInBatch; ++i) {
    maxCapacity = std::max(maxCapacity, outCapacity[i]);
  }

  if (numInBatch <= kLimit) {
    // We can do everything in a single pass without a h2d memcpy
    auto inProvider =
        BatchProviderInlinePointer<kLimit>(numInBatch, (void**)in);
    auto outProvider = BatchProviderInlinePointerCapacity<kLimit>(
        numInBatch, out, outCapacity);

    return floatDecompressSparseDevice(
        res,
        updatedConfig,
        numInBatch,
        inProvider,
        outProvider,
        maxCapacity,
        outSuccess_dev,
        outSize_dev,
        stream);
  }

  // Copy data to device
  // To reduce latency, we prefer to coalesce all data together and copy as one
  // contiguous chunk
  static_assert(sizeof(void*) == sizeof(uintptr_t));
  static_assert(sizeof(uint32_t) <= sizeof(uintptr_t));

  // in, out, outCapacity
  auto params_dev = res.alloc<uintptr_t>(stream, numInBatch * 3);
  auto params_host =
      std::unique_ptr<uintptr_t[]>(new uintptr_t[3 * numInBatch]);

  std::memcpy(&params_host[0], in, numInBatch * sizeof(void*));
  std::memcpy(&params_host[numInBatch], out, numInBatch * sizeof(void*));
  std::memcpy(
      &params_host[2 * numInBatch], outCapacity, numInBatch * sizeof(uint32_t));

  CUDA_VERIFY(cudaMemcpyAsync(
      params_dev.data(),
      params_host.get(),
      3 * numInBatch * sizeof(uintptr_t),
      cudaMemcpyHostToDevice,
      stream));

  auto in_dev = params_dev.data();
  auto out_dev = params_dev.data() + numInBatch;
  auto outCapacity_dev = (const uint32_t*)(params_dev.data() + 2 * numInBatch);

  auto inProvider = BatchProviderPointer((void**)in_dev);
  auto outProvider = BatchProviderPointer((void**)out_dev, outCapacity_dev);

  return floatDecompressSparseDevice(
      res,
      updatedConfig,
      numInBatch,
      inProvider,
      outProvider,
      maxCapacity,
      outSuccess_dev,
      outSize_dev,
      stream);
}
}