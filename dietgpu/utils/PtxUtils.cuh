/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>

namespace dietgpu {

__device__ __forceinline__ unsigned int
getBitfield(uint8_t val, int pos, int len) {
  unsigned int ret;
  asm("bfe.u32 %0, %1, %2, %3;"
      : "=r"(ret)
      : "r"((uint32_t)val), "r"(pos), "r"(len));
  return ret;
}

__device__ __forceinline__ unsigned int
getBitfield(uint16_t val, int pos, int len) {
  unsigned int ret;
  asm("bfe.u32 %0, %1, %2, %3;"
      : "=r"(ret)
      : "r"((uint32_t)val), "r"(pos), "r"(len));
  return ret;
}

__device__ __forceinline__ unsigned int
getBitfield(unsigned int val, int pos, int len) {
  unsigned int ret;
  asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
  return ret;
}

__device__ __forceinline__ uint64_t
getBitfield(uint64_t val, int pos, int len) {
  uint64_t ret;
  asm("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(val), "r"(pos), "r"(len));
  return ret;
}

__device__ __forceinline__ unsigned int
setBitfield(unsigned int val, unsigned int toInsert, int pos, int len) {
  unsigned int ret;
  asm("bfi.b32 %0, %1, %2, %3, %4;"
      : "=r"(ret)
      : "r"(toInsert), "r"(val), "r"(pos), "r"(len));
  return ret;
}

__device__ __forceinline__ uint32_t rotateLeft(uint32_t v, uint32_t shift) {
  uint32_t out;
  asm("shf.l.clamp.b32 %0, %1, %2, %3;"
      : "=r"(out)
      : "r"(v), "r"(v), "r"(shift));
  return out;
}

__device__ __forceinline__ uint32_t rotateRight(uint32_t v, uint32_t shift) {
  uint32_t out;
  asm("shf.r.clamp.b32 %0, %1, %2, %3;"
      : "=r"(out)
      : "r"(v), "r"(v), "r"(shift));
  return out;
}

__device__ __forceinline__ uint64_t rotateLeft(uint64_t v, uint32_t shift) {
  /* From documentation:
    // 128-bit left shift; n < 32
    // [r7,r6,r5,r4] = [r3,r2,r1,r0] << n
    shf.l.clamp.b32  r7,r2,r3,n;
    shf.l.clamp.b32  r6,r1,r2,n;
    shf.l.clamp.b32  r5,r0,r1,n;
    shl.b32          r4,r0,n;

    Let v1 be the least-significant 32 bits of v and v2 be the most significant 32 bits.
    then the most significant 64 bits of [v2, v1, v2, v1] << 1 are what we want.
  */
  uint32_t r7, r6, v1, v2;
  v1 = v & 0xffffffff;
  v2 = v >> 32;
  asm("shf.l.clamp.b32 %0, %1, %2, %3;"
      : "=r"(r7)
      : "r"(v1), "r"(v2), "r"(shift));
  asm("shf.l.clamp.b32 %0, %1, %2, %3;"
      : "=r"(r6)
      : "r"(v2), "r"(v1), "r"(shift));

  uint64_t out = ((uint64_t) r6) + (((uint64_t) r7) << 32);
  return out;
}

__device__ __forceinline__ uint64_t rotateRight(uint64_t v, uint32_t shift) {
  /* From documentation:
    // 128-bit right shift, arithmetic; n < 32
    // [r7,r6,r5,r4] = [r3,r2,r1,r0] >> n
    shf.r.clamp.b32  r4,r0,r1,n;
    shf.r.clamp.b32  r5,r1,r2,n;
    shf.r.clamp.b32  r6,r2,r3,n;
    shr.s32          r7,r3,n;     // result is sign-extended

    Let v1 be the least-significant 32 bits of v and v2 be the most significant 32 bits.
    then the least significant 64 bits of [v2, v1, v2, v1] >> 1 are what we want.
  */
  uint32_t r4, r5, v1, v2;
  v1 = v & 0xffffffff;
  v2 = v >> 32;
  asm("shf.r.clamp.b32 %0, %1, %2, %3;"
      : "=r"(r4)
      : "r"(v1), "r"(v2), "r"(shift));
  asm("shf.r.clamp.b32 %0, %1, %2, %3;"
      : "=r"(r5)
      : "r"(v2), "r"(v1), "r"(shift));
  uint64_t out = ((uint64_t) r4) + (((uint64_t) r5) << 32);
  return out;
}

__device__ __forceinline__ int getLaneId() {
  int laneId;
  asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
  return laneId;
}

__device__ __forceinline__ unsigned getLaneMaskLt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ unsigned getLaneMaskLe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ unsigned getLaneMaskGt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ unsigned getLaneMaskGe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_ge;" : "=r"(mask));
  return mask;
}

template <typename T>
__device__ inline T warpReduceAllMin(T val) {
#if __CUDA_ARCH__ >= 800
  return __reduce_min_sync(0xffffffff, val);
#else
#pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    val = min(val, __shfl_xor_sync(0xffffffff, val, mask, kWarpSize));
  }

  return val;
#endif
}

template <typename T, int Width = kWarpSize>
__device__ inline T warpReduceAllMax(T val) {
#if __CUDA_ARCH__ >= 800
  return __reduce_max_sync(0xffffffff, val);
#else
#pragma unroll
  for (int mask = Width / 2; mask > 0; mask >>= 1) {
    val = max(val, __shfl_xor_sync(0xffffffff, val, mask, kWarpSize));
  }

  return val;
#endif
}

template <typename T, int Width = kWarpSize>
__device__ inline T warpReduceAllSum(T val) {
#if __CUDA_ARCH__ >= 800
  return __reduce_add_sync(0xffffffff, val);
#else
#pragma unroll
  for (int mask = Width / 2; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask, kWarpSize);
  }

  return val;
#endif
}

} // namespace dietgpu
