/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "dietgpu/utils/DeviceDefs.cuh"
#include "dietgpu/utils/PtxUtils.cuh"
#include "dietgpu/utils/StaticUtils.h"

#include <cuda.h>
#include <glog/logging.h>

namespace dietgpu {

// magic number to verify archive integrity
constexpr uint32_t kFloatMagic = 0xf00f;

// current DietGPU version number
constexpr uint32_t kFloatVersion = 0x0001;

// Header on our compressed floating point data
struct __align__(16) GpuFloatHeader {
  __host__ __device__ void setMagicAndVersion() {
    magicAndVersion = (kFloatMagic << 16) | kFloatVersion;
  }

  __host__ __device__ void checkMagicAndVersion() const {
    assert((magicAndVersion >> 16) == kFloatMagic);
    assert((magicAndVersion & 0xffffU) == kFloatVersion);
  }

  __host__ __device__ FloatType getFloatType() const {
    return FloatType(options & 0xf);
  }

  __host__ __device__ void setFloatType(FloatType ft) {
    assert(uint32_t(ft) <= 0xf);
    options = (options & 0xfffffff0U) | uint32_t(ft);
  }

  __host__ __device__ bool getUseChecksum() const {
    return options & 0x10;
  }

  __host__ __device__ void setUseChecksum(bool uc) {
    options = (options & 0xffffffef) | (uint32_t(uc) << 4);
  }

  __host__ __device__ uint32_t getChecksum() const {
    return checksum;
  }

  __host__ __device__ void setChecksum(uint32_t c) {
    checksum = c;
  }

  // (16: magic)(16: version)
  uint32_t magicAndVersion;

  // Number of floating point words of the given float type in the archive
  uint32_t size;

  // (27: unused)(1: use checksum)(4: float type)
  uint32_t options;

  // Optional checksum computed on the input data
  uint32_t checksum;
};

// For float64 compression, there are two rounds of ANS compression. So, we
// need additional header space to store the number of bytes in the first
// compressed segment, so we can easily find where the second compressed segment
// starts during decompression.
struct __align__(16) GpuFloatHeader2 {
  __host__ __device__ uint32_t getFirstCompSegmentBytes() const {
    return firstCompSegmentBytes;
  }

  __host__ __device__ void setFirstCompSegmentBytes(uint32_t b) {
    firstCompSegmentBytes = b;
  }

  // Number of bytes in the first segment of compressed data (for Float64 where
  // there are two different segments of compressed data)
  uint32_t firstCompSegmentBytes;

  // For 16-byte alignment purposes, we need to have these extra fields.
  // Otherwise, the size of this struct will not be 16 bytes and the section
  // of memory directly after it will not be 16-byte aligned.
  uint32_t unusedOne;
  uint64_t unusedTwo;
};

// For sparse floating point compression, we form a bitmap of whether each
// index of the input data is zero or nonzero, write the bitmap to the output,
// and then run regular float compression. This header is placed before the
// bitmap and lets us calculate where the bitmap ends, so that we can find
// the start of the float compressed data.
//
// Note that the "size" field of the GpuFloatHeader is the number of nonzeros
// in the dataset, whereas this "size" field is the total number of floats
// (zeros and nonzeros).
struct __align__(16) GpuSparseFloatHeader {
  __host__ __device__ uint32_t getSize() const {
    return size;
  }

  __host__ __device__ void setSize(uint32_t s) {
    size = s;
  }

  // Number of floating point words of the given float type in the archive
  uint32_t size;

  // For 16-byte alignment purposes, we need to have these extra fields.
  // Otherwise, the size of this struct will not be 16 bytes and the section
  // of memory directly after it will not be 16-byte aligned.
  uint32_t unusedOne;
  uint64_t unusedTwo;
};

static_assert(sizeof(GpuFloatHeader) == 16, "");
static_assert(sizeof(GpuFloatHeader2) == 16, "");
static_assert(sizeof(GpuSparseFloatHeader) == 16, "");

// Different vector datatypes for vectorized operations, i.e., operations
// that read full 16-byte blocks of memory at a time. Vectorized operations
// are an efficient way to access GPU memory.
struct __align__(16) uint64x2 {
  uint64_t x[2];
};

struct __align__(16) uint64x4 {
  uint64_t x[4];
};

struct __align__(16) uint32x4 {
  uint32_t x[4];
};

struct __align__(8) uint32x2 {
  uint32_t x[2];
};

struct __align__(16) uint16x8 {
  uint16_t x[8];
};

struct __align__(8) uint16x4 {
  uint16_t x[4];
};

struct __align__(4) uint16x2 {
  uint16_t x[2];
};

struct __align__(8) uint8x8 {
  uint8_t x[8];
};

struct __align__(4) uint8x4 {
  uint8_t x[4];
};

struct __align__(2) uint8x2 {
  uint8_t x[2];
};

// Convert FloatType to word size/type
template <FloatType FT>
struct FloatTypeInfo;

template <>
struct FloatTypeInfo<FloatType::kFloat16> {
  using WordT = uint16_t;
  using CompT = uint8_t;
  using NonCompT = uint8_t;

  using NonCompSplit1T = uint8_t;
  using NonCompSplit2T = uint8_t; // UNUSED

  // 16 byte vector type
  using VecT = uint16x8;
  using CompVecT = uint8x8;
  using NonCompVecT = uint8x8;

  using NonCompVecSplit1T = uint8x8;
  using NonCompVecSplit2T = uint8x8; // UNUSED

  static __device__ void split(WordT in, CompT* comp, NonCompT& nonComp) {
    // don't bother extracting the specific exponent
    *comp = in >> 8;
    nonComp = in & 0xff;
  }

  static __device__ WordT join(const CompT* comp, NonCompT nonComp) {
    return WordT(*comp) * WordT(256) + WordT(nonComp);
  }

  // How many bytes of data are in the non-compressed portion past the float
  // header?
  static __host__ __device__ uint32_t getUncompDataSize(uint32_t size) {
    // The size of the uncompressed data is always a multiple of 16 bytes, to
    // guarantee alignment for proceeding data segments
    return roundUp(size, 16 / sizeof(NonCompT));
  }

  static __device__ size_t getNumCompSegments() {
    return 1;
  }

  static __device__ bool getIfNonCompSplit() {
    return false;
  }
};

template <>
struct FloatTypeInfo<FloatType::kBFloat16> {
  using WordT = uint16_t;
  using CompT = uint8_t;
  using NonCompT = uint8_t;

  using NonCompSplit1T = uint8_t;
  using NonCompSplit2T = uint8_t; // UNUSED

  // 16 byte vector type
  using VecT = uint16x8;
  using CompVecT = uint8x8;
  using NonCompVecT = uint8x8;

  using NonCompVecSplit1T = uint8x8;
  using NonCompVecSplit2T = uint8x8; // UNUSED

  static __device__ void split(WordT in, CompT* comp, NonCompT& nonComp) {
    uint32_t v = uint32_t(in) * 65536U + uint32_t(in);

    v = rotateLeft(v, 1);
    *comp = v >> 24;
    nonComp = v & 0xff;
  }

  static __device__ WordT join(const CompT *comp, NonCompT nonComp) {
    uint32_t lo = uint32_t(*comp) * 256U + uint32_t(nonComp);
    lo <<= 16;
    uint32_t hi = nonComp;

    uint32_t out;
    asm("shf.r.clamp.b32 %0, %1, %2, %3;"
        : "=r"(out)
        : "r"(lo), "r"(hi), "r"(1));
    return out >>= 16;
  }

  // How many bytes of data are in the non-compressed portion past the float
  // header?
  static __host__ __device__ uint32_t getUncompDataSize(uint32_t size) {
    // The size of the uncompressed data is always a multiple of 16 bytes, to
    // guarantee alignment for proceeding data segments
    return roundUp(size, 16 / sizeof(NonCompT));
  }

  static __device__ size_t getNumCompSegments() {
    return 1;
  }

  static __device__ bool getIfNonCompSplit() {
    return false;
  }
};

template <>
struct FloatTypeInfo<FloatType::kFloat32> {
  using WordT = uint32_t;
  using CompT = uint8_t;
  using NonCompT = uint32_t;

  using NonCompSplit1T = uint16_t;
  using NonCompSplit2T = uint8_t;

  // 16 byte vector type
  using VecT = uint32x4;
  using CompVecT = uint8x4;
  using NonCompVecT = uint32x4;

  using NonCompVecSplit1T = uint16x4;
  using NonCompVecSplit2T = uint8x4;

  static __device__ void split(WordT in, CompT* comp, NonCompT& nonComp) {
    auto v = rotateLeft(in, 1);
    *comp = v >> 24;
    nonComp = v & 0xffffffU;
  }

  static __device__ WordT join(const CompT* comp, NonCompT nonComp) {
    uint32_t v = (uint32_t(*comp) * 16777216U) + uint32_t(nonComp);
    return rotateRight(v, 1);
  }

  // How many bytes of data are in the non-compressed portion past the float
  // header?
  static __host__ __device__ uint32_t getUncompDataSize(uint32_t size) {
    // The size of the uncompressed data is always a multiple of 16 bytes, to
    // guarantee alignment for proceeding data segments
    // We store the low order 2 bytes first, then the high order uncompressed
    // byte afterwards.
    // Both sections should be 16 byte aligned
    return 2 * roundUp(size, 8) + // low order 2 bytes
        roundUp(size, 16); // high order 1 byte, starting at an aligned address
                           // after the low 2 byte segment
  }

  static __device__ size_t getNumCompSegments() {
    return 1;
  }

  static __device__ bool getIfNonCompSplit() {
    return true;
  }
};

template <>
struct FloatTypeInfo<FloatType::kFloat64> {
  using WordT = uint64_t;
  using CompT = uint8_t;
  using NonCompT = uint64_t;

  using NonCompSplit1T = uint32_t;
  using NonCompSplit2T = uint16_t;

  // 16 byte vector type
  using VecT = uint64x2;
  using CompVecT = uint8x2;
  using NonCompVecT = uint64x2;

  using NonCompVecSplit1T = uint32x2;
  using NonCompVecSplit2T = uint16x2;

  static __device__ void split(WordT in, CompT* comp, NonCompT& nonComp) {
    uint64_t v = rotateLeft(in, 1);
    // For Float64, there are two rounds of ANS compression because the exponent
    // is longer than one byte. So, we store the compressed part as an array.
    comp[0] = v >> 56;
    comp[1] = (v >> 48) & 0xffU;

    nonComp = v & 0xffffffffffffU;
  }

  static __device__ WordT join(const CompT* comp, NonCompT nonComp) {
    uint64_t v = (uint64_t(comp[0]) * 72057594037927936U) + (uint64_t(comp[1]) * 281474976710656U) + 
                  uint64_t(nonComp);
    return rotateRight(v, 1);
  }

  static __device__ NonCompVecT mulV(const CompVecT comp, const uint64_t intVal) {
    uint64x2 v;
    v.x[0] = uint64_t(comp.x[0]) * intVal;
    v.x[1] = uint64_t(comp.x[1]) * intVal;
    return v;
  }

  static __device__ NonCompVecT addV(const NonCompVecT comp1, const NonCompVecT comp2) {
    uint64x2 v;
    v.x[0] = comp1.x[0] + comp2.x[0];
    v.x[1] = comp1.x[1] + comp2.x[1];
    return v;
  }

  // How many bytes of data are in the non-compressed portion past the float
  // header?
  static __host__ __device__ uint32_t getUncompDataSize(uint32_t size) {
    // The size of the uncompressed data is always a multiple of 16 bytes, to
    // guarantee alignment for proceeding data segments
    // We store the low order 4 bytes first, then the high order 2 uncompressed
    // bytes afterwards.
    // Both sections should be 16 byte aligned
    return 4 * roundUp(size, 4) + // low order 4 bytes
        2 * roundUp(size, 8);     // high order 2 bytes, starting at an aligned address
  }

  static __device__ size_t getNumCompSegments() {
    return 2;
  }

  static __device__ bool getIfNonCompSplit() {
    return true;
  }
};

inline size_t getWordSizeFromFloatType(FloatType ft) {
  switch (ft) {
    case FloatType::kFloat16:
    case FloatType::kBFloat16:
      return sizeof(uint16_t);
    case FloatType::kFloat32:
      return sizeof(uint32_t);
    case FloatType::kFloat64:
      return sizeof(uint64_t);
    default:
      CHECK(false);
      return 0;
  }
}

// Print out an array of integers or longs stored in GPU memory.
template<typename T>
__global__ void printarr(T *arr, int count) {
    // Only the master thread should print anything out
    if (blockIdx.y == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint32_t i = 0; i < count; ++i) {
            printf("%lu\t", arr[i]);
            if (i % 10 == 9) // line breaks every 10 elements
                printf("\n");
        }
        printf("\n");
    }
}

} // namespace dietgpu
