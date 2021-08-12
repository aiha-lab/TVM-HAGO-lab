#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#include <cuda_fp16.h>
__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}
#else

typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

#define TVM_FORCE_INLINE inline __attribute__((always_inline))
#define TVM_XINLINE TVM_FORCE_INLINE __device__ __host__
#define TVM_ALIGNED(x) __attribute__ ((aligned(x)))
#define TVM_HALF_OPERATOR(RTYPE, OP)                              \
  TVM_XINLINE RTYPE operator OP (half a, half b) {                \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (half a, T b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (T a, half b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }

#define TVM_HALF_ASSIGNOP(AOP, OP)                                \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const T& a) {                    \
    return *this = half(float(*this) OP float(a));                \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const volatile T& a) volatile {  \
    return *this = half(float(*this) OP float(a));                \
  }

class TVM_ALIGNED(2) half {
 public:
  uint16_t half_;

  static TVM_XINLINE half Binary(uint16_t value) {
    half res;
    res.half_ = value;
    return res;
  }

  TVM_XINLINE half() {}

  TVM_XINLINE half(const float& value) { constructor(value); }
  TVM_XINLINE explicit half(const double& value) { constructor(value); }
  TVM_XINLINE explicit half(const int8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const int32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const long long& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint64_t& value) { constructor(value); }

  TVM_XINLINE operator float() const {                          \
    return float(half2float(half_));                            \
  }                                                             \
  TVM_XINLINE operator float() const volatile {                 \
    return float(half2float(half_));                            \
  }


  TVM_HALF_ASSIGNOP(+=, +)
  TVM_HALF_ASSIGNOP(-=, -)
  TVM_HALF_ASSIGNOP(*=, *)
  TVM_HALF_ASSIGNOP(/=, /)

  TVM_XINLINE half operator+() {
    return *this;
  }

  TVM_XINLINE half operator-() {
    return half(-float(*this));
  }

  TVM_XINLINE half operator=(const half& a) {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) {
    return *this = half(a);
  }

  TVM_XINLINE half operator=(const half& a) volatile {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) volatile {
    return *this = half(a);
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static int const fp16FractionBits = 10;
  static int const fp32FractionBits = 23;
  static int32_t const fp32FractionMask = ~(~0u << fp32FractionBits);   // == 0x7fffff
  static int32_t const fp32HiddenBit = 1 << fp32FractionBits;   // == 0x800000
  static int const shift = fp32FractionBits - fp16FractionBits;   // == 13
  static int const shiftSign = 16;
  static int32_t const expAdjust = 127 - 15;   // exp32-127 = exp16-15, so exp16 = exp32 - (127-15)

  static int32_t const infN = 0x7F800000;   // flt32 infinity
  static int32_t const maxN = 0x477FFFFF;   // max flt32 that's a flt16 normal after >> by shift
  static int32_t const minN = 0x38800000;   // min flt16 normal as a flt32
  static int32_t const maxZ = 0x33000000;   // max fp32 number that's still rounded to zero in fp16
  static int32_t const signN = 0x80000000;  // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift;   // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

  static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
  static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400;  // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  TVM_XINLINE uint16_t float2half(const float& value) const {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  // Same as above routine, except for addition of volatile keyword
  TVM_XINLINE uint16_t float2half(
    const volatile float& value) const volatile {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  TVM_XINLINE float half2float(const uint16_t& value) const {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  TVM_XINLINE float half2float(
    const volatile uint16_t& value) const volatile {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  template<typename T>
  TVM_XINLINE void constructor(const T& value) {
    half_ = float2half(float(value));
  }
};

TVM_HALF_OPERATOR(half, +)
TVM_HALF_OPERATOR(half, -)
TVM_HALF_OPERATOR(half, *)
TVM_HALF_OPERATOR(half, /)
TVM_HALF_OPERATOR(bool, >)
TVM_HALF_OPERATOR(bool, <)
TVM_HALF_OPERATOR(bool, >=)
TVM_HALF_OPERATOR(bool, <=)

TVM_XINLINE half __float2half_rn(const float a) {
  return half(a);
}
#endif


// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
#include <sm_61_intrinsics.h>
#endif
#include <mma.h>
extern "C" __global__ void fused_cast_fixed_point_multiply_clip_cast_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) < 401408) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 1605632) {
        ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << ((long)0)) : ((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))])) * (long)1839678963) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_13255070459903635369__kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[32];
  __shared__ int pad_data_shared[832];
  __shared__ int placeholder_shared[512];
  #pragma unroll
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute[(oc_block_init)] = 0;
    compute[((oc_block_init + 8))] = 0;
    compute[((oc_block_init + 16))] = 0;
    compute[((oc_block_init + 24))] = 0;
    compute[((oc_block_init + 4))] = 0;
    compute[((oc_block_init + 12))] = 0;
    compute[((oc_block_init + 20))] = 0;
    compute[((oc_block_init + 28))] = 0;
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) < 416) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 16) + ((int)threadIdx.z)) < 60) {
          ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 448) + (((int)threadIdx.z) * 28)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((int)blockIdx.z) * 401408) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) / 52) * 50176)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) % 52) / 13) * 784)) + (((int)blockIdx.x) * 112)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) % 13) * 4)))))[0];
      }
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) < 256) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + ((int)threadIdx.z)) < 37) {
        if ((((((int)blockIdx.y) * 16) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 7)) + (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) >> 4)) < 128) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 448) + (((int)threadIdx.z) * 28)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 16384) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 7168)) + ((((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) >> 4) * 1024)) + ((((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) & 15) * 4)))))[0];
        }
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 15; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) < 416) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 16) + ((int)threadIdx.z)) < 60) {
            ((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 1664) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 448)) + (((int)threadIdx.z) * 28)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 401408) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) / 52) * 50176)) + (ic_chunk_outer_outer * 3136)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) % 52) / 13) * 784)) + (((int)blockIdx.x) * 112)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) % 13) * 4)) + 3136))))[0];
        }
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) < 256) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16) + ((int)threadIdx.z)) < 37) {
          if ((((((int)blockIdx.y) * 16) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 7)) + (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) >> 4)) < 128) {
              ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 1024) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 448)) + (((int)threadIdx.z) * 28)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 16384) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 7168)) + ((((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) >> 4) * 1024)) + (ic_chunk_outer_outer * 64)) + ((((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) & 15) * 4)) + 64))))[0];
          }
        }
      }
    }
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 4; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_block = 0; oc_block < 4; ++oc_block) {
        compute[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 1664) + ((((int)threadIdx.z) >> 3) * 208)) + (ic_chunk_inner * 52)) + (((int)threadIdx.x) * 8)))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 1024) + ((((int)threadIdx.z) & 7) * 64)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(oc_block)]);
        compute[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 1664) + ((((int)threadIdx.z) >> 3) * 208)) + (ic_chunk_inner * 52)) + (((int)threadIdx.x) * 8)) + 416))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 1024) + ((((int)threadIdx.z) & 7) * 64)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 8))]);
        compute[((oc_block + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 1664) + ((((int)threadIdx.z) >> 3) * 208)) + (ic_chunk_inner * 52)) + (((int)threadIdx.x) * 8)) + 832))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 1024) + ((((int)threadIdx.z) & 7) * 64)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 16))]);
        compute[((oc_block + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 1664) + ((((int)threadIdx.z) >> 3) * 208)) + (ic_chunk_inner * 52)) + (((int)threadIdx.x) * 8)) + 1248))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 1024) + ((((int)threadIdx.z) & 7) * 64)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 24))]);
        compute[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 1664) + ((((int)threadIdx.z) >> 3) * 208)) + (ic_chunk_inner * 52)) + (((int)threadIdx.x) * 8)))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 1024) + ((((int)threadIdx.z) & 7) * 64)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 512))))[0], compute[((oc_block + 4))]);
        compute[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 1664) + ((((int)threadIdx.z) >> 3) * 208)) + (ic_chunk_inner * 52)) + (((int)threadIdx.x) * 8)) + 416))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 1024) + ((((int)threadIdx.z) & 7) * 64)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 512))))[0], compute[((oc_block + 12))]);
        compute[((oc_block + 20))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 1664) + ((((int)threadIdx.z) >> 3) * 208)) + (ic_chunk_inner * 52)) + (((int)threadIdx.x) * 8)) + 832))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 1024) + ((((int)threadIdx.z) & 7) * 64)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 512))))[0], compute[((oc_block + 20))]);
        compute[((oc_block + 28))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 1664) + ((((int)threadIdx.z) >> 3) * 208)) + (ic_chunk_inner * 52)) + (((int)threadIdx.x) * 8)) + 1248))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 1024) + ((((int)threadIdx.z) & 7) * 64)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 512))))[0], compute[((oc_block + 28))]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 4; ++ic_chunk_inner1) {
    #pragma unroll
    for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
      compute[(oc_block1)] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((int)threadIdx.z) >> 3) * 208) + (ic_chunk_inner1 * 52)) + (((int)threadIdx.x) * 8)) + 1664))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.z) & 7) * 64) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 1024))))[0], compute[(oc_block1)]);
      compute[((oc_block1 + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((int)threadIdx.z) >> 3) * 208) + (ic_chunk_inner1 * 52)) + (((int)threadIdx.x) * 8)) + 2080))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.z) & 7) * 64) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 1024))))[0], compute[((oc_block1 + 8))]);
      compute[((oc_block1 + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((int)threadIdx.z) >> 3) * 208) + (ic_chunk_inner1 * 52)) + (((int)threadIdx.x) * 8)) + 2496))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.z) & 7) * 64) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 1024))))[0], compute[((oc_block1 + 16))]);
      compute[((oc_block1 + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((int)threadIdx.z) >> 3) * 208) + (ic_chunk_inner1 * 52)) + (((int)threadIdx.x) * 8)) + 2912))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.z) & 7) * 64) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 1024))))[0], compute[((oc_block1 + 24))]);
      compute[((oc_block1 + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((int)threadIdx.z) >> 3) * 208) + (ic_chunk_inner1 * 52)) + (((int)threadIdx.x) * 8)) + 1664))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.z) & 7) * 64) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 1536))))[0], compute[((oc_block1 + 4))]);
      compute[((oc_block1 + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((int)threadIdx.z) >> 3) * 208) + (ic_chunk_inner1 * 52)) + (((int)threadIdx.x) * 8)) + 2080))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.z) & 7) * 64) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 1536))))[0], compute[((oc_block1 + 12))]);
      compute[((oc_block1 + 20))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((int)threadIdx.z) >> 3) * 208) + (ic_chunk_inner1 * 52)) + (((int)threadIdx.x) * 8)) + 2496))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.z) & 7) * 64) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 1536))))[0], compute[((oc_block1 + 20))]);
      compute[((oc_block1 + 28))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((int)threadIdx.z) >> 3) * 208) + (ic_chunk_inner1 * 52)) + (((int)threadIdx.x) * 8)) + 2912))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.z) & 7) * 64) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 1536))))[0], compute[((oc_block1 + 28))]);
    }
  }
  #pragma unroll
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    ((int*)T_relu)[((((((((((int)blockIdx.z) * 200704) + ((((int)threadIdx.z) >> 3) * 25088)) + (((int)blockIdx.y) * 3136)) + ((((int)threadIdx.z) & 7) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((19 != 0) ? (((long)compute[(ax4)]) << ((long)19)) : ((long)compute[(ax4)])) * (long)1681959885) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + ((((int)threadIdx.z) & 7) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((19 != 0) ? (((long)compute[(ax4)]) << ((long)19)) : ((long)compute[(ax4)])) * (long)1681959885) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + ((((int)threadIdx.z) & 7) * 4)) + ax4))]))) * (long)1734818005) + ((long)1 << ((long)((1 + 31) - 1)))) >> ((long)(1 + 31)))) + ((int*)placeholder3)[((((((((((int)blockIdx.z) * 200704) + ((((int)threadIdx.z) >> 3) * 25088)) + (((int)blockIdx.y) * 3136)) + ((((int)threadIdx.z) & 7) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4))])), (int)(0));
    ((int*)T_relu)[(((((((((((int)blockIdx.z) * 200704) + ((((int)threadIdx.z) >> 3) * 25088)) + (((int)blockIdx.y) * 3136)) + ((((int)threadIdx.z) & 7) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 50176))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)19)) : ((long)compute[((ax4 + 8))])) * (long)1681959885) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + ((((int)threadIdx.z) & 7) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)19)) : ((long)compute[((ax4 + 8))])) * (long)1681959885) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + ((((int)threadIdx.z) & 7) * 4)) + ax4))]))) * (long)1734818005) + ((long)1 << ((long)((1 + 31) - 1)))) >> ((long)(1 + 31)))) + ((int*)placeholder3)[(((((((((((int)blockIdx.z) * 200704) + ((((int)threadIdx.z) >> 3) * 25088)) + (((int)blockIdx.y) * 3136)) + ((((int)threadIdx.z) & 7) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 50176))])), (int)(0));
    ((int*)T_relu)[(((((((((((int)blockIdx.z) * 200704) + ((((int)threadIdx.z) >> 3) * 25088)) + (((int)blockIdx.y) * 3136)) + ((((int)threadIdx.z) & 7) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 100352))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)19)) : ((long)compute[((ax4 + 16))])) * (long)1681959885) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + ((((int)threadIdx.z) & 7) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)19)) : ((long)compute[((ax4 + 16))])) * (long)1681959885) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + ((((int)threadIdx.z) & 7) * 4)) + ax4))]))) * (long)1734818005) + ((long)1 << ((long)((1 + 31) - 1)))) >> ((long)(1 + 31)))) + ((int*)placeholder3)[(((((((((((int)blockIdx.z) * 200704) + ((((int)threadIdx.z) >> 3) * 25088)) + (((int)blockIdx.y) * 3136)) + ((((int)threadIdx.z) & 7) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 100352))])), (int)(0));
    ((int*)T_relu)[(((((((((((int)blockIdx.z) * 200704) + ((((int)threadIdx.z) >> 3) * 25088)) + (((int)blockIdx.y) * 3136)) + ((((int)threadIdx.z) & 7) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 150528))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)19)) : ((long)compute[((ax4 + 24))])) * (long)1681959885) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + ((((int)threadIdx.z) & 7) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)19)) : ((long)compute[((ax4 + 24))])) * (long)1681959885) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + ((((int)threadIdx.z) & 7) * 4)) + ax4))]))) * (long)1734818005) + ((long)1 << ((long)((1 + 31) - 1)))) >> ((long)(1 + 31)))) + ((int*)placeholder3)[(((((((((((int)blockIdx.z) * 200704) + ((((int)threadIdx.z) >> 3) * 25088)) + (((int)blockIdx.y) * 3136)) + ((((int)threadIdx.z) & 7) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 150528))])), (int)(0));
    ((int*)T_relu)[(((((((((((int)blockIdx.z) * 200704) + ((((int)threadIdx.z) >> 3) * 25088)) + (((int)blockIdx.y) * 3136)) + ((((int)threadIdx.z) & 7) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 1568))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)19)) : ((long)compute[((ax4 + 4))])) * (long)1681959885) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + ((((int)threadIdx.z) & 7) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)19)) : ((long)compute[((ax4 + 4))])) * (long)1681959885) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + ((((int)threadIdx.z) & 7) * 4)) + ax4) + 32))]))) * (long)1734818005) + ((long)1 << ((long)((1 + 31) - 1)))) >> ((long)(1 + 31)))) + ((int*)placeholder3)[(((((((((((int)blockIdx.z) * 200704) + ((((int)threadIdx.z) >> 3) * 25088)) + (((int)blockIdx.y) * 3136)) + ((((int)threadIdx.z) & 7) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 1568))])), (int)(0));
    ((int*)T_relu)[(((((((((((int)blockIdx.z) * 200704) + ((((int)threadIdx.z) >> 3) * 25088)) + (((int)blockIdx.y) * 3136)) + ((((int)threadIdx.z) & 7) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 51744))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)19)) : ((long)compute[((ax4 + 12))])) * (long)1681959885) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + ((((int)threadIdx.z) & 7) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)19)) : ((long)compute[((ax4 + 12))])) * (long)1681959885) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + ((((int)threadIdx.z) & 7) * 4)) + ax4) + 32))]))) * (long)1734818005) + ((long)1 << ((long)((1 + 31) - 1)))) >> ((long)(1 + 31)))) + ((int*)placeholder3)[(((((((((((int)blockIdx.z) * 200704) + ((((int)threadIdx.z) >> 3) * 25088)) + (((int)blockIdx.y) * 3136)) + ((((int)threadIdx.z) & 7) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 51744))])), (int)(0));
    ((int*)T_relu)[(((((((((((int)blockIdx.z) * 200704) + ((((int)threadIdx.z) >> 3) * 25088)) + (((int)blockIdx.y) * 3136)) + ((((int)threadIdx.z) & 7) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 101920))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)19)) : ((long)compute[((ax4 + 20))])) * (long)1681959885) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + ((((int)threadIdx.z) & 7) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)19)) : ((long)compute[((ax4 + 20))])) * (long)1681959885) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + ((((int)threadIdx.z) & 7) * 4)) + ax4) + 32))]))) * (long)1734818005) + ((long)1 << ((long)((1 + 31) - 1)))) >> ((long)(1 + 31)))) + ((int*)placeholder3)[(((((((((((int)blockIdx.z) * 200704) + ((((int)threadIdx.z) >> 3) * 25088)) + (((int)blockIdx.y) * 3136)) + ((((int)threadIdx.z) & 7) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 101920))])), (int)(0));
    ((int*)T_relu)[(((((((((((int)blockIdx.z) * 200704) + ((((int)threadIdx.z) >> 3) * 25088)) + (((int)blockIdx.y) * 3136)) + ((((int)threadIdx.z) & 7) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 152096))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 28))]) << ((long)19)) : ((long)compute[((ax4 + 28))])) * (long)1681959885) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + ((((int)threadIdx.z) & 7) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 28))]) << ((long)19)) : ((long)compute[((ax4 + 28))])) * (long)1681959885) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + ((((int)threadIdx.z) & 7) * 4)) + ax4) + 32))]))) * (long)1734818005) + ((long)1 << ((long)((1 + 31) - 1)))) >> ((long)(1 + 31)))) + ((int*)placeholder3)[(((((((((((int)blockIdx.z) * 200704) + ((((int)threadIdx.z) >> 3) * 25088)) + (((int)blockIdx.y) * 3136)) + ((((int)threadIdx.z) & 7) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 152096))])), (int)(0));
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_nn_conv2d_cast_fixed_point_multiply_add_cast_fix_1476761249106760241__1_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[56];
  __shared__ int pad_data_shared[96];
  __shared__ int placeholder_shared[288];
  #pragma unroll
  for (int oc_chunk_init = 0; oc_chunk_init < 2; ++oc_chunk_init) {
    #pragma unroll
    for (int oh_init = 0; oh_init < 7; ++oh_init) {
      #pragma unroll
      for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
        compute[((((oc_chunk_init * 28) + (oh_init * 4)) + oc_block_init))] = 0;
      }
    }
  }
  for (int ic_chunk_outer = 0; ic_chunk_outer < 32; ++ic_chunk_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)pad_data_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 14) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 6))) && ((((((int)blockIdx.x) / 7) * 14) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 6)) < 29)) && (1 <= (((((int)blockIdx.x) % 7) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6)))) && ((((((int)blockIdx.x) % 7) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6)) < 29)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 100352) + (ic_chunk_outer * 3136)) + ((((int)blockIdx.x) / 7) * 1568)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 6) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6) * 4)) - 116))))[0] : (int)(int)0);
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 9; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 36864) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + (((int)threadIdx.z) * 2)) + ((int)threadIdx.y)) / 9) * 4608)) + (ic_chunk_outer * 144)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + (((int)threadIdx.z) * 2)) + ((int)threadIdx.y)) % 9) * 16)) + (((int)threadIdx.x) * 4)))))[0];
    }
    __syncthreads();
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
        #pragma unroll
        for (int oc_chunk = 0; oc_chunk < 2; ++oc_chunk) {
          #pragma unroll
          for (int oh = 0; oh < 7; ++oh) {
            #pragma unroll
            for (int oc_block = 0; oc_block < 4; ++oc_block) {
              compute[((((oc_chunk * 28) + (oh * 4)) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((int)threadIdx.y) * 168) + (oh * 24)) + (kh_inner * 24)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.z) * 288) + (oc_chunk * 144)) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[((((oc_chunk * 28) + (oh * 4)) + oc_block))]);
            }
          }
        }
      }
    }
  }
  #pragma unroll
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 2; ++ax1_inner_inner_inner) {
    #pragma unroll
    for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 7; ++ax2_inner_inner_inner) {
      #pragma unroll
      for (int ax4 = 0; ax4 < 4; ++ax4) {
        ((int*)T_relu)[(((((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 25088)) + (((int)threadIdx.z) * 6272)) + (ax1_inner_inner_inner * 3136)) + ((((int)blockIdx.x) / 7) * 1568)) + (((int)threadIdx.y) * 784)) + (ax2_inner_inner_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) * 4)) + ax4))] = max((int)((((int)(((((0 != 0) ? (((long)((int*)placeholder2)[(((((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 25088)) + (((int)threadIdx.z) * 6272)) + (ax1_inner_inner_inner * 3136)) + ((((int)blockIdx.x) / 7) * 1568)) + (((int)threadIdx.y) * 784)) + (ax2_inner_inner_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) * 4)) + ax4))]) << ((long)0)) : ((long)((int*)placeholder2)[(((((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 25088)) + (((int)threadIdx.z) * 6272)) + (ax1_inner_inner_inner * 3136)) + ((((int)blockIdx.x) / 7) * 1568)) + (((int)threadIdx.y) * 784)) + (ax2_inner_inner_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) * 4)) + ax4))])) * (long)1730082555) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 28) + (ax2_inner_inner_inner * 4)) + ax4))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 28) + (ax2_inner_inner_inner * 4)) + ax4))])) * (long)1219446002) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 32) + (((int)threadIdx.z) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 28) + (ax2_inner_inner_inner * 4)) + ax4))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 28) + (ax2_inner_inner_inner * 4)) + ax4))])) * (long)1219446002) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 32) + (((int)threadIdx.z) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)2074804792) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
      }
    }
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_nn_conv2d_cast_fixed_point_multiply_add_cast_fix_1476761249106760241__kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[28];
  __shared__ int pad_data_shared[1024];
  __shared__ int placeholder_shared[1152];
  #pragma unroll
  for (int oh_init = 0; oh_init < 7; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oh_init * 4) + oc_block_init))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
      ((int*)((signed char*)pad_data_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) & 15)) && ((((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) & 15) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + ((int)threadIdx.x)))) && (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) < 15)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 200704) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 50176)) + ((((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) >> 4) * 784)) + ((((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) & 15) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 4)) - 60))))[0] : (int)(int)0);
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) < 144) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 576) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 4) + ((int)threadIdx.z)) < 18) {
            ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 73728) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) / 18) * 9216)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) % 18) * 16)) + (((int)threadIdx.x) * 4)))))[0];
        }
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 31; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
        ((int*)((signed char*)pad_data_shared + ((((((((ic_chunk_outer_outer + 1) & 1) * 2048) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 512)) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) & 15)) && ((((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) & 15) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + ((int)threadIdx.x)))) && (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) < 15)) ? (int)((int*)((signed char*)placeholder + (((((((((((int)blockIdx.z) * 200704) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 50176)) + (ic_chunk_outer_outer * 1568)) + ((((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) >> 4) * 784)) + ((((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) & 15) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 4)) + 1508))))[0] : (int)(int)0);
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) < 144) {
        if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 576) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 4) + ((int)threadIdx.z)) < 18) {
              ((int*)((signed char*)placeholder_shared + ((((((((ic_chunk_outer_outer + 1) & 1) * 2304) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 512)) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 73728) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) / 18) * 9216)) + (ic_chunk_outer_outer * 288)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) % 18) * 16)) + (((int)threadIdx.x) * 4)) + 288))))[0];
          }
        }
      }
    }
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 2; ++ic_chunk_inner) {
      #pragma unroll
      for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
        #pragma unroll
        for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
          #pragma unroll
          for (int oh = 0; oh < 7; ++oh) {
            #pragma unroll
            for (int oc_block = 0; oc_block < 4; ++oc_block) {
              compute[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((ic_chunk_outer_outer & 1) * 2048) + (((int)threadIdx.z) * 512)) + (ic_chunk_inner * 256)) + ((((int)threadIdx.x) >> 1) * 112)) + (oh * 16)) + (kh_inner * 16)) + (kw_inner * 4)) + ((((int)threadIdx.x) & 1) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((ic_chunk_outer_outer & 1) * 2304) + (((int)threadIdx.y) * 288)) + (ic_chunk_inner * 144)) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((oh * 4) + oc_block))]);
            }
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 2; ++ic_chunk_inner1) {
    #pragma unroll
    for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
      #pragma unroll
      for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
        #pragma unroll
        for (int oh1 = 0; oh1 < 7; ++oh1) {
          #pragma unroll
          for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
            compute[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((((((int)threadIdx.z) * 512) + (ic_chunk_inner1 * 256)) + ((((int)threadIdx.x) >> 1) * 112)) + (oh1 * 16)) + (kh_inner1 * 16)) + (kw_inner1 * 4)) + ((((int)threadIdx.x) & 1) * 4)) + 2048))))[0], ((int*)((signed char*)placeholder_shared + (((((((((int)threadIdx.y) * 288) + (ic_chunk_inner1 * 144)) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)) + 2304))))[0], compute[(((oh1 * 4) + oc_block1))]);
          }
        }
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 7; ++ax2_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((int*)T_relu)[((((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 784)) + ((((int)threadIdx.x) >> 1) * 392)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + ax4))] = max((int)((((int)(((((0 != 0) ? (((long)((int*)placeholder2)[((((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 784)) + ((((int)threadIdx.x) >> 1) * 392)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + ax4))]) << ((long)0)) : ((long)((int*)placeholder2)[((((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 784)) + ((((int)threadIdx.x) >> 1) * 392)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + ax4))])) * (long)1940156806) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((1 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1804115175) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1804115175) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1118691894) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__7_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[28];
  __shared__ int placeholder_shared[1152];
  __shared__ int pad_data_shared[162];
  #pragma unroll
  for (int oh_init = 0; oh_init < 7; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oh_init * 4) + oc_block_init))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 576) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 83) {
          ((int*)((signed char*)placeholder_shared + (((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 12) * 48) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.z) * 28)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((int)blockIdx.y) * 294912) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 36) * 18432)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 36) / 12) * 48)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.z) * 28)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0];
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 127; ++ic_chunk_outer_outer) {
    __syncthreads();
    if ((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 162) {
      if (((((int)threadIdx.z) * 16) + ((int)threadIdx.y)) < 24) {
          ((int*)((signed char*)pad_data_shared + ((((((int)threadIdx.z) * 448) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = (((((9 <= ((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81)) && (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81) < 72)) && (1 <= ((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9))) && (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) < 8)) ? (int)((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.z) * 50176) + (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 81) * 25088)) + (ic_chunk_outer_outer * 196)) + ((((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81) / 9) * 28)) + (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) * 4)) - 32))))[0] : (int)(int)0);
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 576) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 83) {
            ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 2304) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 12) * 48)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.z) * 28)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.y) * 294912) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 36) * 18432)) + (ic_chunk_outer_outer * 144)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 36) / 12) * 48)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.z) * 28)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)) + 144))))[0];
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
        #pragma unroll
        for (int oh = 0; oh < 7; ++oh) {
          #pragma unroll
          for (int oc_block = 0; oc_block < 4; ++oc_block) {
            compute[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((int)threadIdx.z) * 324) + (oh * 36)) + (kh_inner * 36)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 2304) + (((int)threadIdx.y) * 144)) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((oh * 4) + oc_block))]);
          }
        }
      }
    }
  }
  __syncthreads();
  if ((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 162) {
    if (((((int)threadIdx.z) * 16) + ((int)threadIdx.y)) < 24) {
        ((int*)((signed char*)pad_data_shared + ((((((int)threadIdx.z) * 448) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = (((((9 <= ((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81)) && (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81) < 72)) && (1 <= ((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9))) && (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) < 8)) ? (int)((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.z) * 50176) + (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 81) * 25088)) + ((((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81) / 9) * 28)) + (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) * 4)) + 24860))))[0] : (int)(int)0);
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
    #pragma unroll
    for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
      #pragma unroll
      for (int oh1 = 0; oh1 < 7; ++oh1) {
        #pragma unroll
        for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
          compute[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((int)threadIdx.z) * 324) + (oh1 * 36)) + (kh_inner1 * 36)) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)) + 2304))))[0], compute[(((oh1 * 4) + oc_block1))]);
        }
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 7; ++ax2_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 3136)) + (((int)threadIdx.y) * 196)) + (ax2_inner_inner_inner * 28)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1933257701) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1933257701) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1649746970) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_clip_cast_3_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 25; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) < 1605632) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 6422528) {
        ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << ((long)0)) : ((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))])) * (long)1972011630) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
      }
    }
  }
}

extern "C" __global__ void fused_cast_add_right_shift_clip_cast_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 25; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) < 1605632) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 6422528) {
        ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)min((int)(((((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] + 8388608) >> 24)), (int)(127)));
      }
    }
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_clip_cast_5_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) < 401408) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 1605632) {
        ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << ((long)0)) : ((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))])) * (long)1984087727) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
      }
    }
  }
}

extern "C" __global__ void fused_nn_max_pool2d_1_kernel0(void* __restrict__ placeholder, void* __restrict__ tensor) {
  int tensor_local[1];
  tensor_local[(0)] = -2147483648;
  for (int rv = 0; rv < 3; ++rv) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      tensor_local[(0)] = max((int)(tensor_local[(0)]), (int)((((1 <= ((((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) % 3136) / 56) * 2) + rv)) && (1 <= (((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) % 56) * 2) + rv1))) ? (int)((int*)placeholder)[((((((((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 56) * 896) + (rv * 448)) + ((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) % 56) * 8)) + (rv1 * 4)) + (((int)threadIdx.x) & 3)) - 452))] : (int)-2147483648)));
    }
  }
  ((int*)tensor)[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = tensor_local[(0)];
}

extern "C" __global__ void fused_nn_dense_add_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_add, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 32, 8, 16, float> T_dense_wmma_accumulator[1];
  __shared__ half compute_shared[512];
  __shared__ half compute_shared1[128];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 32, 8, 16, half, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 32, 8, 16, half, nvcuda::wmma::col_major> compute_shared_wmma_matrix_b[1];
  (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[0], 0.000000e+00f);
  for (int k_outer_outer = 0; k_outer_outer < 32; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 16; ++ax0_ax1_fused_outer_outer_outer_outer) {
      compute_shared[(((ax0_ax1_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = ((half)((float*)placeholder)[(((((ax0_ax1_fused_outer_outer_outer_outer * 1024) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15)))]);
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0; ax0_ax1_fused_outer_outer_outer_outer1 < 4; ++ax0_ax1_fused_outer_outer_outer_outer1) {
      compute_shared1[(((ax0_ax1_fused_outer_outer_outer_outer1 * 32) + ((int)threadIdx.x)))] = ((half)((float*)placeholder1)[((((((((int)blockIdx.y) * 4096) + (ax0_ax1_fused_outer_outer_outer_outer1 * 1024)) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15)))]);
    }
    __syncthreads();
    (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((half *)compute_shared + 0), 16);
    (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_b[0], ((half *)compute_shared1 + 0), 16);
    (void)nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[0], compute_shared_wmma_matrix_a[0], compute_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
  }
  __syncthreads();
  (void)nvcuda::wmma::store_matrix_sync(((float *)compute_shared + 0), T_dense_wmma_accumulator[0], 8, nvcuda::wmma::mem_row_major);
  __syncthreads();
  for (int ax0_inner_ax1_inner_fused_outer_outer_outer_outer = 0; ax0_inner_ax1_inner_fused_outer_outer_outer_outer < 8; ++ax0_inner_ax1_inner_fused_outer_outer_outer_outer) {
    ((float*)T_add)[(((((ax0_inner_ax1_inner_fused_outer_outer_outer_outer * 4000) + ((((int)threadIdx.x) >> 3) * 1000)) + (((int)blockIdx.y) * 8)) + (((int)threadIdx.x) & 7)))] = (((float*)compute_shared)[(((ax0_inner_ax1_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] + ((float*)placeholder2)[(((((int)blockIdx.y) * 8) + (((int)threadIdx.x) & 7)))]);
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_6343854372805914660__kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ compute, void* __restrict__ placeholder2) {
  int compute1[64];
  __shared__ int placeholder_shared[896];
  __shared__ int pad_data_shared[296];
  #pragma unroll
  for (int n_init = 0; n_init < 4; ++n_init) {
    #pragma unroll
    for (int oc_chunk_init = 0; oc_chunk_init < 4; ++oc_chunk_init) {
      #pragma unroll
      for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
        compute1[((((n_init * 16) + (oc_chunk_init * 4)) + oc_block_init))] = 0;
      }
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.x) >> 2)) < 112) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.x)) < 448) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.z)) < 28) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.x) >> 2)) / 7) * 784) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.x) >> 2)) % 7) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
        }
      }
    }
  }
  for (int kh_outer_outer = 0; kh_outer_outer < 6; ++kh_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.x)) < 296) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.z)) < 19) {
            ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.x) * 4)))))[0] = (((((3 <= (((((int)blockIdx.x) / 7) * 2) + kh_outer_outer)) && ((((((int)blockIdx.x) / 7) * 2) + kh_outer_outer) < 227)) && (3 <= (((((int)blockIdx.x) % 7) * 32) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.x)) % 37)))) && ((((((int)blockIdx.x) % 7) * 32) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.x)) % 37)) < 227)) ? (int)((int*)((signed char*)placeholder1 + ((((((((((int)blockIdx.z) * 1605632) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.x)) / 37) * 200704)) + ((((int)blockIdx.x) / 7) * 1792)) + (kh_outer_outer * 896)) + ((((int)blockIdx.x) % 7) * 128)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.x)) % 37) * 4)) - 2700))))[0] : (int)(int)0);
        }
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.x) >> 2)) < 112) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.x)) < 448) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.z)) < 28) {
              ((int*)((signed char*)placeholder_shared + (((((((kh_outer_outer + 1) & 1) * 1792) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 512)) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.x) >> 2)) / 7) * 784) + (kh_outer_outer * 112)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.x) >> 2)) % 7) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 112))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 7; ++kw_inner) {
      #pragma unroll
      for (int n = 0; n < 4; ++n) {
        #pragma unroll
        for (int oc_chunk = 0; oc_chunk < 4; ++oc_chunk) {
          #pragma unroll
          for (int oc_block = 0; oc_block < 4; ++oc_block) {
            compute1[((((n * 16) + (oc_chunk * 4)) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((int)threadIdx.z) >> 2) * 592) + (n * 148)) + (((int)threadIdx.x) * 8)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((kh_outer_outer & 1) * 1792) + ((((int)threadIdx.z) & 3) * 448)) + (oc_chunk * 112)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute1[((((n * 16) + (oc_chunk * 4)) + oc_block))]);
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.x)) < 296) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.z)) < 19) {
          ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 512) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.x) * 4)))))[0] = ((((((int)blockIdx.x) < 777) && (3 <= (((((int)blockIdx.x) % 7) * 32) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.x)) % 37)))) && ((((((int)blockIdx.x) % 7) * 32) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.x)) % 37)) < 227)) ? (int)((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.z) * 1605632) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.x)) / 37) * 200704)) + ((((int)blockIdx.x) / 7) * 1792)) + ((((int)blockIdx.x) % 7) * 128)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.x)) % 37) * 4)) + 2676))))[0] : (int)(int)0);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 7; ++kw_inner1) {
    #pragma unroll
    for (int n1 = 0; n1 < 4; ++n1) {
      #pragma unroll
      for (int oc_chunk1 = 0; oc_chunk1 < 4; ++oc_chunk1) {
        #pragma unroll
        for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
          compute1[((((n1 * 16) + (oc_chunk1 * 4)) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((int)threadIdx.z) >> 2) * 592) + (n1 * 148)) + (((int)threadIdx.x) * 8)) + (kw_inner1 * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.z) & 3) * 448) + (oc_chunk1 * 112)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute1[((((n1 * 16) + (oc_chunk1 * 4)) + oc_block1))]);
        }
      }
    }
  }
  #pragma unroll
  for (int i0_inner_inner_inner_inner = 0; i0_inner_inner_inner_inner < 4; ++i0_inner_inner_inner_inner) {
    #pragma unroll
    for (int i1_inner_inner_inner = 0; i1_inner_inner_inner < 4; ++i1_inner_inner_inner) {
      #pragma unroll
      for (int i4 = 0; i4 < 4; ++i4) {
        ((int*)compute)[(((((((((((int)blockIdx.z) * 6422528) + ((((int)threadIdx.z) >> 2) * 3211264)) + (i0_inner_inner_inner_inner * 802816)) + ((((int)threadIdx.z) & 3) * 200704)) + (i1_inner_inner_inner * 50176)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 4)) + i4))] = ((int)(((((0 != 0) ? (((long)max((int)((((int)(((((15 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 16) + (i1_inner_inner_inner * 4)) + i4))]) << ((long)15)) : ((long)compute1[((((i0_inner_inner_inner_inner * 16) + (i1_inner_inner_inner * 4)) + i4))])) * (long)1154272231) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)threadIdx.z) & 3) * 16) + (i1_inner_inner_inner * 4)) + i4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((15 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 16) + (i1_inner_inner_inner * 4)) + i4))]) << ((long)15)) : ((long)compute1[((((i0_inner_inner_inner_inner * 16) + (i1_inner_inner_inner * 4)) + i4))])) * (long)1154272231) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)threadIdx.z) & 3) * 16) + (i1_inner_inner_inner * 4)) + i4))])), (int)(0)))) * (long)2107679426) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
      }
    }
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_nn_conv2d_cast_fixed_point_multiply_add_cast_fix_1476761249106760241__2_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[32];
  __shared__ int pad_data_shared[200];
  __shared__ int placeholder_shared[576];
  #pragma unroll
  for (int oh_init = 0; oh_init < 8; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oh_init * 4) + oc_block_init))] = 0;
    }
  }
  if (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) < 100) {
    if (((int)threadIdx.y) < 13) {
        ((int*)((signed char*)pad_data_shared + (((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10))) && ((((((int)blockIdx.x) / 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10)) < 57)) && (1 <= (((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)))) && ((((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)) < 57)) ? (int)((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 200704) + ((((int)blockIdx.x) / 7) * 1792)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10) * 4)) - 228))))[0] : (int)(int)0);
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 15; ++ic_chunk_outer_outer) {
    __syncthreads();
    if (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) < 100) {
      if (((int)threadIdx.y) < 13) {
          ((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer + 1) & 1) * 400) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10))) && ((((((int)blockIdx.x) / 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10)) < 57)) && (1 <= (((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)))) && ((((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)) < 57)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 200704) + (ic_chunk_outer_outer * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10) * 4)) + 12316))))[0] : (int)(int)0);
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) < 144) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) < 576) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + ((int)threadIdx.y)) < 72) {
              ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) / 9) * 2304) + (ic_chunk_outer_outer * 144)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) % 9) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
        #pragma unroll
        for (int oh = 0; oh < 8; ++oh) {
          #pragma unroll
          for (int oc_block = 0; oc_block < 4; ++oc_block) {
            compute[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 400) + (oh * 40)) + (kh_inner * 40)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((oh * 4) + oc_block))]);
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) < 144) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) < 576) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16) + ((int)threadIdx.y)) < 72) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 512) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) / 9) * 2304) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) % 9) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 2160))))[0];
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
    #pragma unroll
    for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
      #pragma unroll
      for (int oh1 = 0; oh1 < 8; ++oh1) {
        #pragma unroll
        for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
          compute[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((oh1 * 40) + (kh_inner1 * 40)) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 400))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[(((oh1 * 4) + oc_block1))]);
        }
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 8; ++ax2_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((int*)T_relu)[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (ax2_inner_inner_inner * 224)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4))] = max((int)((((int)(((((1 != 0) ? (((long)((int*)placeholder2)[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (ax2_inner_inner_inner * 224)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4))]) << ((long)1)) : ((long)((int*)placeholder2)[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (ax2_inner_inner_inner * 224)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4))])) * (long)1972011630) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)2071811799) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.y) * 4) + ax4))])) << ((long)0)) : ((long)(((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)2071811799) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.y) * 4) + ax4))]))) * (long)2128621749) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    }
  }
}

extern "C" __global__ void fused_nn_global_avg_pool2d_1_kernel0(void* __restrict__ placeholder, void* __restrict__ tensor) {
  float tensor1[4];
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    tensor1[(ax4)] = 0.000000e+00f;
    for (int rv0 = 0; rv0 < 7; ++rv0) {
      for (int rv1 = 0; rv1 < 7; ++rv1) {
        tensor1[(ax4)] = (tensor1[(ax4)] + ((float*)placeholder)[((((((((((int)blockIdx.y) * 200704) + (((int)threadIdx.y) * 25088)) + (((int)blockIdx.x) * 1568)) + (((int)threadIdx.x) * 196)) + (rv0 * 28)) + (rv1 * 4)) + ax4))]);
      }
    }
  }
  for (int ax41 = 0; ax41 < 4; ++ax41) {
    ((float*)tensor)[((((((((int)blockIdx.y) * 4096) + (((int)threadIdx.y) * 512)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) * 4)) + ax41))] = (tensor1[(ax41)] * 2.040816e-02f);
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_clip_cast_4_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 13; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) < 802816) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 3211264) {
        ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << ((long)0)) : ((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))])) * (long)2014218145) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[32];
  __shared__ int pad_data_shared[200];
  __shared__ int placeholder_shared[576];
  #pragma unroll
  for (int oh_init = 0; oh_init < 8; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oh_init * 4) + oc_block_init))] = 0;
    }
  }
  if (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) < 100) {
    if (((int)threadIdx.y) < 13) {
        ((int*)((signed char*)pad_data_shared + (((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10))) && ((((((int)blockIdx.x) / 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10)) < 57)) && (1 <= (((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)))) && ((((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)) < 57)) ? (int)((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 200704) + ((((int)blockIdx.x) / 7) * 1792)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10) * 4)) - 228))))[0] : (int)(int)0);
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 15; ++ic_chunk_outer_outer) {
    __syncthreads();
    if (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) < 100) {
      if (((int)threadIdx.y) < 13) {
          ((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer + 1) & 1) * 400) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10))) && ((((((int)blockIdx.x) / 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10)) < 57)) && (1 <= (((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)))) && ((((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)) < 57)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 200704) + (ic_chunk_outer_outer * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10) * 4)) + 12316))))[0] : (int)(int)0);
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) < 144) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) < 576) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + ((int)threadIdx.y)) < 72) {
              ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) / 9) * 2304) + (ic_chunk_outer_outer * 144)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) % 9) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
        #pragma unroll
        for (int oh = 0; oh < 8; ++oh) {
          #pragma unroll
          for (int oc_block = 0; oc_block < 4; ++oc_block) {
            compute[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 400) + (oh * 40)) + (kh_inner * 40)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((oh * 4) + oc_block))]);
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) < 144) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) < 576) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16) + ((int)threadIdx.y)) < 72) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 512) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) / 9) * 2304) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) % 9) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 2160))))[0];
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
    #pragma unroll
    for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
      #pragma unroll
      for (int oh1 = 0; oh1 < 8; ++oh1) {
        #pragma unroll
        for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
          compute[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((oh1 * 40) + (kh_inner1 * 40)) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 400))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[(((oh1 * 4) + oc_block1))]);
        }
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 8; ++ax2_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (ax2_inner_inner_inner * 224)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1090324072) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.y) * 4) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1090324072) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.y) * 4) + ax4))])), (int)(0)))) * (long)1346499769) + ((long)1 << ((long)((22 + 31) - 1)))) >> ((long)(22 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__4_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[32];
  __shared__ int pad_data_shared[116];
  __shared__ int placeholder_shared[384];
  #pragma unroll
  for (int n_init = 0; n_init < 4; ++n_init) {
    #pragma unroll
    for (int oc_chunk_init = 0; oc_chunk_init < 2; ++oc_chunk_init) {
      #pragma unroll
      for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
        compute[((((n_init * 8) + (oc_chunk_init * 4)) + oc_block_init))] = 0;
      }
    }
  }
  for (int ic_chunk_outer = 0; ic_chunk_outer < 32; ++ic_chunk_outer) {
    #pragma unroll
    for (int kh_outer = 0; kh_outer < 3; ++kh_outer) {
      __syncthreads();
      if (((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) < 116) {
        if (((int)threadIdx.z) < 9) {
            ((int*)((signed char*)pad_data_shared + (((((int)threadIdx.z) * 56) + (((int)threadIdx.x) * 4)))))[0] = (((1 <= ((((int)blockIdx.x) * 2) + kh_outer)) && (1 <= (((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) % 29))) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 401408) + ((((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) / 29) * 100352)) + (ic_chunk_outer * 3136)) + (((int)blockIdx.x) * 224)) + (kh_outer * 112)) + ((((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) % 29) * 4)) - 116))))[0] : (int)(int)0);
        }
      }
      #pragma unroll
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 14)) + ((int)threadIdx.x)) < 384) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + ((int)threadIdx.z)) < 28) {
              ((int*)((signed char*)placeholder_shared + ((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 14)) + ((int)threadIdx.x)) / 12) * 48) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 147456) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 14)) + ((int)threadIdx.x)) / 12) * 4608)) + (ic_chunk_outer * 144)) + (kh_outer * 48)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) & 3) * 4)))))[0];
          }
        }
      }
      __syncthreads();
      #pragma unroll
      for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
          #pragma unroll
          for (int oc_chunk = 0; oc_chunk < 2; ++oc_chunk) {
            #pragma unroll
            for (int oc_block = 0; oc_block < 4; ++oc_block) {
              compute[((((n * 8) + (oc_chunk * 4)) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n * 116) + (((int)threadIdx.x) * 8)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 96) + (oc_chunk * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[((((n * 8) + (oc_chunk * 4)) + oc_block))]);
            }
          }
        }
      }
    }
  }
  #pragma unroll
  for (int ax0_inner_inner_inner_inner = 0; ax0_inner_inner_inner_inner < 4; ++ax0_inner_inner_inner_inner) {
    #pragma unroll
    for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 2; ++ax1_inner_inner_inner) {
      #pragma unroll
      for (int ax4 = 0; ax4 < 4; ++ax4) {
        ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 200704) + (ax0_inner_inner_inner_inner * 50176)) + (((int)blockIdx.y) * 25088)) + (((int)threadIdx.z) * 1568)) + (ax1_inner_inner_inner * 784)) + (((int)blockIdx.x) * 56)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 8) + (ax1_inner_inner_inner * 4)) + ax4))]) << ((long)16)) : ((long)compute[((((ax0_inner_inner_inner_inner * 8) + (ax1_inner_inner_inner * 4)) + ax4))])) * (long)1529653425) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 8) + (ax1_inner_inner_inner * 4)) + ax4))]) << ((long)16)) : ((long)compute[((((ax0_inner_inner_inner_inner * 8) + (ax1_inner_inner_inner * 4)) + ax4))])) * (long)1529653425) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])), (int)(0)))) * (long)1252240138) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
      }
    }
  }
}

extern "C" __global__ void fused_divide_add_round_cast_clip_cast_nn_pad_layout_transform_kernel0(void* __restrict__ T_layout_trans, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 25; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) < 1605632) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 6422528) {
        ((signed char*)T_layout_trans)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = (((((int)threadIdx.x) & 3) < 3) ? (signed char)((signed char)max((int)(min((int)(((int)roundf((((float*)placeholder)[((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) / 50176) * 150528) + ((((int)threadIdx.x) & 3) * 50176)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) % 50176)))] * 4.861612e+01f)))), (int)(127))), (int)(-128))) : (signed char)(signed char)0);
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__1_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[32];
  __shared__ int pad_data_shared[200];
  __shared__ int placeholder_shared[576];
  #pragma unroll
  for (int oh_init = 0; oh_init < 8; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oh_init * 4) + oc_block_init))] = 0;
    }
  }
  if (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) < 100) {
    if (((int)threadIdx.y) < 13) {
        ((int*)((signed char*)pad_data_shared + (((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10))) && ((((((int)blockIdx.x) / 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10)) < 57)) && (1 <= (((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)))) && ((((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)) < 57)) ? (int)((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 200704) + ((((int)blockIdx.x) / 7) * 1792)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10) * 4)) - 228))))[0] : (int)(int)0);
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 15; ++ic_chunk_outer_outer) {
    __syncthreads();
    if (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) < 100) {
      if (((int)threadIdx.y) < 13) {
          ((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer + 1) & 1) * 400) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10))) && ((((((int)blockIdx.x) / 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10)) < 57)) && (1 <= (((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)))) && ((((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)) < 57)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 200704) + (ic_chunk_outer_outer * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10) * 4)) + 12316))))[0] : (int)(int)0);
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) < 144) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) < 576) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + ((int)threadIdx.y)) < 72) {
              ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) / 9) * 2304) + (ic_chunk_outer_outer * 144)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) % 9) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
        #pragma unroll
        for (int oh = 0; oh < 8; ++oh) {
          #pragma unroll
          for (int oc_block = 0; oc_block < 4; ++oc_block) {
            compute[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 400) + (oh * 40)) + (kh_inner * 40)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((oh * 4) + oc_block))]);
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) < 144) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) < 576) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16) + ((int)threadIdx.y)) < 72) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 512) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) / 9) * 2304) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) % 9) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 2160))))[0];
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
    #pragma unroll
    for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
      #pragma unroll
      for (int oh1 = 0; oh1 < 8; ++oh1) {
        #pragma unroll
        for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
          compute[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((oh1 * 40) + (kh_inner1 * 40)) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 400))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[(((oh1 * 4) + oc_block1))]);
        }
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 8; ++ax2_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (ax2_inner_inner_inner * 224)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1194331400) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.y) * 4) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1194331400) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.y) * 4) + ax4))])), (int)(0)))) * (long)2092595616) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_2_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ compute, void* __restrict__ placeholder2) {
  int compute1[28];
  __shared__ int placeholder_shared[1152];
  __shared__ int pad_data_shared[162];
  #pragma unroll
  for (int oh_init = 0; oh_init < 7; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute1[(((oh_init * 4) + oc_block_init))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 576) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 83) {
          ((int*)((signed char*)placeholder_shared + (((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 12) * 48) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.z) * 28)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((int)blockIdx.y) * 294912) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 36) * 18432)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 36) / 12) * 48)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.z) * 28)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0];
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 127; ++ic_chunk_outer_outer) {
    __syncthreads();
    if ((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 162) {
      if (((((int)threadIdx.z) * 16) + ((int)threadIdx.y)) < 24) {
          ((int*)((signed char*)pad_data_shared + ((((((int)threadIdx.z) * 448) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = (((((9 <= ((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81)) && (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81) < 72)) && (1 <= ((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9))) && (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) < 8)) ? (int)((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.z) * 50176) + (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 81) * 25088)) + (ic_chunk_outer_outer * 196)) + ((((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81) / 9) * 28)) + (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) * 4)) - 32))))[0] : (int)(int)0);
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 576) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 83) {
            ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 2304) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 12) * 48)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.z) * 28)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.y) * 294912) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 36) * 18432)) + (ic_chunk_outer_outer * 144)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 36) / 12) * 48)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.z) * 28)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)) + 144))))[0];
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
        #pragma unroll
        for (int oh = 0; oh < 7; ++oh) {
          #pragma unroll
          for (int oc_block = 0; oc_block < 4; ++oc_block) {
            compute1[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((int)threadIdx.z) * 324) + (oh * 36)) + (kh_inner * 36)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 2304) + (((int)threadIdx.y) * 144)) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute1[(((oh * 4) + oc_block))]);
          }
        }
      }
    }
  }
  __syncthreads();
  if ((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 162) {
    if (((((int)threadIdx.z) * 16) + ((int)threadIdx.y)) < 24) {
        ((int*)((signed char*)pad_data_shared + ((((((int)threadIdx.z) * 448) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = (((((9 <= ((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81)) && (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81) < 72)) && (1 <= ((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9))) && (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) < 8)) ? (int)((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.z) * 50176) + (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 81) * 25088)) + ((((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81) / 9) * 28)) + (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) * 4)) + 24860))))[0] : (int)(int)0);
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
    #pragma unroll
    for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
      #pragma unroll
      for (int oh1 = 0; oh1 < 7; ++oh1) {
        #pragma unroll
        for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
          compute1[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((int)threadIdx.z) * 324) + (oh1 * 36)) + (kh_inner1 * 36)) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)) + 2304))))[0], compute1[(((oh1 * 4) + oc_block1))]);
        }
      }
    }
  }
  #pragma unroll
  for (int i2_inner_inner_inner = 0; i2_inner_inner_inner < 7; ++i2_inner_inner_inner) {
    #pragma unroll
    for (int i4 = 0; i4 < 4; ++i4) {
      ((int*)compute)[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 3136)) + (((int)threadIdx.y) * 196)) + (i2_inner_inner_inner * 28)) + (((int)threadIdx.x) * 4)) + i4))] = ((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute1[(((i2_inner_inner_inner * 4) + i4))]) << ((long)17)) : ((long)compute1[(((i2_inner_inner_inner * 4) + i4))])) * (long)1839999616) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + i4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute1[(((i2_inner_inner_inner * 4) + i4))]) << ((long)17)) : ((long)compute1[(((i2_inner_inner_inner * 4) + i4))])) * (long)1839999616) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + i4))]))) * (long)2068073536) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__2_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[32];
  __shared__ int pad_data_shared[580];
  __shared__ int placeholder_shared[2304];
  #pragma unroll
  for (int n_init = 0; n_init < 2; ++n_init) {
    #pragma unroll
    for (int oc_chunk_init = 0; oc_chunk_init < 2; ++oc_chunk_init) {
      #pragma unroll
      for (int oh_init = 0; oh_init < 2; ++oh_init) {
        #pragma unroll
        for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
          compute[(((((n_init * 16) + (oc_chunk_init * 8)) + (oh_init * 4)) + oc_block_init))] = 0;
        }
      }
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) < 290) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 16) + ((int)threadIdx.y)) < 21) {
          ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 56)) + (((int)threadIdx.x) * 4)))))[0] = (((1 <= (((((int)blockIdx.x) >> 1) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) % 145) / 29))) && (1 <= (((((int)blockIdx.x) & 1) * 28) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) % 29)))) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 401408) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) / 145) * 200704)) + ((((int)blockIdx.x) >> 1) * 896)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) % 145) / 29) * 224)) + ((((int)blockIdx.x) & 1) * 112)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) % 29) * 4)) - 228))))[0] : (int)(int)0);
      }
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 6; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) < 1152) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + ((int)threadIdx.y)) < 83) {
          ((int*)((signed char*)placeholder_shared + ((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) / 12) * 48) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) / 36) * 2304) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) % 36) / 12) * 48)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) & 3) * 4)))))[0];
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 15; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) < 290) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 16) + ((int)threadIdx.y)) < 21) {
            ((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 1160) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 896)) + (((int)threadIdx.y) * 56)) + (((int)threadIdx.x) * 4)))))[0] = (((1 <= (((((int)blockIdx.x) >> 1) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) % 145) / 29))) && (1 <= (((((int)blockIdx.x) & 1) * 28) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) % 29)))) ? (int)((int*)((signed char*)placeholder + (((((((((((int)blockIdx.z) * 401408) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) / 145) * 200704)) + (ic_chunk_outer_outer * 12544)) + ((((int)blockIdx.x) >> 1) * 896)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) % 145) / 29) * 224)) + ((((int)blockIdx.x) & 1) * 112)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) % 29) * 4)) + 12316))))[0] : (int)(int)0);
        }
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 6; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) < 1152) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16) + ((int)threadIdx.y)) < 83) {
            ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 4608) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) / 12) * 48)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) / 36) * 2304) + (ic_chunk_outer_outer * 144)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) % 36) / 12) * 48)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) & 3) * 4)) + 144))))[0];
        }
      }
    }
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
          #pragma unroll
          for (int oc_chunk = 0; oc_chunk < 2; ++oc_chunk) {
            #pragma unroll
            for (int oh = 0; oh < 2; ++oh) {
              #pragma unroll
              for (int oc_block = 0; oc_block < 4; ++oc_block) {
                compute[(((((n * 16) + (oc_chunk * 8)) + (oh * 4)) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((ic_chunk_outer_outer & 1) * 1160) + (n * 580)) + (oh * 232)) + (kh_inner * 116)) + (((int)threadIdx.x) * 8)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((ic_chunk_outer_outer & 1) * 4608) + (((int)threadIdx.y) * 288)) + (oc_chunk * 144)) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((((n * 16) + (oc_chunk * 8)) + (oh * 4)) + oc_block))]);
              }
            }
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
    #pragma unroll
    for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
      #pragma unroll
      for (int n1 = 0; n1 < 2; ++n1) {
        #pragma unroll
        for (int oc_chunk1 = 0; oc_chunk1 < 2; ++oc_chunk1) {
          #pragma unroll
          for (int oh1 = 0; oh1 < 2; ++oh1) {
            #pragma unroll
            for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
              compute[(((((n1 * 16) + (oc_chunk1 * 8)) + (oh1 * 4)) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((n1 * 580) + (oh1 * 232)) + (kh_inner1 * 116)) + (((int)threadIdx.x) * 8)) + (kw_inner1 * 4)) + 1160))))[0], ((int*)((signed char*)placeholder_shared + (((((((((int)threadIdx.y) * 288) + (oc_chunk1 * 144)) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)) + 4608))))[0], compute[(((((n1 * 16) + (oc_chunk1 * 8)) + (oh1 * 4)) + oc_block1))]);
            }
          }
        }
      }
    }
  }
  #pragma unroll
  for (int ax0_inner_inner_inner_inner = 0; ax0_inner_inner_inner_inner < 2; ++ax0_inner_inner_inner_inner) {
    #pragma unroll
    for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 2; ++ax1_inner_inner_inner) {
      #pragma unroll
      for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 2; ++ax2_inner_inner_inner) {
        #pragma unroll
        for (int ax4 = 0; ax4 < 4; ++ax4) {
          ((signed char*)T_cast)[((((((((((((int)blockIdx.z) * 200704) + (ax0_inner_inner_inner_inner * 100352)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + ((((int)blockIdx.x) >> 1) * 224)) + (ax2_inner_inner_inner * 112)) + ((((int)blockIdx.x) & 1) * 56)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((((ax0_inner_inner_inner_inner * 16) + (ax1_inner_inner_inner * 8)) + (ax2_inner_inner_inner * 4)) + ax4))]) << ((long)16)) : ((long)compute[(((((ax0_inner_inner_inner_inner * 16) + (ax1_inner_inner_inner * 8)) + (ax2_inner_inner_inner * 4)) + ax4))])) * (long)1636774251) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)threadIdx.y) * 8) + (ax1_inner_inner_inner * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((((ax0_inner_inner_inner_inner * 16) + (ax1_inner_inner_inner * 8)) + (ax2_inner_inner_inner * 4)) + ax4))]) << ((long)16)) : ((long)compute[(((((ax0_inner_inner_inner_inner * 16) + (ax1_inner_inner_inner * 8)) + (ax2_inner_inner_inner * 4)) + ax4))])) * (long)1636774251) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)threadIdx.y) * 8) + (ax1_inner_inner_inner * 4)) + ax4))])), (int)(0)))) * (long)1223412314) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
        }
      }
    }
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_clip_cast_1_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 13; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) < 802816) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 3211264) {
        ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << ((long)0)) : ((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))])) * (long)2058331460) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
      }
    }
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_nn_conv2d_cast_fixed_point_multiply_add_cast_fix_1476761249106760241__3_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[32];
  __shared__ int pad_data_shared[200];
  __shared__ int placeholder_shared[576];
  #pragma unroll
  for (int oh_init = 0; oh_init < 8; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oh_init * 4) + oc_block_init))] = 0;
    }
  }
  if (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) < 100) {
    if (((int)threadIdx.y) < 13) {
        ((int*)((signed char*)pad_data_shared + (((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10))) && ((((((int)blockIdx.x) / 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10)) < 57)) && (1 <= (((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)))) && ((((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)) < 57)) ? (int)((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 200704) + ((((int)blockIdx.x) / 7) * 1792)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10) * 4)) - 228))))[0] : (int)(int)0);
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 15; ++ic_chunk_outer_outer) {
    __syncthreads();
    if (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) < 100) {
      if (((int)threadIdx.y) < 13) {
          ((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer + 1) & 1) * 400) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10))) && ((((((int)blockIdx.x) / 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10)) < 57)) && (1 <= (((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)))) && ((((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)) < 57)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 200704) + (ic_chunk_outer_outer * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10) * 4)) + 12316))))[0] : (int)(int)0);
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) < 144) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) < 576) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + ((int)threadIdx.y)) < 72) {
              ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) / 9) * 2304) + (ic_chunk_outer_outer * 144)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) % 9) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
        #pragma unroll
        for (int oh = 0; oh < 8; ++oh) {
          #pragma unroll
          for (int oc_block = 0; oc_block < 4; ++oc_block) {
            compute[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 400) + (oh * 40)) + (kh_inner * 40)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((oh * 4) + oc_block))]);
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) < 144) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) < 576) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16) + ((int)threadIdx.y)) < 72) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 512) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) / 9) * 2304) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) % 9) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 2160))))[0];
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
    #pragma unroll
    for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
      #pragma unroll
      for (int oh1 = 0; oh1 < 8; ++oh1) {
        #pragma unroll
        for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
          compute[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((oh1 * 40) + (kh_inner1 * 40)) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 400))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[(((oh1 * 4) + oc_block1))]);
        }
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 8; ++ax2_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((int*)T_relu)[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (ax2_inner_inner_inner * 224)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4))] = max((int)((((int)(((((0 != 0) ? (((long)((int*)placeholder2)[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (ax2_inner_inner_inner * 224)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4))]) << ((long)0)) : ((long)((int*)placeholder2)[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (ax2_inner_inner_inner * 224)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4))])) * (long)1807194190) + ((long)1 << ((long)((1 + 31) - 1)))) >> ((long)(1 + 31)))) + ((int)(((((1 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1574726727) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.y) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1574726727) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.y) * 4) + ax4))]))) * (long)1137624986) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__6_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[32];
  __shared__ int pad_data_shared[120];
  __shared__ int placeholder_shared[384];
  #pragma unroll
  for (int n_init = 0; n_init < 2; ++n_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((n_init * 4) + oc_block_init))] = 0;
      compute[((((n_init * 4) + oc_block_init) + 16))] = 0;
      compute[((((n_init * 4) + oc_block_init) + 8))] = 0;
      compute[((((n_init * 4) + oc_block_init) + 24))] = 0;
    }
  }
  for (int kh_outer = 0; kh_outer < 3; ++kh_outer) {
    for (int ic_chunk_outer = 0; ic_chunk_outer < 32; ++ic_chunk_outer) {
      __syncthreads();
      #pragma unroll
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 120) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) < 18) {
              ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = (((1 <= ((((int)blockIdx.x) * 2) + kh_outer)) && (1 <= ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 15))) ? (int)((int*)((signed char*)placeholder + (((((((((((int)blockIdx.z) * 200704) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 30) * 50176)) + (ic_chunk_outer * 1568)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 30) / 15) * 784)) + (((int)blockIdx.x) * 112)) + (kh_outer * 56)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 15) * 4)) - 60))))[0] : (int)(int)0);
          }
        }
      }
      #pragma unroll
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 384) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) < 55) {
              ((int*)((signed char*)placeholder_shared + ((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 12) * 48) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 14) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((((int)blockIdx.y) * 147456) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 24) * 9216)) + (ic_chunk_outer * 288)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 24) / 12) * 144)) + (kh_outer * 48)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 14) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0];
          }
        }
      }
      __syncthreads();
      #pragma unroll
      for (int ic_chunk_inner = 0; ic_chunk_inner < 2; ++ic_chunk_inner) {
        #pragma unroll
        for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
          #pragma unroll
          for (int n = 0; n < 2; ++n) {
            #pragma unroll
            for (int oc_block = 0; oc_block < 4; ++oc_block) {
              compute[(((n * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((n * 120) + (ic_chunk_inner * 60)) + (((int)threadIdx.x) * 8)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 96) + (ic_chunk_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((n * 4) + oc_block))]);
              compute[((((n * 4) + oc_block) + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((n * 120) + (ic_chunk_inner * 60)) + (((int)threadIdx.x) * 8)) + (kw_inner * 4)) + 240))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 96) + (ic_chunk_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[((((n * 4) + oc_block) + 16))]);
              compute[((((n * 4) + oc_block) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((n * 120) + (ic_chunk_inner * 60)) + (((int)threadIdx.x) * 8)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.y) * 96) + (ic_chunk_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)) + 768))))[0], compute[((((n * 4) + oc_block) + 8))]);
              compute[((((n * 4) + oc_block) + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((n * 120) + (ic_chunk_inner * 60)) + (((int)threadIdx.x) * 8)) + (kw_inner * 4)) + 240))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.y) * 96) + (ic_chunk_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)) + 768))))[0], compute[((((n * 4) + oc_block) + 24))]);
            }
          }
        }
      }
    }
  }
  #pragma unroll
  for (int ax0_inner_inner_inner_inner = 0; ax0_inner_inner_inner_inner < 2; ++ax0_inner_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (ax0_inner_inner_inner_inner * 25088)) + (((int)blockIdx.y) * 3136)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((17 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1345670983) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((17 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1345670983) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1104268508) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
      ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (ax0_inner_inner_inner_inner * 25088)) + (((int)blockIdx.y) * 3136)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 50176))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 16))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 16))])) * (long)1345670983) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 16))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 16))])) * (long)1345670983) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1104268508) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
      ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (ax0_inner_inner_inner_inner * 25088)) + (((int)blockIdx.y) * 3136)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 1568))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))])) * (long)1345670983) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))])) * (long)1345670983) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])), (int)(0)))) * (long)1104268508) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
      ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (ax0_inner_inner_inner_inner * 25088)) + (((int)blockIdx.y) * 3136)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 51744))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 24))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 24))])) * (long)1345670983) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 24))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 24))])) * (long)1345670983) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])), (int)(0)))) * (long)1104268508) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_clip_cast_2_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 25; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) < 1605632) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 6422528) {
        ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << ((long)0)) : ((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))])) * (long)1763041040) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
      }
    }
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_nn_conv2d_cast_fixed_point_multiply_add_cast_fix_5734298839780500801__kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_multiply, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[28];
  __shared__ int placeholder_shared[1152];
  __shared__ int pad_data_shared[162];
  #pragma unroll
  for (int oh_init = 0; oh_init < 7; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oh_init * 4) + oc_block_init))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 576) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 83) {
          ((int*)((signed char*)placeholder_shared + (((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 12) * 48) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.z) * 28)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((int)blockIdx.y) * 294912) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 36) * 18432)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 36) / 12) * 48)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.z) * 28)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0];
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 127; ++ic_chunk_outer_outer) {
    __syncthreads();
    if ((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 162) {
      if (((((int)threadIdx.z) * 16) + ((int)threadIdx.y)) < 24) {
          ((int*)((signed char*)pad_data_shared + ((((((int)threadIdx.z) * 448) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = (((((9 <= ((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81)) && (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81) < 72)) && (1 <= ((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9))) && (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) < 8)) ? (int)((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.z) * 50176) + (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 81) * 25088)) + (ic_chunk_outer_outer * 196)) + ((((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81) / 9) * 28)) + (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) * 4)) - 32))))[0] : (int)(int)0);
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 576) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 83) {
            ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 2304) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 12) * 48)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.z) * 28)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.y) * 294912) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 36) * 18432)) + (ic_chunk_outer_outer * 144)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 36) / 12) * 48)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.z) * 28)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)) + 144))))[0];
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
        #pragma unroll
        for (int oh = 0; oh < 7; ++oh) {
          #pragma unroll
          for (int oc_block = 0; oc_block < 4; ++oc_block) {
            compute[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((int)threadIdx.z) * 324) + (oh * 36)) + (kh_inner * 36)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 2304) + (((int)threadIdx.y) * 144)) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((oh * 4) + oc_block))]);
          }
        }
      }
    }
  }
  __syncthreads();
  if ((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 162) {
    if (((((int)threadIdx.z) * 16) + ((int)threadIdx.y)) < 24) {
        ((int*)((signed char*)pad_data_shared + ((((((int)threadIdx.z) * 448) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = (((((9 <= ((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81)) && (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81) < 72)) && (1 <= ((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9))) && (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) < 8)) ? (int)((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.z) * 50176) + (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 81) * 25088)) + ((((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81) / 9) * 28)) + (((((((int)threadIdx.z) * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) * 4)) + 24860))))[0] : (int)(int)0);
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
    #pragma unroll
    for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
      #pragma unroll
      for (int oh1 = 0; oh1 < 7; ++oh1) {
        #pragma unroll
        for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
          compute[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((int)threadIdx.z) * 324) + (oh1 * 36)) + (kh_inner1 * 36)) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)) + 2304))))[0], compute[(((oh1 * 4) + oc_block1))]);
        }
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 7; ++ax2_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((float*)T_multiply)[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 3136)) + (((int)threadIdx.y) * 196)) + (ax2_inner_inner_inner * 28)) + (((int)threadIdx.x) * 4)) + ax4))] = (((float)max((int)((((int)(((((0 != 0) ? (((long)((int*)placeholder2)[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 3136)) + (((int)threadIdx.y) * 196)) + (ax2_inner_inner_inner * 28)) + (((int)threadIdx.x) * 4)) + ax4))]) << ((long)0)) : ((long)((int*)placeholder2)[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 3136)) + (((int)threadIdx.y) * 196)) + (ax2_inner_inner_inner * 28)) + (((int)threadIdx.x) * 4)) + ax4))])) * (long)1389035277) + ((long)1 << ((long)((2 + 31) - 1)))) >> ((long)(2 + 31)))) + ((int)(((((0 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1494953172) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1494953172) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2115906545) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0))) * 1.547196e-08f);
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__5_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[28];
  __shared__ int pad_data_shared[1024];
  __shared__ int placeholder_shared[1152];
  #pragma unroll
  for (int oh_init = 0; oh_init < 7; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oh_init * 4) + oc_block_init))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
      ((int*)((signed char*)pad_data_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) & 15)) && ((((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) & 15) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + ((int)threadIdx.x)))) && (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) < 15)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 200704) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 50176)) + ((((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) >> 4) * 784)) + ((((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) & 15) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 4)) - 60))))[0] : (int)(int)0);
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) < 144) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 576) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 4) + ((int)threadIdx.z)) < 18) {
            ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 73728) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) / 18) * 9216)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) % 18) * 16)) + (((int)threadIdx.x) * 4)))))[0];
        }
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 31; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
        ((int*)((signed char*)pad_data_shared + ((((((((ic_chunk_outer_outer + 1) & 1) * 2048) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 512)) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) & 15)) && ((((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) & 15) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + ((int)threadIdx.x)))) && (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) < 15)) ? (int)((int*)((signed char*)placeholder + (((((((((((int)blockIdx.z) * 200704) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 50176)) + (ic_chunk_outer_outer * 1568)) + ((((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) >> 4) * 784)) + ((((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) & 15) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 4)) + 1508))))[0] : (int)(int)0);
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) < 144) {
        if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 576) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 4) + ((int)threadIdx.z)) < 18) {
              ((int*)((signed char*)placeholder_shared + ((((((((ic_chunk_outer_outer + 1) & 1) * 2304) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 512)) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 73728) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) / 18) * 9216)) + (ic_chunk_outer_outer * 288)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) % 18) * 16)) + (((int)threadIdx.x) * 4)) + 288))))[0];
          }
        }
      }
    }
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 2; ++ic_chunk_inner) {
      #pragma unroll
      for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
        #pragma unroll
        for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
          #pragma unroll
          for (int oh = 0; oh < 7; ++oh) {
            #pragma unroll
            for (int oc_block = 0; oc_block < 4; ++oc_block) {
              compute[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((ic_chunk_outer_outer & 1) * 2048) + (((int)threadIdx.z) * 512)) + (ic_chunk_inner * 256)) + ((((int)threadIdx.x) >> 1) * 112)) + (oh * 16)) + (kh_inner * 16)) + (kw_inner * 4)) + ((((int)threadIdx.x) & 1) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((ic_chunk_outer_outer & 1) * 2304) + (((int)threadIdx.y) * 288)) + (ic_chunk_inner * 144)) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((oh * 4) + oc_block))]);
            }
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 2; ++ic_chunk_inner1) {
    #pragma unroll
    for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
      #pragma unroll
      for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
        #pragma unroll
        for (int oh1 = 0; oh1 < 7; ++oh1) {
          #pragma unroll
          for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
            compute[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((((((int)threadIdx.z) * 512) + (ic_chunk_inner1 * 256)) + ((((int)threadIdx.x) >> 1) * 112)) + (oh1 * 16)) + (kh_inner1 * 16)) + (kw_inner1 * 4)) + ((((int)threadIdx.x) & 1) * 4)) + 2048))))[0], ((int*)((signed char*)placeholder_shared + (((((((((int)threadIdx.y) * 288) + (ic_chunk_inner1 * 144)) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)) + 2304))))[0], compute[(((oh1 * 4) + oc_block1))]);
          }
        }
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 7; ++ax2_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[((((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 784)) + ((((int)threadIdx.x) >> 1) * 392)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1954738632) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1954738632) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1275273255) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_1_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ compute, void* __restrict__ placeholder2) {
  int compute1[28];
  __shared__ int pad_data_shared[1024];
  __shared__ int placeholder_shared[1152];
  #pragma unroll
  for (int oh_init = 0; oh_init < 7; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute1[(((oh_init * 4) + oc_block_init))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
      ((int*)((signed char*)pad_data_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) & 15)) && ((((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) & 15) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + ((int)threadIdx.x)))) && (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) < 15)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 200704) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 50176)) + ((((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) >> 4) * 784)) + ((((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) & 15) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 4)) - 60))))[0] : (int)(int)0);
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) < 144) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 576) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 4) + ((int)threadIdx.z)) < 18) {
            ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 73728) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) / 18) * 9216)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) % 18) * 16)) + (((int)threadIdx.x) * 4)))))[0];
        }
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 31; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
        ((int*)((signed char*)pad_data_shared + ((((((((ic_chunk_outer_outer + 1) & 1) * 2048) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 512)) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) & 15)) && ((((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) & 15) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + ((int)threadIdx.x)))) && (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) < 15)) ? (int)((int*)((signed char*)placeholder + (((((((((((int)blockIdx.z) * 200704) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 50176)) + (ic_chunk_outer_outer * 1568)) + ((((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) >> 4) * 784)) + ((((((int)threadIdx.z) * 8) + ((int)threadIdx.y)) & 15) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 4)) + 1508))))[0] : (int)(int)0);
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) < 144) {
        if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 576) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 4) + ((int)threadIdx.z)) < 18) {
              ((int*)((signed char*)placeholder_shared + ((((((((ic_chunk_outer_outer + 1) & 1) * 2304) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 512)) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 73728) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) / 18) * 9216)) + (ic_chunk_outer_outer * 288)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) % 18) * 16)) + (((int)threadIdx.x) * 4)) + 288))))[0];
          }
        }
      }
    }
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 2; ++ic_chunk_inner) {
      #pragma unroll
      for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
        #pragma unroll
        for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
          #pragma unroll
          for (int oh = 0; oh < 7; ++oh) {
            #pragma unroll
            for (int oc_block = 0; oc_block < 4; ++oc_block) {
              compute1[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((ic_chunk_outer_outer & 1) * 2048) + (((int)threadIdx.z) * 512)) + (ic_chunk_inner * 256)) + ((((int)threadIdx.x) >> 1) * 112)) + (oh * 16)) + (kh_inner * 16)) + (kw_inner * 4)) + ((((int)threadIdx.x) & 1) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((ic_chunk_outer_outer & 1) * 2304) + (((int)threadIdx.y) * 288)) + (ic_chunk_inner * 144)) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute1[(((oh * 4) + oc_block))]);
            }
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 2; ++ic_chunk_inner1) {
    #pragma unroll
    for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
      #pragma unroll
      for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
        #pragma unroll
        for (int oh1 = 0; oh1 < 7; ++oh1) {
          #pragma unroll
          for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
            compute1[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((((((int)threadIdx.z) * 512) + (ic_chunk_inner1 * 256)) + ((((int)threadIdx.x) >> 1) * 112)) + (oh1 * 16)) + (kh_inner1 * 16)) + (kw_inner1 * 4)) + ((((int)threadIdx.x) & 1) * 4)) + 2048))))[0], ((int*)((signed char*)placeholder_shared + (((((((((int)threadIdx.y) * 288) + (ic_chunk_inner1 * 144)) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)) + 2304))))[0], compute1[(((oh1 * 4) + oc_block1))]);
          }
        }
      }
    }
  }
  #pragma unroll
  for (int i2_inner_inner_inner = 0; i2_inner_inner_inner < 7; ++i2_inner_inner_inner) {
    #pragma unroll
    for (int i4 = 0; i4 < 4; ++i4) {
      ((int*)compute)[((((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 784)) + ((((int)threadIdx.x) >> 1) * 392)) + (i2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + i4))] = ((int)(((((0 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute1[(((i2_inner_inner_inner * 4) + i4))]) << ((long)16)) : ((long)compute1[(((i2_inner_inner_inner * 4) + i4))])) * (long)1968217277) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + i4))])) << ((long)0)) : ((long)(((int)(((((16 != 0) ? (((long)compute1[(((i2_inner_inner_inner * 4) + i4))]) << ((long)16)) : ((long)compute1[(((i2_inner_inner_inner * 4) + i4))])) * (long)1968217277) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + i4))]))) * (long)2088981575) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_13255070459903635369__1_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[32];
  __shared__ int pad_data_shared[432];
  __shared__ int placeholder_shared[256];
  #pragma unroll
  for (int n_init = 0; n_init < 4; ++n_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((n_init * 4) + oc_block_init))] = 0;
      compute[((((n_init * 4) + oc_block_init) + 16))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 14)) + ((int)threadIdx.x)) < 216) {
        ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 448) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((int)blockIdx.z) * 401408) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 14)) + ((int)threadIdx.x)) / 54) * 100352)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 14)) + ((int)threadIdx.x)) % 54) / 27) * 3136)) + (((int)blockIdx.x) * 224)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 14)) + ((int)threadIdx.x)) % 27) * 4)))))[0];
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 14)) + ((int)threadIdx.x)) < 128) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.z)) < 10) {
        if ((((((int)blockIdx.y) * 16) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 14)) + (((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) >> 3)) < 64) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 448) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 8192) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 7168)) + ((((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) >> 3) * 512)) + ((((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) & 7) * 4)))))[0];
        }
      }
    }
  }
  #pragma unroll
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 15; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 14)) + ((int)threadIdx.x)) < 216) {
          ((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 864) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 448)) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 401408) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 14)) + ((int)threadIdx.x)) / 54) * 100352)) + (ic_chunk_outer_outer * 6272)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 14)) + ((int)threadIdx.x)) % 54) / 27) * 3136)) + (((int)blockIdx.x) * 224)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 14)) + ((int)threadIdx.x)) % 27) * 4)) + 6272))))[0];
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 14)) + ((int)threadIdx.x)) < 128) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.z)) < 10) {
          if ((((((int)blockIdx.y) * 16) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 14)) + (((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) >> 3)) < 64) {
              ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 512) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 448)) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 8192) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 7168)) + ((((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) >> 3) * 512)) + (ic_chunk_outer_outer * 32)) + ((((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) & 7) * 4)) + 32))))[0];
          }
        }
      }
    }
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 2; ++ic_chunk_inner) {
      #pragma unroll
      for (int n = 0; n < 4; ++n) {
        #pragma unroll
        for (int oc_block = 0; oc_block < 4; ++oc_block) {
          compute[(((n * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 864) + (n * 216)) + (ic_chunk_inner * 108)) + (((int)threadIdx.x) * 8)))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 512) + (((int)threadIdx.z) * 32)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(((n * 4) + oc_block))]);
          compute[((((n * 4) + oc_block) + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 864) + (n * 216)) + (ic_chunk_inner * 108)) + (((int)threadIdx.x) * 8)))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 512) + (((int)threadIdx.z) * 32)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 256))))[0], compute[((((n * 4) + oc_block) + 16))]);
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 2; ++ic_chunk_inner1) {
    #pragma unroll
    for (int n1 = 0; n1 < 4; ++n1) {
      #pragma unroll
      for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
        compute[(((n1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((n1 * 216) + (ic_chunk_inner1 * 108)) + (((int)threadIdx.x) * 8)) + 864))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 32) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 512))))[0], compute[(((n1 * 4) + oc_block1))]);
        compute[((((n1 * 4) + oc_block1) + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((n1 * 216) + (ic_chunk_inner1 * 108)) + (((int)threadIdx.x) * 8)) + 864))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 32) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 768))))[0], compute[((((n1 * 4) + oc_block1) + 16))]);
      }
    }
  }
  #pragma unroll
  for (int ax0_inner_inner_inner_inner = 0; ax0_inner_inner_inner_inner < 4; ++ax0_inner_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((int*)T_relu)[((((((((((int)blockIdx.z) * 200704) + (ax0_inner_inner_inner_inner * 50176)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.x) * 56)) + (((int)threadIdx.x) * 4)) + ax4))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1170196338) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1170196338) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)1537115942) + ((long)1 << ((long)((1 + 31) - 1)))) >> ((long)(1 + 31)))) + ((int*)placeholder3)[((((((((((int)blockIdx.z) * 200704) + (ax0_inner_inner_inner_inner * 50176)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.x) * 56)) + (((int)threadIdx.x) * 4)) + ax4))])), (int)(0));
      ((int*)T_relu)[(((((((((((int)blockIdx.z) * 200704) + (ax0_inner_inner_inner_inner * 50176)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.x) * 56)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 16))]) << ((long)18)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 16))])) * (long)1170196338) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 16))]) << ((long)18)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 16))])) * (long)1170196338) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4) + 32))]))) * (long)1537115942) + ((long)1 << ((long)((1 + 31) - 1)))) >> ((long)(1 + 31)))) + ((int*)placeholder3)[(((((((((((int)blockIdx.z) * 200704) + (ax0_inner_inner_inner_inner * 50176)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.x) * 56)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))])), (int)(0));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ compute, void* __restrict__ placeholder2) {
  int compute1[56];
  __shared__ int pad_data_shared[96];
  __shared__ int placeholder_shared[288];
  #pragma unroll
  for (int oc_chunk_init = 0; oc_chunk_init < 2; ++oc_chunk_init) {
    #pragma unroll
    for (int oh_init = 0; oh_init < 7; ++oh_init) {
      #pragma unroll
      for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
        compute1[((((oc_chunk_init * 28) + (oh_init * 4)) + oc_block_init))] = 0;
      }
    }
  }
  for (int ic_chunk_outer = 0; ic_chunk_outer < 32; ++ic_chunk_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)pad_data_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 14) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 6))) && ((((((int)blockIdx.x) / 7) * 14) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 6)) < 29)) && (1 <= (((((int)blockIdx.x) % 7) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6)))) && ((((((int)blockIdx.x) % 7) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6)) < 29)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 100352) + (ic_chunk_outer * 3136)) + ((((int)blockIdx.x) / 7) * 1568)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 6) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6) * 4)) - 116))))[0] : (int)(int)0);
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 9; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 36864) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + (((int)threadIdx.z) * 2)) + ((int)threadIdx.y)) / 9) * 4608)) + (ic_chunk_outer * 144)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + (((int)threadIdx.z) * 2)) + ((int)threadIdx.y)) % 9) * 16)) + (((int)threadIdx.x) * 4)))))[0];
    }
    __syncthreads();
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
        #pragma unroll
        for (int oc_chunk = 0; oc_chunk < 2; ++oc_chunk) {
          #pragma unroll
          for (int oh = 0; oh < 7; ++oh) {
            #pragma unroll
            for (int oc_block = 0; oc_block < 4; ++oc_block) {
              compute1[((((oc_chunk * 28) + (oh * 4)) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((int)threadIdx.y) * 168) + (oh * 24)) + (kh_inner * 24)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.z) * 288) + (oc_chunk * 144)) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute1[((((oc_chunk * 28) + (oh * 4)) + oc_block))]);
            }
          }
        }
      }
    }
  }
  #pragma unroll
  for (int i1_inner_inner_inner = 0; i1_inner_inner_inner < 2; ++i1_inner_inner_inner) {
    #pragma unroll
    for (int i2_inner_inner_inner = 0; i2_inner_inner_inner < 7; ++i2_inner_inner_inner) {
      #pragma unroll
      for (int i4 = 0; i4 < 4; ++i4) {
        ((int*)compute)[(((((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 25088)) + (((int)threadIdx.z) * 6272)) + (i1_inner_inner_inner * 3136)) + ((((int)blockIdx.x) / 7) * 1568)) + (((int)threadIdx.y) * 784)) + (i2_inner_inner_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) * 4)) + i4))] = ((int)(((((1 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute1[((((i1_inner_inner_inner * 28) + (i2_inner_inner_inner * 4)) + i4))]) << ((long)16)) : ((long)compute1[((((i1_inner_inner_inner * 28) + (i2_inner_inner_inner * 4)) + i4))])) * (long)2051410042) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 32) + (((int)threadIdx.z) * 8)) + (i1_inner_inner_inner * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((16 != 0) ? (((long)compute1[((((i1_inner_inner_inner * 28) + (i2_inner_inner_inner * 4)) + i4))]) << ((long)16)) : ((long)compute1[((((i1_inner_inner_inner * 28) + (i2_inner_inner_inner * 4)) + i4))])) * (long)2051410042) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 32) + (((int)threadIdx.z) * 8)) + (i1_inner_inner_inner * 4)) + i4))]))) * (long)1127693289) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
      }
    }
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_clip_cast_6_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) < 200704) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 802816) {
        ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << ((long)0)) : ((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))])) * (long)1773341876) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__3_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[56];
  __shared__ int pad_data_shared[96];
  __shared__ int placeholder_shared[288];
  #pragma unroll
  for (int oc_chunk_init = 0; oc_chunk_init < 2; ++oc_chunk_init) {
    #pragma unroll
    for (int oh_init = 0; oh_init < 7; ++oh_init) {
      #pragma unroll
      for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
        compute[((((oc_chunk_init * 28) + (oh_init * 4)) + oc_block_init))] = 0;
      }
    }
  }
  for (int ic_chunk_outer = 0; ic_chunk_outer < 32; ++ic_chunk_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)pad_data_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 14) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 6))) && ((((((int)blockIdx.x) / 7) * 14) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 6)) < 29)) && (1 <= (((((int)blockIdx.x) % 7) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6)))) && ((((((int)blockIdx.x) % 7) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6)) < 29)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 100352) + (ic_chunk_outer * 3136)) + ((((int)blockIdx.x) / 7) * 1568)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 6) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6) * 4)) - 116))))[0] : (int)(int)0);
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 9; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 36864) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + (((int)threadIdx.z) * 2)) + ((int)threadIdx.y)) / 9) * 4608)) + (ic_chunk_outer * 144)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + (((int)threadIdx.z) * 2)) + ((int)threadIdx.y)) % 9) * 16)) + (((int)threadIdx.x) * 4)))))[0];
    }
    __syncthreads();
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
        #pragma unroll
        for (int oc_chunk = 0; oc_chunk < 2; ++oc_chunk) {
          #pragma unroll
          for (int oh = 0; oh < 7; ++oh) {
            #pragma unroll
            for (int oc_block = 0; oc_block < 4; ++oc_block) {
              compute[((((oc_chunk * 28) + (oh * 4)) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((int)threadIdx.y) * 168) + (oh * 24)) + (kh_inner * 24)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.z) * 288) + (oc_chunk * 144)) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[((((oc_chunk * 28) + (oh * 4)) + oc_block))]);
            }
          }
        }
      }
    }
  }
  #pragma unroll
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 2; ++ax1_inner_inner_inner) {
    #pragma unroll
    for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 7; ++ax2_inner_inner_inner) {
      #pragma unroll
      for (int ax4 = 0; ax4 < 4; ++ax4) {
        ((signed char*)T_cast)[(((((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 25088)) + (((int)threadIdx.z) * 6272)) + (ax1_inner_inner_inner * 3136)) + ((((int)blockIdx.x) / 7) * 1568)) + (((int)threadIdx.y) * 784)) + (ax2_inner_inner_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 28) + (ax2_inner_inner_inner * 4)) + ax4))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 28) + (ax2_inner_inner_inner * 4)) + ax4))])) * (long)1425803333) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 32) + (((int)threadIdx.z) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 28) + (ax2_inner_inner_inner * 4)) + ax4))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 28) + (ax2_inner_inner_inner * 4)) + ax4))])) * (long)1425803333) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 32) + (((int)threadIdx.z) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])), (int)(0)))) * (long)1102733142) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_13255070459903635369__2_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[8];
  __shared__ int pad_data_shared[440];
  __shared__ int placeholder_shared[128];
  #pragma unroll
  for (int n_init = 0; n_init < 2; ++n_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((n_init * 4) + oc_block_init))] = 0;
    }
  }
  #pragma unroll
  for (int ic_chunk_outer = 0; ic_chunk_outer < 4; ++ic_chunk_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 28)) + ((int)threadIdx.x)) < 440) {
          ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.z) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 401408) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 28)) + ((int)threadIdx.x)) / 220) * 200704)) + (ic_chunk_outer * 50176)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 28)) + ((int)threadIdx.x)) % 220) / 55) * 12544)) + (((int)blockIdx.x) * 448)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 28)) + ((int)threadIdx.x)) % 55) * 4)))))[0];
      }
    }
    if (((((int)threadIdx.z) * 7) + (((int)threadIdx.x) >> 2)) < 32) {
      if (((((int)threadIdx.z) * 28) + ((int)threadIdx.x)) < 128) {
        if (((int)threadIdx.z) < 5) {
            ((int*)((signed char*)placeholder_shared + (((((int)threadIdx.z) * 112) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 2048) + ((((((int)threadIdx.z) * 7) + (((int)threadIdx.x) >> 2)) >> 2) * 256)) + (ic_chunk_outer * 64)) + ((((((int)threadIdx.z) * 7) + (((int)threadIdx.x) >> 2)) & 3) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 4; ++ic_chunk_inner) {
      #pragma unroll
      for (int n = 0; n < 2; ++n) {
        #pragma unroll
        for (int oc_block = 0; oc_block < 4; ++oc_block) {
          compute[(((n * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n * 880) + (ic_chunk_inner * 220)) + (((int)threadIdx.x) * 8)))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 64) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(((n * 4) + oc_block))]);
        }
      }
    }
  }
  #pragma unroll
  for (int ax0_inner_inner_inner_inner = 0; ax0_inner_inner_inner_inner < 2; ++ax0_inner_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((int*)T_relu)[((((((((((int)blockIdx.z) * 200704) + (ax0_inner_inner_inner_inner * 100352)) + (((int)blockIdx.y) * 25088)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1608224298) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1608224298) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)1462212557) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((((((int)blockIdx.z) * 200704) + (ax0_inner_inner_inner_inner * 100352)) + (((int)blockIdx.y) * 25088)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))])), (int)(0));
    }
  }
}

extern "C" __global__ void fused_layout_transform_nn_batch_flatten_kernel0(void* __restrict__ tensor, void* __restrict__ placeholder) {
  ((float*)tensor)[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = ((float*)placeholder)[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))];
}

