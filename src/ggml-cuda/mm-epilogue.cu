#include "mm-epilogue.cuh"

static __device__ __forceinline__ float op_gelu_erf(float x) {
    constexpr float SQRT_2_INV = 0.7071067811865475f;
    return 0.5f * x * (1.0f + erff(x * SQRT_2_INV));
}

template <bool APPLY_GELU>
static __global__ void k_mm_row_bias(
        float *       __restrict__ x,
        const float * __restrict__ bias,
        const int64_t              ne0,
        const int64_t              ne) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne) {
        return;
    }
    float v = x[i] + bias[i % ne0];
    if constexpr (APPLY_GELU) {
        v = op_gelu_erf(v);
    }
    x[i] = v;
}

void ggml_cuda_op_mm_row_bias(
        ggml_backend_cuda_context & ctx,
        float *                     x,
        const float *               bias,
        int64_t                     ne0,
        int64_t                     ne,
        bool                        apply_gelu_erf) {
    GGML_ASSERT(x && bias);
    GGML_ASSERT(ne0 > 0 && ne >= 0 && ne % ne0 == 0);

    if (ne == 0) {
        return;
    }

    constexpr int block = 256;
    const int grid = (int) ((ne + block - 1) / block);
    cudaStream_t stream = ctx.stream();

    if (apply_gelu_erf) {
        k_mm_row_bias<true><<<grid, block, 0, stream>>>(x, bias, ne0, ne);
    } else {
        k_mm_row_bias<false><<<grid, block, 0, stream>>>(x, bias, ne0, ne);
    }
}

template <bool APPLY_GELU>
static __global__ void k_mm_f16_to_f32_bias(
        const half *  __restrict__ src,
        float *       __restrict__ dst,
        const float * __restrict__ bias,
        const int64_t              ne0,
        const int64_t              ne) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne) {
        return;
    }
    float v = __half2float(src[i]) + bias[i % ne0];
    if constexpr (APPLY_GELU) {
        v = op_gelu_erf(v);
    }
    dst[i] = v;
}

// Vectorized epilogue: one thread covers VEC consecutive elements of a row.
//
// EWI-1963 f16-stream. The scalar kernel above is one element per thread and
// pays a 64-bit `i % ne0` per element; at [1024, 5184] f32 it runs at ~930 GB/s
// against a ~1.8 TB/s device. This kernel is the same arithmetic on the same
// values -- `__half2float` lane-for-lane, `+ bias`, then `op_gelu_erf` in the
// same order -- so it is bit-identical; it just removes the modulo entirely by
// putting the row on blockIdx.y, and moves 4 elements per thread.
//
// Requires: ne0 % VEC == 0 (so no pair straddles a row boundary, which is what
// makes the bias index a plain multiply), and 16-byte-aligned src/dst/bias.
// The dispatcher checks all of those and falls back to the scalar kernel.
template <bool APPLY_GELU, int VEC>
static __global__ void k_mm_f16_to_f32_bias_vec(
        const half *  __restrict__ src,
        float *       __restrict__ dst,
        const float * __restrict__ bias,
        const int64_t              nbias,   // = ne0 / VEC, vectors per row
        const int64_t              nvec) {   // vectors per row actually launched
    const int64_t col = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= nvec) {
        return;
    }
    const int64_t row = blockIdx.y;
    const int64_t e   = (row * nbias + col) * VEC;
    const int64_t b   = col * VEC;

    const half * s = src + e;
    const float * bias_v = bias + b;

    if constexpr (VEC == 4) {
        const half2 a0 = *reinterpret_cast<const half2 *>(s);
        const half2 a1 = *reinterpret_cast<const half2 *>(s + 2);
        float4 v;
        v.x = __low2float (a0) + bias_v[0];
        v.y = __high2float(a0) + bias_v[1];
        v.z = __low2float (a1) + bias_v[2];
        v.w = __high2float(a1) + bias_v[3];
        if constexpr (APPLY_GELU) {
            v.x = op_gelu_erf(v.x);
            v.y = op_gelu_erf(v.y);
            v.z = op_gelu_erf(v.z);
            v.w = op_gelu_erf(v.w);
        }
        *reinterpret_cast<float4 *>(dst + e) = v;
    } else {
        const half2 a0 = *reinterpret_cast<const half2 *>(s);
        float2 v;
        v.x = __low2float (a0) + bias_v[0];
        v.y = __high2float(a0) + bias_v[1];
        if constexpr (APPLY_GELU) {
            v.x = op_gelu_erf(v.x);
            v.y = op_gelu_erf(v.y);
        }
        *reinterpret_cast<float2 *>(dst + e) = v;
    }
}

void ggml_cuda_op_mm_f16_to_f32_bias(
        ggml_backend_cuda_context & ctx,
        const half *                src,
        float *                     dst,
        const float *               bias,
        int64_t                     ne0,
        int64_t                     ne,
        bool                        apply_gelu_erf) {
    GGML_ASSERT(src && dst && bias);
    GGML_ASSERT(ne0 > 0 && ne >= 0 && ne % ne0 == 0);
    if (ne == 0) {
        return;
    }
    constexpr int block = 256;
    cudaStream_t stream = ctx.stream();

    const int64_t nrows = ne / ne0;
    const uintptr_t a_src  = reinterpret_cast<uintptr_t>(src);
    const uintptr_t a_dst  = reinterpret_cast<uintptr_t>(dst);
    const uintptr_t a_bias = reinterpret_cast<uintptr_t>(bias);

    // ne0 = 1024 in every ViT/neck linear this serves, so VEC=4 will be taken;
    // the VEC=2 kernel is the guard for a row length that is even but not a
    // multiple of four.
    const int vec = (ne0 % 4 == 0 && a_src % 8 == 0 && a_dst % 16 == 0 && a_bias % 16 == 0) ? 4 :
                    (ne0 % 2 == 0 && a_src % 4 == 0 && a_dst % 8  == 0 && a_bias % 8  == 0) ? 2 : 0;
    const bool vec_ok = vec != 0 && nrows <= 65535;

    if (vec_ok) {
        const int64_t nbias = ne0 / vec;
        const int64_t nvec  = nbias;
        dim3 grid((unsigned) ((nvec + block - 1) / block), (unsigned) nrows, 1);
        if (vec == 4) {
            if (apply_gelu_erf) {
                k_mm_f16_to_f32_bias_vec<true,  4><<<grid, block, 0, stream>>>(src, dst, bias, nbias, nvec);
            } else {
                k_mm_f16_to_f32_bias_vec<false, 4><<<grid, block, 0, stream>>>(src, dst, bias, nbias, nvec);
            }
        } else {
            if (apply_gelu_erf) {
                k_mm_f16_to_f32_bias_vec<true,  2><<<grid, block, 0, stream>>>(src, dst, bias, nbias, nvec);
            } else {
                k_mm_f16_to_f32_bias_vec<false, 2><<<grid, block, 0, stream>>>(src, dst, bias, nbias, nvec);
            }
        }
        return;
    }

    const int grid = (int) ((ne + block - 1) / block);
    if (apply_gelu_erf) {
        k_mm_f16_to_f32_bias<true><<<grid, block, 0, stream>>>(src, dst, bias, ne0, ne);
    } else {
        k_mm_f16_to_f32_bias<false><<<grid, block, 0, stream>>>(src, dst, bias, ne0, ne);
    }
}
