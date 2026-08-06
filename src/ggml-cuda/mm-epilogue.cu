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
    const int grid = (int) ((ne + block - 1) / block);
    cudaStream_t stream = ctx.stream();
    if (apply_gelu_erf) {
        k_mm_f16_to_f32_bias<true><<<grid, block, 0, stream>>>(src, dst, bias, ne0, ne);
    } else {
        k_mm_f16_to_f32_bias<false><<<grid, block, 0, stream>>>(src, dst, bias, ne0, ne);
    }
}
