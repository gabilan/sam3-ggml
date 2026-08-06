#pragma once
#include "common.cuh"

// In-place row bias (+ optional gelu_erf) on a contiguous F32 matrix laid out
// as [ne0, …] with ne0 contiguous (ggml mul_mat output). Used to fuse ViT
// broadcast bias into the large-GEMM path (unequal-shape ADD).
void ggml_cuda_op_mm_row_bias(
        ggml_backend_cuda_context & ctx,
        float *                     x,
        const float *               bias,
        int64_t                     ne0,
        int64_t                     ne,
        bool                        apply_gelu_erf);

// F16→F32 convert with fused row bias (+ optional gelu). Replaces the separate
// convert_unary + k_bin_bcast/mm_row_bias pair after COMPUTE_16F GEMM.
void ggml_cuda_op_mm_f16_to_f32_bias(
        ggml_backend_cuda_context & ctx,
        const half *                src,
        float *                     dst,
        const float *               bias,
        int64_t                     ne0,
        int64_t                     ne,
        bool                        apply_gelu_erf);
