#include "common.cuh"

#define CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE 256

void ggml_cuda_conv_2d_transpose_p0(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Fused k2s2 ConvTranspose + channel bias (+ optional gelu_erf).
// Writes directly to `dst` (ADD or GELU output buffer). `bias` length c_out F32.
// Returns false if shapes are not the k2s2 path.
bool ggml_cuda_conv_2d_transpose_k2s2_bias(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * kernel,
    const ggml_tensor * input,
    const ggml_tensor * bias,
    ggml_tensor * dst,
    bool apply_gelu_erf);

bool ggml_cuda_conv_2d_transpose_k2s2_bias_gelu(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * kernel,
    const ggml_tensor * input,
    const ggml_tensor * bias,
    ggml_tensor * dst);
