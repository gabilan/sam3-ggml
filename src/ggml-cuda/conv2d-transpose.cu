#include "conv2d-transpose.cuh"
#include "convert.cuh"

// Generic path: every output element loops Cin × Kh × Kw (with stride alignment skips).
template <typename kernel_t>
static __global__ void conv2d_transpose_kernel(const float * __restrict__ input,
                                               const kernel_t * __restrict__ kernel,
                                               float * __restrict__ output,
                                               const int in_w,
                                               const int in_h,
                                               const int out_w,
                                               const int out_h,
                                               const int kernel_w,
                                               const int kernel_h,
                                               const int stride,
                                               const int c_in,
                                               const int c_out,
                                               const int batches) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int total_elements = out_w * out_h * c_out * batches;

    if (global_idx >= total_elements) {
        return;
    }

    const int out_x_idx = global_idx % out_w;
    const int out_y_idx = (global_idx / out_w) % out_h;
    const int c_idx     = (global_idx / (out_w * out_h)) % c_out;
    const int n_idx     = global_idx / (out_w * out_h * c_out);

    float accumulator = 0;

    for (int c_in_idx = 0; c_in_idx < c_in; c_in_idx++) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            int in_y = out_y_idx - kh;
            if (in_y < 0 || in_y % stride) {
                continue;
            }
            in_y /= stride;
            if (in_y >= in_h) {
                continue;
            }

            for (int kw = 0; kw < kernel_w; ++kw) {
                int in_x = out_x_idx - kw;
                if (in_x < 0 || in_x % stride) {
                    continue;
                }
                in_x /= stride;
                if (in_x >= in_w) {
                    continue;
                }

                const int input_idx = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + (in_w) *in_y + in_x;
                const int kernel_idx =
                    (kernel_h * kernel_w * c_out) * c_in_idx + (kernel_h * kernel_w) * c_idx + (kernel_w) *kh + kw;

                float    input_val = input[input_idx];
                kernel_t kern_val  = kernel[kernel_idx];

                accumulator += input_val * ggml_cuda_cast<float>(kern_val);
            }
        }
    }

    output[(out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + (out_w) *out_y_idx + out_x_idx] = accumulator;
}

static __device__ __forceinline__ float sam3_op_gelu_erf(float x) {
    constexpr float SQRT_2_INV = 0.7071067811865475f;
    return 0.5f * x * (1.0f + erff(x * SQRT_2_INV));
}

// Scatter GEMM result [spatial, 4*c_out] (col-major) → NCHW 2× upsample + bias/gelu.
template <bool APPLY_BIAS, bool APPLY_GELU>
static __global__ void conv2d_transpose_k2s2_scatter_kernel(
        const float * __restrict__ gemm_out,
        const float * __restrict__ bias,
        float * __restrict__ output,
        const int in_w,
        const int in_h,
        const int out_w,
        const int out_h,
        const int c_out,
        const int batches) {
    const int spatial = in_w * in_h;
    const int total   = spatial * c_out * batches;
    const int idx     = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int co    = idx % c_out;
    const int spat  = (idx / c_out) % spatial;
    const int n_idx = idx / (c_out * spatial);

    const int in_y = spat / in_w;
    const int in_x = spat - in_y * in_w;
    const int out_y0 = in_y << 1;
    const int out_x0 = in_x << 1;
    const int out_plane = out_w * out_h;
    const int ldc = spatial; // col-major gemm: (s, j) at s + j*ldc

    const float * col = gemm_out + (size_t) n_idx * (size_t) spatial * (size_t) (4 * c_out)
                        + (size_t) spat;
    float v0 = col[(size_t) (4 * co + 0) * ldc];
    float v1 = col[(size_t) (4 * co + 1) * ldc];
    float v2 = col[(size_t) (4 * co + 2) * ldc];
    float v3 = col[(size_t) (4 * co + 3) * ldc];

    if constexpr (APPLY_BIAS) {
        const float b = bias[co];
        v0 += b;
        v1 += b;
        v2 += b;
        v3 += b;
    }
    if constexpr (APPLY_GELU) {
        v0 = sam3_op_gelu_erf(v0);
        v1 = sam3_op_gelu_erf(v1);
        v2 = sam3_op_gelu_erf(v2);
        v3 = sam3_op_gelu_erf(v3);
    }

    float * o = output + (size_t) n_idx * (size_t) out_plane * (size_t) c_out
                + (size_t) co * (size_t) out_plane;
    o[(size_t) out_y0 * out_w + out_x0]         = v0;
    o[(size_t) out_y0 * out_w + out_x0 + 1]     = v1;
    o[(size_t) (out_y0 + 1) * out_w + out_x0]     = v2;
    o[(size_t) (out_y0 + 1) * out_w + out_x0 + 1] = v3;
}

// k2s2 as GEMM: input NCHW contiguous is col-major (spatial × c_in);
// weight [2,2,c_out,c_in] is col-major ((4*c_out) × c_in).
// tmp = input * weight^T  →  (spatial × 4*c_out), then scatter to 2× grid.
template <bool APPLY_BIAS, bool APPLY_GELU>
static bool launch_k2s2_gemm_scatter(
        ggml_backend_cuda_context & ctx,
        const float * input,
        const void * kernel,
        ggml_type kernel_type,
        const float * bias,
        float * output,
        int in_w, int in_h, int out_w, int out_h,
        int c_in, int c_out, int batches) {
    const int spatial = in_w * in_h;
    const int n_gemm_cols = 4 * c_out;
    const size_t tmp_elems = (size_t) batches * (size_t) spatial * (size_t) n_gemm_cols;

    ggml_cuda_pool_alloc<float> tmp_alloc(ctx.pool(), tmp_elems);
    float * tmp = tmp_alloc.get();

    // cuBLAS rejects mixed f32-A / f16-B GemmEx on this path; promote weights.
    ggml_cuda_pool_alloc<float> w_f32_alloc(ctx.pool());
    const float * w_f32 = nullptr;
    const size_t w_ne = (size_t) n_gemm_cols * (size_t) c_in;
    if (kernel_type == GGML_TYPE_F16) {
        w_f32 = w_f32_alloc.alloc(w_ne);
        const to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
        GGML_ASSERT(to_fp32 != nullptr);
        to_fp32(kernel, w_f32_alloc.get(), w_ne, ctx.stream());
        w_f32 = w_f32_alloc.get();
    } else if (kernel_type == GGML_TYPE_F32) {
        w_f32 = (const float *) kernel;
    } else {
        return false;
    }

    cudaStream_t st = ctx.stream();
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), st));

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    for (int n = 0; n < batches; ++n) {
        const float * a = input + (size_t) n * (size_t) c_in * (size_t) spatial;
        float *       c = tmp   + (size_t) n * (size_t) n_gemm_cols * (size_t) spatial;
        CUBLAS_CHECK(cublasSgemm(
            ctx.cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_T,
            spatial, n_gemm_cols, c_in,
            &alpha,
            a, spatial,
            w_f32, n_gemm_cols,
            &beta,
            c, spatial));
    }

    const int total = spatial * c_out * batches;
    const int block = 256;
    const int grid  = (total + block - 1) / block;
    conv2d_transpose_k2s2_scatter_kernel<APPLY_BIAS, APPLY_GELU><<<grid, block, 0, st>>>(
        tmp, bias, output, in_w, in_h, out_w, out_h, c_out, batches);
    return true;
}

bool ggml_cuda_conv_2d_transpose_k2s2_bias(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * kernel,
        const ggml_tensor * input,
        const ggml_tensor * bias,
        ggml_tensor * dst,
        bool apply_gelu_erf) {
    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(bias->type == GGML_TYPE_F32);

    const int input_w      = input->ne[0];
    const int input_h      = input->ne[1];
    const int output_w     = dst->ne[0];
    const int output_h     = dst->ne[1];
    const int channels_in  = input->ne[2];
    const int channels_out = kernel->ne[2];
    const int kernel_w     = kernel->ne[0];
    const int kernel_h     = kernel->ne[1];
    const int batches      = input->ne[3];

    if (!(kernel_w == 2 && kernel_h == 2 &&
          output_w == input_w * 2 && output_h == input_h * 2 &&
          channels_in == kernel->ne[3] &&
          channels_in > 0 && channels_out > 0 &&
          (int) ggml_nelements(bias) == channels_out &&
          ggml_is_contiguous(input) && ggml_is_contiguous(kernel) &&
          ggml_is_contiguous(dst))) {
        return false;
    }

    const float * bias_data = (const float *) bias->data;
    const float * in_data   = (const float *) input->data;
    float *       out_data  = (float *) dst->data;

    if (apply_gelu_erf) {
        return launch_k2s2_gemm_scatter<true, true>(
            ctx, in_data, kernel->data, kernel->type, bias_data, out_data,
            input_w, input_h, output_w, output_h, channels_in, channels_out, batches);
    }
    return launch_k2s2_gemm_scatter<true, false>(
        ctx, in_data, kernel->data, kernel->type, bias_data, out_data,
        input_w, input_h, output_w, output_h, channels_in, channels_out, batches);
}

bool ggml_cuda_conv_2d_transpose_k2s2_bias_gelu(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * kernel,
        const ggml_tensor * input,
        const ggml_tensor * bias,
        ggml_tensor * dst) {
    return ggml_cuda_conv_2d_transpose_k2s2_bias(ctx, kernel, input, bias, dst, true);
}

void ggml_cuda_conv_2d_transpose_p0(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const float * input_data  = (const float *) input->data;
    float *       output_data = (float *) dst->data;
    const void *  kernel_data = kernel->data;

    const int input_w      = input->ne[0];
    const int input_h      = input->ne[1];
    const int output_w     = dst->ne[0];
    const int output_h     = dst->ne[1];
    const int channels_in  = input->ne[2];
    const int channels_out = kernel->ne[2];
    const int kernel_w     = kernel->ne[0];
    const int kernel_h     = kernel->ne[1];
    const int stride       = dst->op_params[0];
    const int batches      = input->ne[3];

    GGML_ASSERT(channels_in == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    GGML_ASSERT(ggml_is_contiguous(input));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const bool k2s2 = (kernel_w == 2 && kernel_h == 2 && stride == 2 &&
                       output_w == input_w * 2 && output_h == input_h * 2 &&
                       channels_in > 0 && channels_out > 0);

    if (k2s2) {
        launch_k2s2_gemm_scatter<false, false>(
            ctx, input_data, kernel_data, kernel->type, nullptr, output_data,
            input_w, input_h, output_w, output_h, channels_in, channels_out, batches);
        return;
    }

    cudaStream_t st = ctx.stream();
    const int total  = output_w * output_h * channels_out * batches;
    const int blocks = (total + CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE;

    if (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_kernel<half><<<blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, st>>>(
            input_data, (const half *) kernel_data, output_data, input_w, input_h, output_w, output_h, kernel_w,
            kernel_h, stride, channels_in, channels_out, batches);
    } else {
        conv2d_transpose_kernel<float><<<blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, st>>>(
            input_data, (const float *) kernel_data, output_data, input_w, input_h, output_w, output_h, kernel_w,
            kernel_h, stride, channels_in, channels_out, batches);
    }
}
