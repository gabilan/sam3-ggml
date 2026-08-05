// CUDA kernels for ggml_win_part / ggml_win_unpart (SAM3 ViT windowed attention).
// Logic mirrors Metal kernel_win_part_f32 / kernel_win_unpart_f32 and the CPU
// reference in ggml-cpu/ops.cpp — F32 only, matching ggml_win_part's assert.

#include "win-part.cuh"

#define CUDA_WIN_PART_BLOCK_SIZE 256

static __global__ void win_part_f32(
        const char * __restrict__ src0, float * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t nb00, const int64_t nb01, const int64_t nb02,
        const int64_t ne0,  const int64_t ne1,  const int64_t ne2,
        const int32_t npx,  const int32_t npy,  const int32_t w) {
    const int64_t np    = (int64_t) npx * npy;
    const int64_t total = ne0 * ne1 * ne2 * np;
    const int64_t gid   = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) {
        return;
    }

    const int64_t i0   = gid % ne0;
    const int64_t rem1 = gid / ne0;
    const int64_t i1   = rem1 % ne1;
    const int64_t rem2 = rem1 / ne1;
    const int64_t i2   = rem2 % ne2;
    const int64_t i3   = rem2 / ne2;

    const int64_t py = i3 / npx;
    const int64_t px = i3 % npx;

    const int64_t src_col = px * w + i1;
    const int64_t src_row = py * w + i2;

    const int64_t dst_idx = i3 * ne2 * ne1 * ne0 + i2 * ne1 * ne0 + i1 * ne0 + i0;

    if (src_col >= ne01 || src_row >= ne02) {
        dst[dst_idx] = 0.0f;
    } else {
        const float * src_ptr = (const float *) (src0 + src_row * nb02 + src_col * nb01 + i0 * nb00);
        dst[dst_idx] = *src_ptr;
    }
}

static __global__ void win_unpart_f32(
        const char * __restrict__ src0, float * __restrict__ dst,
        const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t ne0,  const int64_t ne1,  const int64_t ne2,
        const int32_t npx,  const int32_t w) {
    const int64_t total = ne0 * ne1 * ne2;
    const int64_t gid   = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) {
        return;
    }

    const int64_t i0  = gid % ne0;
    const int64_t rem = gid / ne0;
    const int64_t i1  = rem % ne1;
    const int64_t i2  = rem / ne1;

    const int64_t px = i1 / w;
    const int64_t py = i2 / w;
    const int64_t wx = i1 % w;
    const int64_t wy = i2 % w;
    const int64_t win_idx = py * npx + px;

    const float * src_ptr = (const float *) (src0 + win_idx * nb03 + wy * nb02 + wx * nb01 + i0 * nb00);
    const int64_t dst_idx = i2 * ne1 * ne0 + i1 * ne0 + i0;
    dst[dst_idx] = *src_ptr;
}

void ggml_cuda_op_win_part(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const int32_t npx = ((const int32_t *) (dst->op_params))[0];
    const int32_t npy = ((const int32_t *) (dst->op_params))[1];
    const int32_t w   = ((const int32_t *) (dst->op_params))[2];

    const int64_t np    = (int64_t) npx * npy;
    const int64_t total = dst->ne[0] * dst->ne[1] * dst->ne[2] * np;
    const int     blocks = (int) ((total + CUDA_WIN_PART_BLOCK_SIZE - 1) / CUDA_WIN_PART_BLOCK_SIZE);

    win_part_f32<<<blocks, CUDA_WIN_PART_BLOCK_SIZE, 0, ctx.stream()>>>(
            (const char *) src0->data, (float *) dst->data,
            src0->ne[0], src0->ne[1], src0->ne[2],
            src0->nb[0], src0->nb[1], src0->nb[2],
            dst->ne[0],  dst->ne[1],  dst->ne[2],
            npx, npy, w);
}

void ggml_cuda_op_win_unpart(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const int32_t w = ((const int32_t *) (dst->op_params))[0];
    // Same npx formula as Metal / CPU
    const int64_t w0  = dst->ne[1];
    const int     px  = (int) ((w - (w0 % w)) % w);
    const int32_t npx = (int32_t) ((px + w0) / w);

    const int64_t total  = dst->ne[0] * dst->ne[1] * dst->ne[2];
    const int     blocks = (int) ((total + CUDA_WIN_PART_BLOCK_SIZE - 1) / CUDA_WIN_PART_BLOCK_SIZE);

    win_unpart_f32<<<blocks, CUDA_WIN_PART_BLOCK_SIZE, 0, ctx.stream()>>>(
            (const char *) src0->data, (float *) dst->data,
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            dst->ne[0],  dst->ne[1],  dst->ne[2],
            npx, w);
}
