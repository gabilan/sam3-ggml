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

// ── EWI-1963 kernel sweep: vectorized variants ──────────────────────────────
//
// Both kernels above are one element per thread and decompose the flat thread
// index with *runtime-divisor 64-bit* divides and modulos: win_part pays four
// (gid/ne0, /ne1, /ne2, i3/npx) plus their remainders, win_unpart another four
// (gid/ne0, /ne1, i1/w, i2/w) -- for a 4-byte payload that is otherwise a
// straight copy. At [1024,72,72,1] -> [1024,24,24,9] this is 5.3 M threads per
// call, ~250 instructions each against ~8 bytes of traffic.
//
// These variants move VEC=4 consecutive floats per thread. The innermost axis
// (i0) is contiguous in the source (nb00 == sizeof(float)) and in the
// destination of win_part / the source of win_unpart, so a float4 covers one
// row's four neighbouring columns; the outer index math is then paid once per
// four elements, and in 32-bit arithmetic with fast_div_modulo instead of the
// 64-bit subroutines.
//
// The mapping from element to address is unchanged -- each element is the same
// load or the same zero fill of the same byte -- so the output is
// bit-identical to the scalar kernels. Both are gated on their own alignment
// and range preconditions with the scalar kernel as the fallback.

static __global__ void win_part_f32_vec4(
        const char * __restrict__ src0, float * __restrict__ dst,
        const int64_t nb01, const int64_t nb02,
        const int64_t ne01, const int64_t ne02,
        const uint3 v_ne0v, const uint3 v_ne1, const uint3 v_ne2, const uint3 v_npx,
        const int64_t ne1, const int64_t ne2,
        const int32_t w, const int64_t total_v) {
    const int64_t gid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_v) {
        return;
    }

    // fast_div_modulo returns <n/divisor, n%divisor> in <.x, .y> -- the inner
    // index of each level is the remainder, and the quotient is the numerator
    // of the next level down.
    const uint2 q0 = fast_div_modulo((uint32_t) gid, v_ne0v);   // rem1, i0v
    const uint2 q1 = fast_div_modulo(q0.x, v_ne1);              // rem2, i1
    const uint2 q2 = fast_div_modulo(q1.x, v_ne2);              // i3,   i2
    const uint2 q3 = fast_div_modulo(q2.x, v_npx);              // py,   px

    const int64_t i0v     = q0.y;
    const int64_t i1      = q1.y;
    const int64_t i2      = q2.y;
    const int64_t i3      = q2.x;
    const int64_t src_col = (int64_t) q3.y * w + i1;            // px*w + i1
    const int64_t src_row = (int64_t) q3.x * w + i2;            // py*w + i2
    const int64_t dst_v   = ((i3 * ne2 + i2) * ne1 + i1) * (int64_t) v_ne0v.z + i0v;

    float4 out;
    if (src_col >= ne01 || src_row >= ne02) {
        out = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        out = *reinterpret_cast<const float4 *>(src0 + src_row * nb02 + src_col * nb01 + i0v * 16);
    }
    reinterpret_cast<float4 *>(dst)[dst_v] = out;
}

static __global__ void win_unpart_f32_vec4(
        const char * __restrict__ src0, float * __restrict__ dst,
        const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const uint3 v_ne0v, const uint3 v_ne1, const uint3 v_w,
        const int64_t ne1, const int64_t npx, const int64_t total_v) {
    const int64_t gid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_v) {
        return;
    }

    // <div, mod> as above.
    const uint2 q0 = fast_div_modulo((uint32_t) gid, v_ne0v);   // rem, i0v
    const uint2 q1 = fast_div_modulo(q0.x, v_ne1);              // i2,  i1
    const uint2 q2 = fast_div_modulo(q1.y, v_w);                // px,  wx
    const uint2 q3 = fast_div_modulo(q1.x, v_w);                // py,  wy

    const int64_t i0v     = q0.y;
    const int64_t i1      = q1.y;
    const int64_t i2      = q1.x;
    const int64_t win_idx = (int64_t) q3.x * npx + q2.x;        // py*npx + px
    const int64_t src_off = win_idx * nb03 + (int64_t) q3.y * nb02 + (int64_t) q2.y * nb01 + i0v * 16;
    const int64_t dst_v   = ((i2 * ne1 + i1) * (int64_t) v_ne0v.z + i0v);

    reinterpret_cast<float4 *>(dst)[dst_v] =
        *reinterpret_cast<const float4 *>(src0 + src_off);
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

    // Vectorized path: four contiguous columns per thread. Gated on the source's
    // innermost axis being packed floats whose outer strides keep every vector
    // base 16-byte aligned, and on every divisor fitting the 32-bit fastdiv.
    const int64_t ne0 = dst->ne[0];
    const bool vec_ok =
        ne0 >= 4 && ne0 % 4 == 0 && total % 4 == 0 &&
        src0->nb[0] == (size_t) sizeof(float) &&
        src0->nb[1] % 16 == 0 && src0->nb[2] % 16 == 0 &&
        ((uintptr_t) src0->data) % 16 == 0 && ((uintptr_t) dst->data) % 16 == 0 &&
        npx > 0 && npy > 0 &&
        (uint64_t) (total / 4) <= 0xffffffffull &&
        (uint64_t) (ne0 / 4) <= 0xffffffffull &&
        (uint64_t) dst->ne[1] <= 0xffffffffull && (uint64_t) dst->ne[2] <= 0xffffffffull &&
        (uint64_t) npx <= 0xffffffffull;

    if (vec_ok) {
        const int64_t total_v = total / 4;
        const int     vblocks = (int) ((total_v + CUDA_WIN_PART_BLOCK_SIZE - 1) / CUDA_WIN_PART_BLOCK_SIZE);
        win_part_f32_vec4<<<vblocks, CUDA_WIN_PART_BLOCK_SIZE, 0, ctx.stream()>>>(
                (const char *) src0->data, (float *) dst->data,
                src0->nb[1], src0->nb[2],
                src0->ne[1], src0->ne[2],
                init_fastdiv_values((uint32_t) (ne0 / 4)),
                init_fastdiv_values((uint32_t) dst->ne[1]),
                init_fastdiv_values((uint32_t) dst->ne[2]),
                init_fastdiv_values((uint32_t) npx),
                dst->ne[1], dst->ne[2],
                w, total_v);
        return;
    }

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

    // Vectorized path: see win_part_f32_vec4. Same preconditions, plus nb03 for
    // the window axis.
    const int64_t ne0 = dst->ne[0];
    const bool vec_ok =
        ne0 >= 4 && ne0 % 4 == 0 && total % 4 == 0 &&
        src0->nb[0] == (size_t) sizeof(float) &&
        src0->nb[1] % 16 == 0 && src0->nb[2] % 16 == 0 && src0->nb[3] % 16 == 0 &&
        ((uintptr_t) src0->data) % 16 == 0 && ((uintptr_t) dst->data) % 16 == 0 &&
        npx > 0 && w > 0 &&
        (uint64_t) (total / 4) <= 0xffffffffull &&
        (uint64_t) (ne0 / 4) <= 0xffffffffull &&
        (uint64_t) dst->ne[1] <= 0xffffffffull && (uint64_t) w <= 0xffffffffull;

    if (vec_ok) {
        const int64_t total_v = total / 4;
        const int     vblocks = (int) ((total_v + CUDA_WIN_PART_BLOCK_SIZE - 1) / CUDA_WIN_PART_BLOCK_SIZE);
        win_unpart_f32_vec4<<<vblocks, CUDA_WIN_PART_BLOCK_SIZE, 0, ctx.stream()>>>(
                (const char *) src0->data, (float *) dst->data,
                src0->nb[1], src0->nb[2], src0->nb[3],
                init_fastdiv_values((uint32_t) (ne0 / 4)),
                init_fastdiv_values((uint32_t) dst->ne[1]),
                init_fastdiv_values((uint32_t) w),
                dst->ne[1], npx, total_v);
        return;
    }

    win_unpart_f32<<<blocks, CUDA_WIN_PART_BLOCK_SIZE, 0, ctx.stream()>>>(
            (const char *) src0->data, (float *) dst->data,
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            dst->ne[0],  dst->ne[1],  dst->ne[2],
            npx, w);
}
