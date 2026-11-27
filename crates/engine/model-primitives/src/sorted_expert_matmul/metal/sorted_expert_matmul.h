// Helpers of Astronomical's sorted-expert 4-bit matmul: the M5 matrix-unit
// fragments (TensorFold's nax.h, M5 path, Apache-2.0), MLX's dequantization
// and MLX's expert row-tile schedule. MLX's custom-kernel API places this
// header ahead of the generated kernel functions; see kernel_source.rs.

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

namespace astro_sorted_expert {
using namespace metal;

// One simdgroup's 16x16 fragment: 8 values a lane.
template <typename T>
using frag = vec<T, 8>;

// Lane l holds rows home.y and home.y + 8, columns home.x .. home.x + 3 of
// every fragment: the M5 operand layout.
inline short2 frag_home(ushort l) {
  return short2(short((l & 8) + ((l & 1) << 2)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}

// The 16x16 block at (r, c) of a row-major matrix with leading dimension ld.
template <typename T, typename P>
inline void frag_get(thread frag<T>& f, P p, int ld, int r, int c, short2 home) {
  const P q = p + (r + home.y) * ld + (c + home.x);
  _Pragma("clang loop unroll(full)")
  for (short e = 0; e < 8; e++) {
    f[e] = T(q[(e >> 2) * 8 * ld + (e & 3)]);
  }
}

// frag_get from device rows, zeros at rows nr and past, which are not read.
template <typename T>
inline void frag_get_live_rows(thread frag<T>& f, const device T* p, int ld, int r, int c, short2 home, int nr) {
  _Pragma("clang loop unroll(full)")
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.y + (e >> 2) * 8;
    f[e] = rr < nr ? p[rr * ld + c + home.x + (e & 3)] : T(0);
  }
}

// frag_get from device rows in two 8-byte loads and no bound: rows past the
// expert's own (the next expert's, never past M) are read, multiply into
// accumulators that are never stored, and change no stored value, as in MLX.
inline void frag_get_device(thread frag<bfloat>& f, const device bfloat* p, int ld, int r, int c, short2 home) {
  const device bfloat* q = p + (r + home.y) * ld + (c + home.x);
  f = frag<bfloat>(*(const device vec<bfloat, 4>*)q, *(const device vec<bfloat, 4>*)(q + 8 * ld));
}

// The fragment's rows below nr into y (leading dimension ld), as bf16.
template <typename O>
inline void frag_put_rows_below(thread const frag<float>& f, device O* p, int ld, int r, int c, short2 home, int nr) {
  _Pragma("clang loop unroll(full)")
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.y + (e >> 2) * 8;
    if (rr < nr) {
      p[rr * ld + c + home.x + (e & 3)] = O(f[e]);
    }
  }
}

// (lo | hi) += a * (b0 | b1): one 16x32x16 multiply-accumulate on the matrix
// unit, with the relaxed precision MLX's own kernels ask for.
template <bool TA, bool TB, typename C, typename A, typename B>
inline void mma_16x32(thread frag<C>& lo, thread frag<C>& hi, thread const frag<A>& a, thread const frag<B>& b0,
                      thread const frag<B>& b1) {
  using namespace mpp::tensor_ops;
  constexpr auto shape = matmul2d_descriptor(16, 32, 16, TA, TB, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<shape, execution_simdgroup> op;
  auto left = op.template get_left_input_cooperative_tensor<A, B, C>();
  auto right = op.template get_right_input_cooperative_tensor<A, B, C>();
  auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,
                                                            metal::remove_addrspace_t<decltype(right)>, C>();
  _Pragma("clang loop unroll(full)")
  for (short e = 0; e < 8; e++) {
    left[e] = a[e];
    right[e] = b0[e];
    right[8 + e] = b1[e];
    acc[e] = lo[e];
    acc[8 + e] = hi[e];
  }
  op.run(left, right, acc);
  _Pragma("clang loop unroll(full)")
  for (short e = 0; e < 8; e++) {
    lo[e] = acc[e];
    hi[e] = acc[8 + e];
  }
}

// 16 packed bytes of one 64-group's row as 32 bf16 weights, exactly as MLX
// dequantizes them: s * q + b per nibble, the high one as (s / 16) * (q & 0xf0)
// so no shift is needed.
// The 16 bytes are read as one 128-bit load and the 32 weights written as
// four 16-byte stores: the same values as byte by byte, in a quarter of the
// instructions.
inline void dequantize_16_bytes(const device uchar* w, bfloat scale, bfloat bias, threadgroup bfloat* out) {
  const float s = float(scale), b = float(bias), s_hi = s / 16.0f;
  const vec<uchar, 16> packed_bytes = *(const device vec<uchar, 16>*)w;
  _Pragma("clang loop unroll(full)")
  for (short quarter = 0; quarter < 4; quarter++) {
    vec<bfloat, 8> weights;
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 4; i++) {
      const uchar q = packed_bytes[4 * quarter + i];
      weights[2 * i] = bfloat(s * (q & 0x0f) + b);
      weights[2 * i + 1] = bfloat(s_hi * (q & 0xf0) + b);
    }
    *(threadgroup vec<bfloat, 8>*)(out + 8 * quarter) = weights;
  }
}

// frag_get from a threadgroup block in two 8-byte loads (TensorFold's
// frag_get_tg): a lane's four columns are contiguous, so (c + home.x) * 2 and
// ld * 2 being multiples of 8 make each half one aligned load.
inline void frag_get_threadgroup(thread frag<bfloat>& f, const threadgroup bfloat* p, int ld, int r, int c,
                                 short2 home) {
  const threadgroup bfloat* q = p + (r + home.y) * ld + (c + home.x);
  f = frag<bfloat>(*(const threadgroup vec<bfloat, 4>*)q, *(const threadgroup vec<bfloat, 4>*)(q + 8 * ld));
}

// The tile'th BM-row tile over the experts in order (MLX's schedule_row_tile,
// as TensorFold writes it): its expert, first row and row count; false past
// the last tile. A popular expert's rows spread over several threadgroups.
// `P` is a device or a constant pointer: MLX hands a custom kernel a small
// input (a few experts' first rows) in constant memory.
template <int BM, typename P>
inline bool expert_tile(P first_rows, int experts, int total, int tile, uint lane,
                        thread int& expert, thread int& row, thread int& rows) {
  int before = 0;
  for (int base = 0; base < experts; base += 32) {
    const int e = base + int(lane);
    const int first = e < experts ? first_rows[e] : total;
    const int last = e + 1 < experts ? first_rows[e + 1] : total;
    const int count = (last - first + BM - 1) / BM;
    const int upto = before + simd_prefix_inclusive_sum(count);
    const int owner = simd_sum(int(upto <= tile));
    if (owner < 32) {
      expert = base + owner;
      row = simd_shuffle(first, ushort(owner)) + (tile - simd_shuffle(upto - count, ushort(owner))) * BM;
      rows = min(BM, simd_shuffle(last, ushort(owner)) - row);
      return true;
    }
    before = simd_shuffle(upto, ushort(31));
  }
  return false;
}

}  // namespace astro_sorted_expert
