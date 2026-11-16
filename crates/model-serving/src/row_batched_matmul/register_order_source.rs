//! Metal source of the register-ordered 4-bit projection and of the one-time
//! conversion from MLX's packed layout into it.
//!
//! Ported from TensorFold's `tf_lane_reg` and `tf_lane_order` (release 1.2.0,
//! `zig/kernels/metal/core/lane_projection.metal` and `lane_order.metal`,
//! Apache-2.0, Copyright 2026 TensorFold contributors), which its own engine
//! runs for every 4-bit projection of the Qwen 27B on chips with matrix units.
//!
//! The kernel in `kernel_source` hands the matrix unit a `uint4b_format`
//! tensor view of MLX's weight and lets the tensor operation load and widen
//! the codes. Here each lane instead loads its own sixteen bytes of codes per
//! group as two `uint4` reads, already in the order the operation's right
//! input registers hold them, and widens them itself: a 4-bit code `c` placed
//! in the low mantissa bits of the fp16 number 1024 reads as `1024 + c`, so
//! one OR and one subtraction give `c` exactly as fp16. The next group's
//! codes, scale and bias pairs and row sums are loaded before the current
//! group's product runs, which hides the memory latency the tensor view
//! exposed. Scale and bias of a column share one 32-bit word, and a lane reads
//! four columns' words in one `uint4`.

/// The tensor operations, the per-group load a lane holds, and its fetch.
pub(crate) const REGISTER_ORDERED_MATMUL_HEADER: &str = r#"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// One lane's codes, its two scale-and-bias quads and its two row sums for one group.
struct RegisterGroupLoad { uint4 low_codes, high_codes, low_pairs, high_pairs; float low_sum, high_sum; };

inline RegisterGroupLoad fetch_register_group(
    const device uint4* lane_codes, const device uint4* pair_quads, const device float* row_sums,
    int group, int output_width, int column, int row, int row_stride) {
  return {lane_codes[(size_t)group * 64], lane_codes[(size_t)group * 64 + 32],
          pair_quads[((size_t)group * output_width + column) / 4],
          pair_quads[((size_t)group * output_width + column + 16) / 4],
          row_sums[group * row_stride + row], row_sums[group * row_stride + row + 8]};
}
"#;

/// Inputs: `X` [M, K] bf16, `XS` [K / 64, MP] float32 group sums, `W` the
/// register-ordered codes, `SB` [K / 64, N] uint32 (scale in the low half,
/// bias in the high half), `row_dimensions` [M, MP] int32. Output `Y` [M, N]
/// bf16. Template integers: `N`, `K`, `SK` input slices (one simdgroup each).
///
/// A threadgroup covers one 32-column tile and one 16-row block. Slice `s`
/// takes the contiguous groups `[s G / SK, (s + 1) G / SK)` and the slices are
/// added in slice order, so a row's bits depend on the weight shape and `SK`
/// only, never on how many rows share the call.
pub(crate) const REGISTER_ORDERED_MATMUL_SOURCE: &str = r#"
  const int first_row = int(threadgroup_position_in_grid.y) * 16;
  const int block_row_count = row_dimensions[0] - first_row;
  if (block_row_count <= 0) return;
  const int row_stride = row_dimensions[1];
  const int lane = int(thread_index_in_simdgroup);
  const int slice_index = int(simdgroup_index_in_threadgroup);
  const int M = min(block_row_count, 16);
  const int tile = int(threadgroup_position_in_grid.x);
  const int first_column = tile * 32;
  constexpr int GROUP_COUNT = K / 64;
  const int column_base = 4 * (lane & 1) + 8 * ((lane >> 3) & 1);
  const int row_base = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1);
  const int first_group = (slice_index * GROUP_COUNT) / SK;
  const int last_group = ((slice_index + 1) * GROUP_COUNT) / SK;

  constexpr auto descriptor = matmul2d_descriptor(16, 32, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroup> operation;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> rows_tensor(
      (device bfloat*)X + (size_t)first_row * K, dextents<int32_t, 2>(K, M));
  auto first_rows_slice = rows_tensor.slice(0, 0);
  auto codes = operation.template get_right_input_cooperative_tensor<bfloat, half, float>();
  auto products = operation.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(first_rows_slice)>, metal::remove_addrspace_t<decltype(codes)>, float>();
  thread half2* code_pairs = (thread half2*)&codes[0];
  const device uint4* lane_codes = (const device uint4*)W + (size_t)tile * GROUP_COUNT * 64 + lane;
  const device uint4* pair_quads = (const device uint4*)SB;
  const device float* row_sums = XS + first_row;
  const half2 fp16_code_offset = half2(1024.0h);
  float accumulated[16];
  for (int index = 0; index < 16; index++) accumulated[index] = 0.0f;
  RegisterGroupLoad next = {};
  if (first_group < last_group)
    next = fetch_register_group(lane_codes, pair_quads, row_sums, first_group, N,
                                first_column + column_base, row_base, row_stride);
  for (int group = first_group; group < last_group; group++) {
    const RegisterGroupLoad current = next;
    for (int word_index = 0; word_index < 8; word_index++) {
      // 1024 + code as fp16 bits for codes (0,4) (1,5) (2,6) (3,7) of a word;
      // subtracting 1024 is exact.
      const uint word = word_index < 4 ? current.low_codes[word_index] : current.high_codes[word_index - 4];
      const uint low_nibbles = word & 0x0F0F0F0Fu, high_nibbles = (word >> 4) & 0x0F0F0F0Fu;
      const int element = 16 * (word_index & 1) + 2 * (word_index >> 1);
      code_pairs[element] = as_type<half2>((low_nibbles & 0x00FF00FFu) | 0x64006400u) - fp16_code_offset;
      code_pairs[element + 1] = as_type<half2>((high_nibbles & 0x00FF00FFu) | 0x64006400u) - fp16_code_offset;
      code_pairs[element + 8] = as_type<half2>(((low_nibbles >> 8) & 0x00FF00FFu) | 0x64006400u) - fp16_code_offset;
      code_pairs[element + 9] = as_type<half2>(((high_nibbles >> 8) & 0x00FF00FFu) | 0x64006400u) - fp16_code_offset;
    }
    if (group + 1 < last_group)
      next = fetch_register_group(lane_codes, pair_quads, row_sums, group + 1, N,
                                  first_column + column_base, row_base, row_stride);
    auto rows_slice = rows_tensor.slice(group * 64, 0);
    operation.run(rows_slice, codes, products);
    for (int pair = 0; pair < 4; pair++) {
      const float low_scale = as_type<float>(current.low_pairs[pair] << 16);
      const float low_bias = as_type<float>(current.low_pairs[pair] & 0xFFFF0000u);
      const float high_scale = as_type<float>(current.high_pairs[pair] << 16);
      const float high_bias = as_type<float>(current.high_pairs[pair] & 0xFFFF0000u);
      accumulated[pair] = fma(low_scale, products[pair], fma(low_bias, current.low_sum, accumulated[pair]));
      accumulated[4 + pair] = fma(low_scale, products[4 + pair], fma(low_bias, current.high_sum, accumulated[4 + pair]));
      accumulated[8 + pair] = fma(high_scale, products[8 + pair], fma(high_bias, current.low_sum, accumulated[8 + pair]));
      accumulated[12 + pair] = fma(high_scale, products[12 + pair], fma(high_bias, current.high_sum, accumulated[12 + pair]));
    }
  }
  threadgroup float slice_partials[(SK > 1 ? SK - 1 : 1) * 16 * 32];
  if (SK > 1) {
    if (slice_index > 0)
      for (int index = 0; index < 16; index++) slice_partials[((slice_index - 1) * 16 + index) * 32 + lane] = accumulated[index];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slice_index == 0)
      for (int other_slice = 1; other_slice < SK; other_slice++)
        for (int index = 0; index < 16; index++) accumulated[index] += slice_partials[((other_slice - 1) * 16 + index) * 32 + lane];
  }
  if (slice_index == 0)
    for (int index = 0; index < 16; index++) {
      const int row = row_base + 8 * ((index >> 2) & 1);
      const int column = first_column + column_base + 16 * (index >> 3) + (index & 3);
      if (row < M) Y[(size_t)(first_row + row) * N + column] = static_cast<bfloat>(accumulated[index]);
    }
"#;

/// Inputs: MLX's packed `Wq` [N, K / 8] uint32 and its bf16 `S` scales and `B`
/// biases [N, K / 64]. Outputs: `R` the codes in register order, one 256-word
/// block per 32-column tile and group (tile-major), and `P` [K / 64, N] uint32
/// scale-and-bias words. One threadgroup of 256 threads converts one block.
///
/// Block word `t` holds, for lane `(t >> 2) & 31`, the eight codes that lane
/// unpacks from its `t & 3`-th word of the low (t < 128) or high half; each
/// comes from two 16-bit runs of four codes, `a` and `c`, two words apart in
/// the column's tile-order run, and is laid out a0 a2 c0 c2 a1 a3 c1 c3, low
/// nibble first, which is the order the unpack above reads back.
pub(crate) const REGISTER_ORDER_CONVERSION_SOURCE: &str = r#"
  constexpr int GROUP_COUNT = K / 64;
  const uint t = thread_position_in_threadgroup.x;
  const uint block = threadgroup_position_in_grid.x;
  const uint tile = block / GROUP_COUNT, group = block % GROUP_COUNT;
  const uint lane = (t >> 2) & 31, word_slot = 4 * (t >> 7) + (t & 3);
  const uint column = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1) + 8 * (word_slot >> 1);
  const uint source_word = ((lane >> 3) & 1) + 4 * (word_slot & 1);
  const uint shift = 16 * (lane & 1);
  const device uint32_t* column_words = Wq + (size_t)(tile * 32 + column) * (K / 8) + group * 8;
  const uint a = (column_words[source_word] >> shift) & 0xFFFFu;
  const uint c = (column_words[source_word + 2] >> shift) & 0xFFFFu;
  R[(size_t)block * 256 + t] = (a & 0xFu) | ((a >> 4) & 0xF0u) | ((c & 0xFu) << 8) | ((c << 4) & 0xF000u)
      | (((a >> 4) & 0xFu) << 16) | (((a >> 8) & 0xF0u) << 16) | (((c >> 4) & 0xFu) << 24) | (((c >> 8) & 0xF0u) << 24);
  if (t < 32) {
    const uint output_column = tile * 32 + t;
    const uint scale_bits = as_type<ushort>(S[(size_t)output_column * GROUP_COUNT + group]);
    const uint bias_bits = as_type<ushort>(B[(size_t)output_column * GROUP_COUNT + group]);
    P[(size_t)group * N + output_column] = scale_bits | (bias_bits << 16);
  }
"#;

/// The conversion run backwards, for the few readers that need MLX's layout
/// back (dequantizing for Neural Engine programs, reading output-head rows).
/// Inputs: `R` and `P` as the conversion writes them for a weight `N` wide,
/// `tile_ids` [T] int32. Outputs, for the 32 columns of each listed tile in
/// list order: `Wq` [32 T, K / 8] uint32 and `S`, `B` [32 T, K / 64] bf16.
/// One threadgroup of 256 threads restores one tile and group: each thread
/// unpacks its register word into the two 16-bit runs it came from, parks
/// them in threadgroup memory, and after the barrier writes one MLX word.
pub(crate) const REGISTER_ORDER_RESTORATION_SOURCE: &str = r#"
  constexpr int GROUP_COUNT = K / 64;
  const uint t = thread_position_in_threadgroup.x;
  const uint block = threadgroup_position_in_grid.x;
  const uint listed_tile = block / GROUP_COUNT, group = block % GROUP_COUNT;
  const uint tile = uint(tile_ids[listed_tile]);
  const uint lane = (t >> 2) & 31, word_slot = 4 * (t >> 7) + (t & 3);
  const uint column = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1) + 8 * (word_slot >> 1);
  const uint source_word = ((lane >> 3) & 1) + 4 * (word_slot & 1);
  const uint half_index = lane & 1;
  // Codes in the register word, low nibble first: a0 a2 c0 c2 a1 a3 c1 c3.
  const uint ordered = R[((size_t)tile * GROUP_COUNT + group) * 256 + t];
  const uint a = (ordered & 0xFu) | (((ordered >> 16) & 0xFu) << 4) | (((ordered >> 4) & 0xFu) << 8) | (((ordered >> 20) & 0xFu) << 12);
  const uint c = ((ordered >> 8) & 0xFu) | (((ordered >> 24) & 0xFu) << 4) | (((ordered >> 12) & 0xFu) << 8) | (((ordered >> 28) & 0xFu) << 12);
  threadgroup ushort column_halves[512];
  column_halves[(column * 8 + source_word) * 2 + half_index] = ushort(a);
  column_halves[(column * 8 + source_word + 2) * 2 + half_index] = ushort(c);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint restored_column = t / 8, restored_word = t % 8;
  const uint output_row = listed_tile * 32 + restored_column;
  Wq[(size_t)output_row * (K / 8) + group * 8 + restored_word] =
      uint(column_halves[t * 2]) | (uint(column_halves[t * 2 + 1]) << 16);
  if (t < 32) {
    const uint pair_word = P[(size_t)group * N + tile * 32 + t];
    S[(size_t)(listed_tile * 32 + t) * GROUP_COUNT + group] = as_type<bfloat>(ushort(pair_word & 0xFFFFu));
    B[(size_t)(listed_tile * 32 + t) * GROUP_COUNT + group] = as_type<bfloat>(ushort(pair_word >> 16));
  }
"#;
