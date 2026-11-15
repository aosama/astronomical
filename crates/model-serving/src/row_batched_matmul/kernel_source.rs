//! Metal source of the row-batched 4-bit affine matmul.
//!
//! The arithmetic is ported from TensorFold's `lane_qmm` (tag v0.3.5.1,
//! `src/tensorfold/kernels/qwen/dense/v1/lane_qmm.py`, Apache-2.0, Copyright
//! 2026 TensorFold contributors). Stock MLX on M5 either runs its one-row
//! kernel once per row (up to 12 rows) or a 32-row matrix-unit tile with no
//! split along the input, which measured 3.85x one row for 16 rows over the
//! Qwen3.8 27B's projections and 5 to 6x on its narrow 5,120-wide outputs
//! (`cargo measure-k-row-verify-cost`). This kernel reads each weight group
//! once for all rows and splits long inputs across simdgroups.
//!
//! The matrix units multiply the raw 4-bit codes (`uint4b_format`, which is
//! MLX's packed nibble order) by the bf16 rows; the affine dequantization is
//! applied after the product, per group: `y += scale * (x . q) + bias * sum(x)`.
//! The per-row group sums of `x` come from a separate small kernel so that
//! every projection reading the same rows can share them.

/// Brings in the M5 matrix-unit tensor operations.
pub(crate) const ROW_BATCHED_MATMUL_HEADER: &str = r#"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"#;

/// Inputs: `X` [M, K] bf16 rows, `row_dimensions` [M, MP] int32.
/// Output: `XS` [K / GS, MP] float32, the sum of each row over each group;
/// rows from M up to the padded MP are zero.
pub(crate) const GROUP_SUM_SOURCE: &str = r#"
  const int M = row_dimensions[0];
  const int MP = row_dimensions[1];
  const uint row = thread_position_in_grid.y;
  const uint group = thread_position_in_grid.x;
  if (group >= K / GS || int(row) >= MP) return;
  float group_sum = 0.0f;
  if (int(row) < M) {
    for (int value_index = 0; value_index < GS; value_index++) {
      group_sum += float(X[row * K + group * GS + value_index]);
    }
  }
  XS[group * MP + row] = group_sum;
"#;

/// Inputs: `X` [M, K] bf16, `XS` [K / GS, MP] float32, `Wq` [N, K / 8]
/// uint32 codes regrouped by `tile_packed_weight`, `SBt` [K / GS, N, 2] bf16 scale
/// and bias pairs, `row_dimensions` [M, MP] int32.
/// Output: `Y` [M, N] bf16.
///
/// Template integers: `TMR` 16-row blocks per threadgroup, `N`, `K`, `NT`
/// output columns per simdgroup, `SK` input slices (one simdgroup each),
/// `GS` group size, `EDGE` 1 when the last row block passes MP.
///
/// Input slices are added in slice order so a column's bits depend on the
/// weight shape and `SK`, never on the row count: a 16-row verify gives the
/// same row values as 16 one-row passes through this kernel.
pub(crate) const ROW_BATCHED_MATMUL_SOURCE: &str = r#"
  const ushort lane = thread_index_in_simdgroup;
  const ushort slice_index = simdgroup_index_in_threadgroup;
  const short quad_index = lane >> 2;
  const short fragment_row = (quad_index & 4) | ((lane >> 1) & 3);
  const short fragment_column = ((quad_index & 2) | (lane & 1)) * 4;
  const int M = row_dimensions[0];
  const int MP = row_dimensions[1];
  constexpr int GROUP_COUNT = K / GS;
  constexpr int FRAGMENT_COUNT = NT / 16;
  const int first_column = threadgroup_position_in_grid.x * NT;
  const int first_row = threadgroup_position_in_grid.y * 16 * TMR;
  const int first_group = (slice_index * GROUP_COUNT) / SK;
  const int end_group = ((slice_index + 1) * GROUP_COUNT) / SK;

  constexpr auto descriptor = matmul2d_descriptor(
      16 * TMR, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroup> operation;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> rows_tensor(
      (device bfloat*)X + (int64_t)first_row * K, dextents<int32_t, 2>(K, M - first_row));

  float accumulated[TMR][FRAGMENT_COUNT * 8];
  for (int block = 0; block < TMR; block++)
    for (int index = 0; index < FRAGMENT_COUNT * 8; index++) accumulated[block][index] = 0.0f;
  const device uint4* scale_bias_words = (const device uint4*)SBt;
  bool column_in_range[FRAGMENT_COUNT];
  for (int column_fragment = 0; column_fragment < FRAGMENT_COUNT; column_fragment++)
    column_in_range[column_fragment] = first_column + column_fragment * 16 + fragment_column < N;

  for (int group = first_group; group < end_group; group++) {
    float scales[FRAGMENT_COUNT][4];
    float biases[FRAGMENT_COUNT][4];
    for (int column_fragment = 0; column_fragment < FRAGMENT_COUNT; column_fragment++) {
      const uint4 packed_pairs = column_in_range[column_fragment]
          ? scale_bias_words[(group * N + first_column + column_fragment * 16 + fragment_column) / 4]
          : uint4(0);
      const vec<bfloat, 8> pairs = as_type<vec<bfloat, 8>>(packed_pairs);
      for (int pair = 0; pair < 4; pair++) {
        scales[column_fragment][pair] = float(pairs[2 * pair]);
        biases[column_fragment][pair] = float(pairs[2 * pair + 1]);
      }
    }
    auto rows_slice = rows_tensor.slice(group * GS, 0);
    // Tiled weights hold each column tile's group as one contiguous NT x GS
    // block of codes (`tile_packed_weight`), so this read is one 1 KB run
    // instead of NT 32-byte pieces a whole row apart.
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> codes_slice(
        (device uchar*)Wq
            + (int64_t)(threadgroup_position_in_grid.x * GROUP_COUNT + group) * (NT * GS / 2),
        dextents<int32_t, 2>(GS, NT));
    auto products = operation.template get_destination_cooperative_tensor<
        decltype(rows_slice), decltype(codes_slice), float>();
    operation.run(rows_slice, codes_slice, products);
    for (int block = 0; block < TMR; block++) {
      const bool block_live = !EDGE || first_row + block * 16 < MP;
      const float low_sum = block_live
          ? XS[group * MP + first_row + block * 16 + fragment_row] : 0.0f;
      const float high_sum = block_live
          ? XS[group * MP + first_row + block * 16 + fragment_row + 8] : 0.0f;
      for (int column_fragment = 0; column_fragment < FRAGMENT_COUNT; column_fragment++)
        for (int row_half = 0; row_half < 2; row_half++)
          for (int pair = 0; pair < 4; pair++) {
            const int index = column_fragment * 8 + row_half * 4 + pair;
            accumulated[block][index] = fma(
                scales[column_fragment][pair],
                products[block * FRAGMENT_COUNT * 8 + index],
                fma(biases[column_fragment][pair], row_half ? high_sum : low_sum,
                    accumulated[block][index]));
          }
    }
  }

  threadgroup float slice_partials[(SK > 1 ? SK - 1 : 1) * FRAGMENT_COUNT * 8 * 32];
  for (int block = 0; block < TMR; block++) {
    if (SK > 1) {
      if (slice_index > 0)
        for (int index = 0; index < FRAGMENT_COUNT * 8; index++)
          slice_partials[((slice_index - 1) * FRAGMENT_COUNT * 8 + index) * 32 + lane] =
              accumulated[block][index];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (slice_index == 0)
        for (int other_slice = 1; other_slice < SK; other_slice++)
          for (int index = 0; index < FRAGMENT_COUNT * 8; index++)
            accumulated[block][index] +=
                slice_partials[((other_slice - 1) * FRAGMENT_COUNT * 8 + index) * 32 + lane];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (slice_index == 0)
      for (int column_fragment = 0; column_fragment < FRAGMENT_COUNT; column_fragment++)
        for (int row_half = 0; row_half < 2; row_half++) {
          const int row = first_row + block * 16 + fragment_row + 8 * row_half;
          const int column = first_column + column_fragment * 16 + fragment_column;
          if (row < M && column < N)
            for (int pair = 0; pair < 4; pair++)
              Y[row * N + column + pair] =
                  static_cast<bfloat>(accumulated[block][column_fragment * 8 + row_half * 4 + pair]);
        }
  }
"#;
