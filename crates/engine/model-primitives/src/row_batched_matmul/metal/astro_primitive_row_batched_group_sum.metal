// The per-group sums of the rows a row-batched projection reads: X [M, K]
// bf16 in, XS [K / GS, MP] float32 out, the sum of each row over each group;
// rows from M up to the padded MP are zero. Every projection reading the same
// rows shares them, since the affine dequantization is applied after the
// product: y += scale * (x . q) + bias * sum(x).
//
// @input X bfloat16_t
// @input row_dimensions int32_t always-small
// @output XS float
// @template K int
// @template GS int
// @compile-with K=5120 GS=64
// @compile-with K=64 GS=64
// @contract direct_mlx::row_batched_matmul
// @probe row_batched_matmul

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
