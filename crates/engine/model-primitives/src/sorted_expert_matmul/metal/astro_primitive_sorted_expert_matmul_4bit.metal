// Astronomical's sorted-expert 4-bit affine matmul over rows already sorted
// by expert: X [M, K] bf16, MLX's packed Wq [E, N, K / 8] uint32 with its
// bf16 scales S and biases B [E, N, K / 64], each expert's first row
// first_rows [E] and the row count M [1]; Y [M, N] bf16. The arithmetic and
// layout are described in the shared body file.
//
// @header astro_primitive_sorted_expert_matmul.h
// @body astro_primitive_sorted_expert_matmul_body.metal
// @input X bfloat16_t
// @input Wq uint32_t
// @input S bfloat16_t
// @input B bfloat16_t
// @input first_rows int32_t may-be-small
// @input row_count int32_t always-small
// @output Y bfloat16_t
// @template N int
// @template K int
// @template E int
// @template BM int
// @template WN int
// @template STAGE_ROWS int
// @template TILE_SCHEDULE int
// @compile-with N=512 K=2048 E=256 BM=96 WN=4 STAGE_ROWS=1 TILE_SCHEDULE=1
// @compile-with N=128 K=128 E=4 BM=64 WN=2 STAGE_ROWS=0 TILE_SCHEDULE=0
// @contract direct_mlx::sorted_expert_matmul
// @probe sorted_expert_matmul
// The sorted rows are X's own rows: sorted row r is X's row r.
#define SOURCE_ROW(sorted_row) long(sorted_row)
