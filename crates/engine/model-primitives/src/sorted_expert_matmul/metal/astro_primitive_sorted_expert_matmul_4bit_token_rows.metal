// The sorted-expert 4-bit affine matmul reading each sorted row from its
// token's own row through the sort order: X [tokens, K] bf16 holds one row
// per token, order [M] is the sort of the token-major assignments, and TOPK
// is the experts per token. The other inputs are the sorted-row kernel's.
//
// @header astro_primitive_sorted_expert_matmul.h
// @body astro_primitive_sorted_expert_matmul_body.metal
// @input X bfloat16_t
// @input Wq uint32_t
// @input S bfloat16_t
// @input B bfloat16_t
// @input first_rows int32_t may-be-small
// @input order uint32_t|int32_t may-be-small
// @input row_count int32_t always-small
// @output Y bfloat16_t
// @template N int
// @template K int
// @template E int
// @template BM int
// @template WN int
// @template STAGE_ROWS int
// @template TILE_SCHEDULE int
// @template TOPK int
// @compile-with N=512 K=2048 E=256 BM=96 WN=4 STAGE_ROWS=1 TILE_SCHEDULE=1 TOPK=8
// @compile-with N=128 K=128 E=4 BM=96 WN=4 STAGE_ROWS=1 TILE_SCHEDULE=0 TOPK=2
// @contract direct_mlx::sorted_expert_matmul
// @probe sorted_expert_matmul
// The sorted rows are read through the sort: X holds one row per token, and
// sorted row r is token order[r] / TOPK's row, so no sorted copy of the
// token rows is ever made. Only the staged layouts read through it.
#define SOURCE_ROW(sorted_row) long(uint(order[sorted_row]) / uint(TOPK))
