// The 6-bit twin of astro_primitive_sorted_expert_gate_up_swiglu_4bit_token_rows:
// a prompt chunk's routed gate and up projections and their SwiGLU in one
// tiled sorted-expert kernel, reading each sorted row from its token's row of
// X [tokens, K] bf16 through order [M] (TOPK experts a token), with tiles /
// tile_count from astro_primitive_sorted_expert_tile_list. Wq, S, B are the
// gate stack and Wu, Su, Bu the up stack: MLX's 6-bit packed
// [E, N, K * 6 / 32] uint32 (four weights in every three bytes) with bf16
// [E, N, K / 64] scales and biases. Y [M, N] bf16 is MLX's compiled
// gate * sigmoid(gate) * up of the two projections, bit for bit.
//
// @header astro_primitive_sorted_expert_matmul.h astro_primitive_sorted_expert_tiled_matmul.h
// @body astro_primitive_sorted_expert_tiled_matmul_body.metal
// @input X bfloat16_t
// @input Wq uint32_t
// @input S bfloat16_t
// @input B bfloat16_t
// @input Wu uint32_t
// @input Su bfloat16_t
// @input Bu bfloat16_t
// @input tiles uint32_t
// @input tile_count uint32_t always-small
// @input order uint32_t|int32_t may-be-small
// @output Y bfloat16_t
// @template N int
// @template K int
// @template GX int
// @template TOPK int
// @compile-with N=512 K=2048 GX=32 TOPK=8
// @compile-with N=64 K=128 GX=0 TOPK=2
// @contract direct_mlx::sorted_expert_tiled_matmul
// @probe sorted_expert_tiled_matmul_6bit
#define SOURCE_ROW(sorted_row) long(uint(order[sorted_row]) / uint(TOPK))
#define PAIRED 1
#define WEIGHT_BITS 6
#define WEIGHT_PACKED(reads_up) ((reads_up) ? Wu : Wq)
#define WEIGHT_SCALES(reads_up) ((reads_up) ? Su : S)
#define WEIGHT_BIASES(reads_up) ((reads_up) ? Bu : B)
