// A prompt chunk's routed gate and up projections and their SwiGLU in one
// tiled sorted-expert kernel: X [tokens, K] bf16 holds one row per token,
// order [M] is the sort of the token-major assignments (TOPK experts a
// token), each sorted row is read from its token's row through it, and
// tiles / tile_count come from astro_primitive_sorted_expert_tile_list.
// Wq, S, B are the gate stack and Wu, Su, Bu the up stack (MLX's packed
// [E, N, K / 8] uint32 with bf16 [E, N, K / 64] scales and biases). Y [M, N]
// bf16 is MLX's compiled gate * sigmoid(gate) * up of the two projections,
// bit for bit, without writing gate or up.
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
// @probe sorted_expert_tiled_matmul
#define SOURCE_ROW(sorted_row) long(uint(order[sorted_row]) / uint(TOPK))
#define PAIRED 1
#define WEIGHT_PACKED(reads_up) ((reads_up) ? Wu : Wq)
#define WEIGHT_SCALES(reads_up) ((reads_up) ? Su : S)
#define WEIGHT_BIASES(reads_up) ((reads_up) ? Bu : B)
