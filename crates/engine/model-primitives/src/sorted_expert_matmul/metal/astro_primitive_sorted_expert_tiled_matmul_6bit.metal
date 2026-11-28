// The tiled sorted-expert 6-bit affine matmul over rows already sorted by
// expert, which a prompt chunk's routed down projection takes: X [M, K]
// bf16, MLX's 6-bit packed Wq [E, N, K * 6 / 32] uint32 (four weights in
// every three bytes) with bf16 scales S and biases B [E, N, K / 64], tiles /
// tile_count from astro_primitive_sorted_expert_tile_list; Y [M, N] bf16
// with MLX's bits.
//
// @header astro_primitive_sorted_expert_matmul.h astro_primitive_sorted_expert_tiled_matmul.h
// @body astro_primitive_sorted_expert_tiled_matmul_body.metal
// @input X bfloat16_t
// @input Wq uint32_t
// @input S bfloat16_t
// @input B bfloat16_t
// @input tiles uint32_t
// @input tile_count uint32_t always-small
// @output Y bfloat16_t
// @template N int
// @template K int
// @template GX int
// @compile-with N=2048 K=512 GX=32
// @compile-with N=128 K=128 GX=0
// @contract direct_mlx::sorted_expert_tiled_matmul
// @probe sorted_expert_tiled_matmul_6bit
#define SOURCE_ROW(sorted_row) long(sorted_row)
#define PAIRED 0
#define WEIGHT_BITS 6
#define WEIGHT_PACKED(reads_up) Wq
#define WEIGHT_SCALES(reads_up) S
#define WEIGHT_BIASES(reads_up) B
