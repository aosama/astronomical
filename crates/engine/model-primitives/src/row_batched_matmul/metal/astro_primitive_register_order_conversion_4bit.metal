// The one-time conversion of MLX's packed 4-bit layout into register order:
// Wq [N, K / 8] uint32 and its bf16 scales S and biases B [N, K / 64] in;
// R the codes in register order, one 256-word block per 32-column tile and
// group (tile-major), and P [K / 64, N] uint32 scale-and-bias words out. One
// threadgroup of 256 threads converts one block.
//
// Block word t holds, for lane (t >> 2) & 31, the eight codes that lane
// unpacks from its t & 3-th word of the low (t < 128) or high half; each
// comes from two 16-bit runs of four codes, a and c, two words apart in the
// column's tile-order run, and is laid out a0 a2 c0 c2 a1 a3 c1 c3, low
// nibble first, which is the order the register-ordered matmul reads back.
//
// @input Wq uint32_t
// @input S bfloat16_t
// @input B bfloat16_t
// @output R uint32_t
// @output P uint32_t
// @template N int
// @template K int
// @compile-with N=5120 K=5120
// @compile-with N=32 K=64
// @contract direct_mlx::register_ordered_matmul
// @probe row_batched_matmul

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
