// The register-order conversion run backwards, for the few readers that need
// MLX's layout back (dequantizing for Neural Engine programs, reading
// output-head rows): R and P as the conversion writes them for a weight N
// wide, and tile_ids [T] int32, in; for the 32 columns of each listed tile in
// list order, Wq [32 T, K / 8] uint32 and S, B [32 T, K / 64] bf16 out. One
// threadgroup of 256 threads restores one tile and group: each thread unpacks
// its register word into the two 16-bit runs it came from, parks them in
// threadgroup memory, and after the barrier writes one MLX word.
//
// @input R uint32_t
// @input P uint32_t
// @input tile_ids int32_t may-be-small
// @output Wq uint32_t
// @output S bfloat16_t
// @output B bfloat16_t
// @template N int
// @template K int
// @compile-with N=5120 K=5120
// @compile-with N=32 K=64
// @contract direct_mlx::register_ordered_matmul
// @probe none, it rebuilds MLX's layout for readers off the serving path; its contract compares the round trip with the original weight

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
