// One thread per `(assignment, output row)` pair.
//
// The thread reads the packed row of the assignment's expert word by word,
// unpacks every code of each 32-code group straight from registers (a code
// may straddle two words at 3 bits, exactly as `codebook::unpack_codes_from_packed_row`
// describes), looks the code up in the thread-local level table, and
// accumulates the product with the activation element in float32. The row
// scale multiplies the finished dot product once, mirroring the materializing
// decoder's `level * scale` weight followed by the matrix product; the only
// difference is that no bfloat16 weight is ever rounded or written.
//
// Template parameters (all compile-time constants, so every loop bound and
// array size is static):
// * `OutputT` - the activation dtype the projection is written in.
// * `bits` - codebook width, 2, 3, or 4.
// * `input_width` - contracted width (the decoded row length), a multiple of 32.
// * `words_per_row` - packed `uint32` words per expert row (`input_width * bits / 32`).
// * `output_rows` - rows per expert (the projection's output width).
// * `shared_hidden_row` - true when every assignment multiplies the same
//   single activation row (one decode token), false when `hidden` carries one
//   row per assignment.
//
// The level table and the code unpacking come from the header shared with the
// page decoder.
//
// @header astro_format_jangtq2_codebook_fragments.h
// @input hidden OutputT
// @input expert_indices int32_t|uint32_t may-be-small
// @input packed_weights uint32_t may-be-small
// @input row_scales float16_t may-be-small
// @input codebook_levels float may-be-small
// @output projected OutputT
// @template OutputT typename
// @template bits int
// @template input_width int
// @template words_per_row int
// @template output_rows int
// @template shared_hidden_row bool
// @compile-with OutputT=bfloat16_t bits=2 input_width=4096 words_per_row=256 output_rows=1024 shared_hidden_row=true
// @compile-with OutputT=float bits=3 input_width=32 words_per_row=3 output_rows=64 shared_hidden_row=false
// @compile-with OutputT=float16_t bits=4 input_width=1024 words_per_row=128 output_rows=3072 shared_hidden_row=true
// @contract direct_mlx::fused_gathered_projection
// @probe fused_jangtq2_gathered_projection
  const uint global_index = thread_position_in_grid.x;
  const uint assignment_index = global_index / (uint)output_rows;
  const uint output_row = global_index % (uint)output_rows;
  const uint expert_index = (uint)expert_indices[assignment_index];
  const uint hidden_row = shared_hidden_row ? 0u : assignment_index;
  const size_t packed_row_base =
      ((size_t)expert_index * (size_t)output_rows + (size_t)output_row) * (size_t)words_per_row;
  const size_t hidden_row_base = (size_t)hidden_row * (size_t)input_width;

  ASTRO_FORMAT_JANGTQ2_LOAD_LEVEL_TABLE

  float dot = 0.0f;
  constexpr int group_count = input_width / 32;
  constexpr uint32_t code_mask = (1u << bits) - 1u;
  for (int group_index = 0; group_index < group_count; ++group_index) {
    uint32_t group_words[bits];
    for (int word_index = 0; word_index < bits; ++word_index) {
      group_words[word_index] = packed_weights[packed_row_base + group_index * bits + word_index];
    }
    const int element_base = group_index * 32;
    for (int code_index = 0; code_index < 32; ++code_index) {
  ASTRO_FORMAT_JANGTQ2_UNPACK_CODE

      dot += level_table[code] * float(hidden[hidden_row_base + element_base + code_index]);
    }
  }
  const float row_scale =
      float(row_scales[(size_t)expert_index * (size_t)output_rows + (size_t)output_row]);
  projected[global_index] = (OutputT)(dot * row_scale);
