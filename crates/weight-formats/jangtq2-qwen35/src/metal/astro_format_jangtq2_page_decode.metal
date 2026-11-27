// One thread per `(row, 32-code group)`: the thread loads the group's
// `bits` packed words, unpacks all 32 codes from registers (a code may
// straddle two words at 3 bits, exactly as
// `codebook::unpack_codes_from_packed_row` describes), multiplies each
// level by the row's float16 scale in float32, and writes 32 bfloat16
// weights. Nothing but the packed page, the row scales, the tiny level table,
// and the output ever exists, which is why page admission prices only the
// packed source and the decoded output.
//
// Template parameters (compile-time constants):
// * `OutputT` - the decoded weight dtype (bfloat16 in serving).
// * `bits` - codebook width, 2, 3, or 4.
// * `input_width` - decoded row length, a multiple of 32.
// * `words_per_row` - packed `uint32` words per row (`input_width * bits / 32`).
//
// The level table and the code unpacking come from the header shared with the
// fused gathered projection.
//
// @header astro_format_jangtq2_codebook_fragments.h
// @input packed_weights uint32_t may-be-small
// @input row_scales float16_t may-be-small
// @input codebook_levels float may-be-small
// @output decoded OutputT
// @template OutputT typename
// @template bits int
// @template input_width int
// @template words_per_row int
// @compile-with OutputT=bfloat16_t bits=2 input_width=4096 words_per_row=256
// @compile-with OutputT=bfloat16_t bits=3 input_width=32 words_per_row=3
// @compile-with OutputT=bfloat16_t bits=4 input_width=1024 words_per_row=128
// @contract direct_mlx::mlx_decode
// @probe none, no capability probe exists for it yet
  constexpr int group_count = input_width / 32;
  constexpr uint32_t code_mask = (1u << bits) - 1u;
  const uint global_index = thread_position_in_grid.x;
  const uint row_index = global_index / (uint)group_count;
  const uint group_index = global_index % (uint)group_count;
  const size_t packed_group_base =
      (size_t)row_index * (size_t)words_per_row + (size_t)group_index * (size_t)bits;
  uint32_t group_words[bits];
  for (int word_index = 0; word_index < bits; ++word_index) {
    group_words[word_index] = packed_weights[packed_group_base + word_index];
  }
  ASTRO_FORMAT_JANGTQ2_LOAD_LEVEL_TABLE

  const float row_scale = float(row_scales[row_index]);
  const size_t output_base =
      (size_t)row_index * (size_t)input_width + (size_t)group_index * 32;
  for (int code_index = 0; code_index < 32; ++code_index) {
  ASTRO_FORMAT_JANGTQ2_UNPACK_CODE

    decoded[output_base + code_index] = (OutputT)(level_table[code] * row_scale);
  }
