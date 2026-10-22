//! Raw Metal source for the fused jangtq2 gathered projection kernel.
//!
//! Split from the kernel owner and launch logic in the parent so each file
//! stays inside the source-size budget; the source is a pure string constant
//! consumed once by `FusedJangtq2GatheredProjectionKernel::new`.

/// One thread per `(assignment, output row)` pair.
///
/// The thread reads the packed row of the assignment's expert word by word,
/// unpacks every code of each 32-code group straight from registers (a code
/// may straddle two words at 3 bits, exactly as `codebook::unpack_codes_from_packed_row`
/// describes), looks the code up in the thread-local level table, and
/// accumulates the product with the activation element in float32. The row
/// scale multiplies the finished dot product once, mirroring the materializing
/// decoder's `level · scale` weight followed by the matrix product; the only
/// difference is that no bfloat16 weight is ever rounded or written.
///
/// Template parameters (all compile-time constants, so every loop bound and
/// array size is static):
/// * `OutputT` — the activation dtype the projection is written in.
/// * `bits` — codebook width, 2, 3, or 4.
/// * `input_width` — contracted width (the decoded row length), a multiple of 32.
/// * `words_per_row` — packed `uint32` words per expert row (`input_width · bits / 32`).
/// * `output_rows` — rows per expert (the projection's output width).
/// * `shared_hidden_row` — true when every assignment multiplies the same
///   single activation row (one decode token), false when `hidden` carries one
///   row per assignment.
pub(super) const FUSED_GATHERED_PROJECTION_KERNEL_SOURCE: &str = r#"
  const uint global_index = thread_position_in_grid.x;
  const uint assignment_index = global_index / (uint)output_rows;
  const uint output_row = global_index % (uint)output_rows;
  const uint expert_index = (uint)expert_indices[assignment_index];
  const uint hidden_row = shared_hidden_row ? 0u : assignment_index;
  const size_t packed_row_base =
      ((size_t)expert_index * (size_t)output_rows + (size_t)output_row) * (size_t)words_per_row;
  const size_t hidden_row_base = (size_t)hidden_row * (size_t)input_width;

  float level_table[1 << bits];
  for (int code = 0; code < (1 << bits); ++code) {
    level_table[code] = float(codebook_levels[code]);
  }

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
      const int start_bit = code_index * bits;
      const int word_index = start_bit >> 5;
      const int bit_offset = start_bit & 31;
      uint32_t spread_value = group_words[word_index] >> bit_offset;
      if (bit_offset + bits > 32) {
        spread_value |= group_words[word_index + 1] << (32 - bit_offset);
      }
      const uint32_t code = spread_value & code_mask;
      dot += level_table[code] * float(hidden[hidden_row_base + element_base + code_index]);
    }
  }
  const float row_scale =
      float(row_scales[(size_t)expert_index * (size_t)output_rows + (size_t)output_row]);
  projected[global_index] = (OutputT)(dot * row_scale);
"#;
