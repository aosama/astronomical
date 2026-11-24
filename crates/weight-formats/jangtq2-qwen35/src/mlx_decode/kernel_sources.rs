//! Raw Metal source for the jangtq2 page decode kernel.
//!
//! Split from the decoder so each file stays inside the source-size budget.
//! The source is assembled from string constants, compiled once per inference
//! thread; the level table and the code unpacking come from
//! `codebook_kernel_fragments`, shared with the fused gathered projection.

use crate::codebook_kernel_fragments::{CODE_UNPACK_SOURCE, LEVEL_TABLE_SOURCE};

const PAGE_DECODE_KERNEL_SOURCE_OPENING: &str = r#"
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
"#;
const PAGE_DECODE_KERNEL_SOURCE_BEFORE_UNPACK: &str = r#"
  const float row_scale = float(row_scales[row_index]);
  const size_t output_base =
      (size_t)row_index * (size_t)input_width + (size_t)group_index * 32;
  for (int code_index = 0; code_index < 32; ++code_index) {
"#;
const PAGE_DECODE_KERNEL_SOURCE_CLOSING: &str = r#"
    decoded[output_base + code_index] = (OutputT)(level_table[code] * row_scale);
  }
"#;

/// One thread per `(row, 32-code group)`: the thread loads the group's
/// `bits` packed words, unpacks all 32 codes from registers (a code may
/// straddle two words at 3 bits, exactly as
/// `codebook::unpack_codes_from_packed_row` describes), multiplies each
/// level by the row's float16 scale in float32, and writes 32 bfloat16
/// weights. Nothing but the packed page, the row scales, the tiny level table,
/// and the output ever exists, which is why page admission prices only the
/// packed source and the decoded output.
///
/// Template parameters (compile-time constants):
/// * `OutputT` — the decoded weight dtype (bfloat16 in serving).
/// * `bits` — codebook width, 2, 3, or 4.
/// * `input_width` — decoded row length, a multiple of 32.
/// * `words_per_row` — packed `uint32` words per row (`input_width · bits / 32`).
pub(super) fn page_decode_kernel_source() -> String {
    [
        PAGE_DECODE_KERNEL_SOURCE_OPENING,
        LEVEL_TABLE_SOURCE,
        PAGE_DECODE_KERNEL_SOURCE_BEFORE_UNPACK,
        CODE_UNPACK_SOURCE,
        PAGE_DECODE_KERNEL_SOURCE_CLOSING,
    ]
    .concat()
}
