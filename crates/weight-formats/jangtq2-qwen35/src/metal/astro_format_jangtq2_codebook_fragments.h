// Statements shared by the two kernels that read packed jangtq2 codes: the
// page decoder and the fused gathered projection (issue #2340).
//
// Both kernels must turn a packed 32-code group into the same codes and the
// same levels, or the fused projection stops matching the materializing
// decoder bit for bit. Writing the unpacking once means it cannot drift
// between them; the golden-vector and fused projection parity tests prove the
// assembled kernels. These are macros, not functions, because the statements
// read the kernel's template parameters and buffers where they are used and
// MLX passes a small buffer in a different address space than a large one: a
// function would need a template for each of those.

// Loads the codebook's levels into a thread-local float32 table, indexed by
// code. Expects `bits` and `codebook_levels` in scope.
#define ASTRO_FORMAT_JANGTQ2_LOAD_LEVEL_TABLE \
  float level_table[1 << bits]; \
  for (int code = 0; code < (1 << bits); ++code) { \
    level_table[code] = float(codebook_levels[code]); \
  }

// Reads code `code_index` of the 32-code group held in `group_words`: a code
// may straddle two words at 3 bits, exactly as
// `codebook::unpack_codes_from_packed_row` describes. Expects `bits`,
// `code_mask`, `group_words` and `code_index` in scope, and defines `code`.
#define ASTRO_FORMAT_JANGTQ2_UNPACK_CODE \
    const int start_bit = code_index * bits; \
    const int word_index = start_bit >> 5; \
    const int bit_offset = start_bit & 31; \
    uint32_t spread_value = group_words[word_index] >> bit_offset; \
    if (bit_offset + bits > 32) { \
      spread_value |= group_words[word_index + 1] << (32 - bit_offset); \
    } \
    const uint32_t code = spread_value & code_mask;
