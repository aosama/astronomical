//! The jangtq2 cubic scalar codebook and its MLX-compatible bitstream.
//!
//! Two operations matter on the serving side: evaluating the level for a code
//! arithmetically (the reference kernels do this in ALU rather than gathering
//! from a table, keeping decode cheap at every width) and unpacking the
//! LSB-first bitstream into codes. Packing is encode-side only and stays in
//! the JANG toolchain; this module never needs to produce packed words.
//!
//! Bitstream layout (byte-identical to MLX affine packing — an invariant the
//! reference format asserts in its own tests): within one row, code `j` starts
//! at bit `j · bits` of a continuous LSB-first stream, and every 32 codes
//! occupy exactly `bits` `uint32` words. A code may straddle two words; the
//! row length is always a multiple of 32 codes.

/// The bit widths the jangtq2 cubic codebook family defines. Wider widths
/// degenerate to a uniform step, which is exactly affine quantization and
/// therefore never shipped in the codebook format.
pub const SUPPORTED_CODEBOOK_BIT_WIDTHS: &[u32] = &[2, 3, 4];

/// `(bit_width, alpha, beta)` for `level(q) = u · (alpha + beta · u²)`.
///
/// The constants are the reference `CUBIC_PARAMS` verbatim (jjang-ai/jangq,
/// `jangh/format.py`) — measured against N(0,1): 2-bit is exactly Lloyd-Max,
/// 3-bit is +0.2% MSE, 4-bit is +2%. Do not re-fit them: bundles in the wild
/// were encoded with exactly these values.
const CUBIC_PARAMETERS: &[(u32, f64, f64)] = &[
    (2, 0.892_999_999_999_999_9, 0.050_65),
    (3, 0.471_249_999_999_999_95, 0.011_599_999_999_999_997),
    (4, 0.240_5, 0.002_100_000_000_000_002),
];

/// Number of quantization codes per packed `uint32` word boundary: the row is
/// processed in 32-code groups, so every group spans exactly `bits` words.
const CODES_PER_GROUP: u32 = 32;

/// Returns whether the codebook family covers the bit width.
#[must_use]
pub fn is_supported_bit_width(bit_width: u32) -> bool {
    SUPPORTED_CODEBOOK_BIT_WIDTHS.contains(&bit_width)
}

/// Returns the `(alpha, beta)` cubic parameters for the bit width, or `None`
/// when the width is outside the codebook family.
#[must_use]
pub fn cubic_parameters(bit_width: u32) -> Option<(f64, f64)> {
    CUBIC_PARAMETERS
        .iter()
        .find(|&&(supported_bit_width, _, _)| supported_bit_width == bit_width)
        .map(|&(_, alpha, beta)| (alpha, beta))
}

/// Evaluates the level for one code arithmetically — the decode hot path.
///
/// Computation runs in `f64` and narrows to `f32` once, matching the
/// reference (which builds `float64` tables and casts to `float32`), so the
/// arithmetic and tabulated level values agree bit-for-bit.
#[must_use]
pub fn codebook_level_value(bit_width: u32, code: u32) -> Option<f32> {
    if !is_supported_bit_width(bit_width) || code >= 1_u32 << bit_width {
        return None;
    }
    let (alpha, beta) = cubic_parameters(bit_width)?;
    let centered_code_index = f64::from(code) - f64::from((1_u32 << bit_width) - 1) / 2.0;
    Some((centered_code_index * (alpha + beta * centered_code_index * centered_code_index)) as f32)
}

/// Builds the full ascending level table in code order. Useful for
/// verification tooling; the decode path evaluates arithmetically instead.
#[must_use]
pub fn codebook_levels(bit_width: u32) -> Option<Vec<f32>> {
    if !is_supported_bit_width(bit_width) {
        return None;
    }
    // Every code is in range by construction, so the level evaluation
    // cannot fail once the bit width itself is supported.
    Some(
        (0..1_u32 << bit_width)
            .map(|code| {
                codebook_level_value(bit_width, code)
                    .expect("a supported bit width keeps every in-range code decodable")
            })
            .collect(),
    )
}

/// Unpacks one row's packed words into quantization codes, in input order.
///
/// `packed_code_words` holds the row’s `uint32` words
/// (`inputs · bits / 32` of them). Rows are always a whole number of 32-code
/// groups, so a straddling code always has its second word available.
///
/// Returns `None` on a malformed row: an unsupported bit width, an empty word
/// slice, or a word count that is not a whole multiple of the bit width (a
/// row length that is not a multiple of 32 codes).
#[must_use]
pub fn unpack_codes_from_packed_row(packed_code_words: &[u32], bit_width: u32) -> Option<Vec<u8>> {
    if !is_supported_bit_width(bit_width) || packed_code_words.is_empty() {
        return None;
    }
    // Whole 32-code groups only: words % bits == 0 keeps the row aligned.
    if !packed_code_words.len().is_multiple_of(bit_width as usize) {
        return None;
    }
    let code_count = packed_code_words.len() * u32::BITS as usize / bit_width as usize;
    let code_mask: u32 = (1_u32 << bit_width) - 1;
    let mut codes = Vec::with_capacity(code_count);
    for code_index in 0..code_count {
        let start_bit = code_index * bit_width as usize;
        let (word_index, bit_offset) = (
            start_bit / u32::BITS as usize,
            start_bit % u32::BITS as usize,
        );
        let mut spread_value = packed_code_words[word_index] >> bit_offset;
        let bits_available = u32::BITS as usize - bit_offset;
        if bits_available < bit_width as usize {
            // The code straddles into the next word; the shift below is < 32
            // because bit_offset > 0 whenever a straddle happens.
            spread_value |= packed_code_words[word_index + 1] << bits_available;
        }
        codes.push((spread_value & code_mask) as u8);
    }
    Some(codes)
}

/// Returns the packed word count for a row of `input_width` codes.
#[must_use]
pub const fn packed_word_count_for_input_width(input_width: u32, bit_width: u32) -> u32 {
    input_width * bit_width / u32::BITS
}

/// Derives the decoded bfloat16 weight shape for a selected expert page.
///
/// The packed source has shape `(expert_capacity, output_width, packed_words)`;
/// the page replaces the leading expert axis with `selected_expert_count`,
/// while the input width expands from each packed word according to `bit_width`.
/// Returns `None` for unsupported widths, empty or malformed shapes, page sizes
/// above capacity, or packed rows that are not complete 32-code groups.
#[must_use]
pub fn decoded_weight_page_shape(
    packed_source_shape: &[usize],
    bit_width: u32,
    selected_expert_count: usize,
) -> Option<Vec<usize>> {
    if !is_supported_bit_width(bit_width)
        || packed_source_shape.len() != 3
        || packed_source_shape.contains(&0)
        || selected_expert_count == 0
        || selected_expert_count > packed_source_shape[0]
    {
        return None;
    }
    let packed_row_bit_count = packed_source_shape[2].checked_mul(u32::BITS as usize)?;
    let bit_width_as_usize = bit_width as usize;
    if packed_row_bit_count % bit_width_as_usize != 0 {
        return None;
    }
    let input_width = packed_row_bit_count / bit_width_as_usize;
    if !input_width.is_multiple_of(CODES_PER_GROUP as usize) {
        return None;
    }
    Some(vec![
        selected_expert_count,
        packed_source_shape[1],
        input_width,
    ])
}

/// Returns the codes-per-group constant used by the bitstream layout.
#[must_use]
pub const fn codes_per_group() -> u32 {
    CODES_PER_GROUP
}
