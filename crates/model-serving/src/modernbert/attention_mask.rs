//! Host-built additive attention masks for the ModernBERT trunk.
//!
//! ModernBERT alternates global layers, where every token attends to every
//! token, with local layers, where a token attends only to tokens within a
//! window around it. A batch of padded rows additionally has to hide the
//! padding. The mask values are computed here as plain numbers so the rules
//! can be tested without a graphics processor.

use crate::modernbert::token_batch::PaddedTokenBatch;

/// The additive value that hides a position in a batch that carries padding.
///
/// It is the most negative finite `f32`, as the reference implementation
/// uses, and deliberately not negative infinity. A padded query row far from
/// any real token has every key hidden; with negative infinity its softmax is
/// `0 / 0 = NaN`, and that `NaN` then spreads to real rows through the next
/// layer's key products. With a finite value the row gets uniform attention
/// over hidden keys instead, which is finite and which no real row ever reads.
pub(crate) const HIDDEN_POSITION_MASK_VALUE: f32 = f32::MIN;

// The whole point of the value is that it is finite; a later edit to
// negative infinity must fail the build, not a test someone has to run.
const _: () = assert!(HIDDEN_POSITION_MASK_VALUE.is_finite());

/// Whether a sequence is long enough for the local window to hide any pair.
///
/// Within `window_half + 1` positions every pair is inside the window, so the
/// mask would be all zeros and unmasked attention is mathematically identical.
pub(crate) const fn local_window_hides_any_pair(
    sequence_length: usize,
    window_half: usize,
) -> bool {
    sequence_length > window_half + 1
}

/// The `[length, length]` local-window mask, row-major: zero where the two
/// positions are within `window_half` of each other and `hidden_value` elsewhere.
pub(crate) fn local_window_mask_values(
    sequence_length: usize,
    window_half: usize,
    hidden_value: f32,
) -> Vec<f32> {
    let mut mask_values = vec![hidden_value; sequence_length * sequence_length];
    if sequence_length == 0 {
        return mask_values;
    }
    for query_position in 0..sequence_length {
        let first_visible_key = query_position.saturating_sub(window_half);
        let last_visible_key = (query_position + window_half).min(sequence_length - 1);
        for key_position in first_visible_key..=last_visible_key {
            mask_values[query_position * sequence_length + key_position] = 0.0;
        }
    }
    mask_values
}

/// The `[sequence_count, padded_length]` key-padding mask, row-major: zero at
/// a row's real tokens and [`HIDDEN_POSITION_MASK_VALUE`] at its padding.
pub(crate) fn key_padding_mask_values(token_batch: &PaddedTokenBatch) -> Vec<f32> {
    let padded_length = token_batch.padded_length();
    let mut mask_values =
        vec![HIDDEN_POSITION_MASK_VALUE; token_batch.sequence_count() * padded_length];
    for (sequence_index, sequence_length) in token_batch.sequence_lengths().iter().enumerate() {
        let row_start = sequence_index * padded_length;
        for mask_value in &mut mask_values[row_start..row_start + sequence_length] {
            *mask_value = 0.0;
        }
    }
    mask_values
}

#[cfg(test)]
#[path = "attention_mask_tests.rs"]
mod tests;
