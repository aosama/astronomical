//! The token sequence a MLX working-buffer probe runs at load.
//!
//! A probe's cost has to resemble a real prompt's. An expert layer's cost
//! depends on how many different experts the chunk routes to: a constant
//! input sends every token to the same few experts and touches far less
//! than a real prompt does. A sequence that walks the vocabulary does not.

/// Step between the probe's token ids: a prime far from any power of two,
/// so consecutive positions land all over the vocabulary.
const PROBE_TOKEN_ID_STRIDE: u64 = 7_919;

/// Token ids that walk the vocabulary in large steps, wrapping around.
pub(crate) fn vocabulary_walking_token_ids(token_count: i32, vocabulary_size: u32) -> Vec<i32> {
    let vocabulary_size = u64::from(vocabulary_size.max(1));
    let mut token_ids = Vec::with_capacity(usize::try_from(token_count).unwrap_or(0));
    for token_position in 0..u64::try_from(token_count).unwrap_or(0) {
        let token_id = token_position.wrapping_mul(PROBE_TOKEN_ID_STRIDE) % vocabulary_size;
        token_ids.push(i32::try_from(token_id).unwrap_or(0));
    }
    token_ids
}

#[cfg(test)]
#[path = "probe_token_ids_tests.rs"]
mod tests;
