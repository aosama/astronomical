//! Vocabulary-boundary rules for one Qwen3.5 generation session.

/// Returns whether any token ID sits at or beyond the model's vocabulary
/// size. Token IDs are indexes into the model's vocabulary table, so an ID
/// equal to the size is already one past the final valid entry.
pub(super) fn contains_token_id_outside_vocabulary(
    token_ids: &[u32],
    vocabulary_size: u32,
) -> bool {
    for token_id in token_ids {
        if *token_id >= vocabulary_size {
            return true;
        }
    }
    false
}

#[cfg(test)]
#[path = "generation_vocabulary_boundaries_tests.rs"]
mod generation_vocabulary_boundaries_tests;
