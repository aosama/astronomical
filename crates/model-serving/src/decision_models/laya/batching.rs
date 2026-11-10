//! Groups a request's question sequences into forward passes of bounded size.
//!
//! Every sequence in a pass is padded to the longest one, so the cost of a
//! pass is its sequence count times its longest length. Passing a request's
//! 256 questions at 1,024 tokens each in one batch would build a
//! quarter-million-token graph and a gigabyte of attention masks; grouping
//! under a padded-token budget keeps each pass small. Sequences are grouped
//! by length so little of a pass is padding.

/// The most padded tokens one forward pass may hold.
///
/// It is the budget the reference implementation batches under, and keeps a
/// pass's local-attention mask in the tens of megabytes.
pub(crate) const PADDED_TOKENS_PER_PASS: usize = 16_384;

/// Splits sequences into passes, returning each pass as indices into `sequence_lengths`.
///
/// Every index appears in exactly one pass. A single sequence longer than the
/// budget still gets a pass of its own.
pub(crate) fn plan_forward_passes(
    sequence_lengths: &[usize],
    padded_tokens_per_pass: usize,
) -> Vec<Vec<usize>> {
    let mut indices_by_length: Vec<usize> = (0..sequence_lengths.len()).collect();
    indices_by_length.sort_by_key(|sequence_index| sequence_lengths[*sequence_index]);
    let mut forward_passes: Vec<Vec<usize>> = Vec::new();
    let mut current_pass: Vec<usize> = Vec::new();
    for sequence_index in indices_by_length {
        // Sequences arrive shortest first, so the newest is the pass's longest.
        let padded_length = sequence_lengths[sequence_index];
        let padded_tokens_with_this_sequence = padded_length * (current_pass.len() + 1);
        if !current_pass.is_empty() && padded_tokens_with_this_sequence > padded_tokens_per_pass {
            forward_passes.push(std::mem::take(&mut current_pass));
        }
        current_pass.push(sequence_index);
    }
    if !current_pass.is_empty() {
        forward_passes.push(current_pass);
    }
    forward_passes
}

#[cfg(test)]
#[path = "batching_tests.rs"]
mod tests;
