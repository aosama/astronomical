//! Groups sequences into forward passes of bounded size: Laya's serving,
//! and every family's whole-model training.
//!
//! Every sequence in a pass is padded to the longest one, so the cost of a
//! pass is its sequence count times its longest length. Passing a request's
//! 256 questions at 1,024 tokens each in one batch would build a
//! quarter-million-token graph and a gigabyte of attention masks; grouping
//! under a padded-token budget keeps each pass small. Sequences are grouped
//! by length so little of a pass is padding.

/// The most padded tokens one of Laya's serving passes may hold.
///
/// It is the budget the reference implementation batches under, and keeps a
/// pass's local-attention mask in the tens of megabytes.
pub const LAYA_SERVING_PADDED_TOKENS_PER_PASS: usize = 16_384;

/// Splits sequences into passes, returning each pass as indices into `sequence_lengths`.
///
/// Every index appears in exactly one pass. A single sequence longer than the
/// budget still gets a pass of its own.
pub fn plan_forward_passes(
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

/// Whether `forward_pass` (indices into `sequence_lengths`) holds more
/// padded tokens than `padded_tokens_per_pass`: only a pass of one
/// sequence longer than the budget can, since
/// [`plan_forward_passes`] closes every other pass before it does.
pub fn pass_exceeds_budget(
    forward_pass: &[usize],
    sequence_lengths: &[usize],
    padded_tokens_per_pass: usize,
) -> bool {
    let mut longest_length = 0;
    for sequence_index in forward_pass {
        longest_length = longest_length.max(sequence_lengths[*sequence_index]);
    }
    longest_length * forward_pass.len() > padded_tokens_per_pass
}

/// The share of the runtime's memory limit one training micro-batch may
/// use, so the model, the optimizer and macOS keep the rest.
const TRAINING_MEMORY_LIMIT_DIVISOR: usize = 4;
/// The largest training micro-batch, in padded tokens: where Laya's
/// measured step time stops improving with size, and where Kev's does too
/// (53 ms a question at 2,048 against 52 at 1,024).
const MAXIMUM_TRAINING_PADDED_TOKENS_PER_PASS: usize = 1_024;
/// The smallest: one long question still forms a pass of its own.
const MINIMUM_TRAINING_PADDED_TOKENS_PER_PASS: usize = 128;

/// The padded tokens one training micro-batch may hold under a runtime
/// memory limit of `memory_limit_bytes`, for a model whose backward pass
/// keeps `bytes_per_padded_token` of activations per token: a quarter of
/// the limit, never more than the measured sweet spot. A Mac with less
/// memory trains in smaller batches, so it trains more slowly rather than
/// swapping; found live, when fixed 2,048-token batches froze a 48 GB Mac.
pub fn training_tokens_per_pass(memory_limit_bytes: usize, bytes_per_padded_token: usize) -> usize {
    (memory_limit_bytes / TRAINING_MEMORY_LIMIT_DIVISOR / bytes_per_padded_token.max(1)).clamp(
        MINIMUM_TRAINING_PADDED_TOKENS_PER_PASS,
        MAXIMUM_TRAINING_PADDED_TOKENS_PER_PASS,
    )
}

/// The share of the memory limit one sequence's kept activations may take
/// before its layers are recomputed instead: half, so the model, the
/// optimizer and macOS keep the other half. Twice the batching budget, so a
/// text that only overflows the batching budget, such as Clef's built-in
/// check texts with their long prompt, keeps today's faster path. Measured on
/// Clef 9B (16 MB a token, linear), a stored text this long peaks near 29 GB
/// under a 36.9 GB limit.
const RECOMPUTATION_MEMORY_LIMIT_DIVISOR: usize = 2;

/// The padded tokens above which a training pass recomputes its layers
/// during the backward pass rather than keep their activations.
pub fn recomputation_threshold_tokens(
    memory_limit_bytes: usize,
    bytes_per_padded_token: usize,
) -> usize {
    (memory_limit_bytes / RECOMPUTATION_MEMORY_LIMIT_DIVISOR / bytes_per_padded_token.max(1))
        .max(MINIMUM_TRAINING_PADDED_TOKENS_PER_PASS)
}

#[cfg(test)]
#[path = "forward_pass_planning_tests.rs"]
mod tests;
