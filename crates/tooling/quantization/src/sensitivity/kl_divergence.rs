//! The KL divergence formula every sensitivity probe measures, as a float32
//! CPU reference. It lives with sensitivity rather than the encoder because
//! it scores a forward pass, not a weight encoding; probe models implement
//! the same formula on device and are pinned to this one by the parity
//! fixture.

#[cfg(test)]
#[path = "kl_divergence_tests.rs"]
mod kl_divergence_tests;

/// KL(reference ‖ probe) over row-major logits of `[positions, vocabulary]`,
/// averaged over positions, in float32 arithmetic: the CPU pin for the
/// formula every probe model must implement on device.
///
/// The formula is OptiQ's `_kl_from_ref`: per position, `log_softmax` both
/// rows via `logits - logsumexp(logits)`, `p_ref = softmax(reference)`, and
/// sum `p_ref * (log p_ref - log p_probe)`; then mean over positions.
/// NaN when the inputs cannot be read as `[positions, vocabulary]`, which the
/// allocator treats as maximally sensitive rather than silently cheap.
#[must_use]
pub fn kl_divergence_reference_vs_probe(
    reference_logits: &[f32],
    probe_logits: &[f32],
    vocabulary_size: usize,
) -> f64 {
    if vocabulary_size == 0 || reference_logits.len() != probe_logits.len() {
        return f64::NAN;
    }
    let position_count = reference_logits.len() / vocabulary_size;
    if position_count == 0 {
        return f64::NAN;
    }
    let mut divergence_sum = 0.0_f32;
    let reference_rows = reference_logits.chunks_exact(vocabulary_size);
    let probe_rows = probe_logits.chunks_exact(vocabulary_size);
    for (reference_row, probe_row) in reference_rows.zip(probe_rows) {
        divergence_sum += position_kl_divergence(reference_row, probe_row);
    }
    f64::from(divergence_sum / position_count as f32)
}

/// `sum(p_ref * (log p_ref - log p_probe))` for one position.
fn position_kl_divergence(reference_row: &[f32], probe_row: &[f32]) -> f32 {
    let reference_log_normalizer = log_sum_exp(reference_row);
    let probe_log_normalizer = log_sum_exp(probe_row);
    let mut position_divergence = 0.0_f32;
    for (reference_logit, probe_logit) in reference_row.iter().zip(probe_row) {
        let reference_log_probability = reference_logit - reference_log_normalizer;
        let probe_log_probability = probe_logit - probe_log_normalizer;
        position_divergence +=
            reference_log_probability.exp() * (reference_log_probability - probe_log_probability);
    }
    position_divergence
}

/// `log(sum(exp(row)))`, shifted by the row maximum so large logits cannot
/// overflow the exponential.
fn log_sum_exp(row: &[f32]) -> f32 {
    let mut row_maximum = f32::NEG_INFINITY;
    for logit in row {
        row_maximum = row_maximum.max(*logit);
    }
    let mut shifted_exponential_sum = 0.0_f32;
    for logit in row {
        shifted_exponential_sum += (logit - row_maximum).exp();
    }
    row_maximum + shifted_exponential_sum.ln()
}
