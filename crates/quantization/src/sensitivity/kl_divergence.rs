//! The KL divergence formula every sensitivity probe measures, as a float32
//! CPU reference. It lives with sensitivity rather than the encoder because
//! it scores a forward pass, not a weight encoding; probe models implement
//! the same formula on device and are pinned to this one by the parity
//! fixture.

/// KL(reference ‖ probe) over row-major logits of `[positions, vocabulary]`,
/// averaged over positions, in float32 arithmetic: the CPU pin for the
/// formula every probe model must implement on device.
///
/// Work to port (OptiQ's `_kl_from_ref`): per position, `log_softmax` both
/// rows via `logits - logsumexp(logits)`, `p_ref = softmax(reference)`, and
/// sum `p_ref * (log p_ref - log p_probe)`; then mean over positions.
#[must_use]
pub fn kl_divergence_reference_vs_probe(
    reference_logits: &[f32],
    probe_logits: &[f32],
    vocabulary_size: usize,
) -> f64 {
    let _ = (reference_logits, probe_logits, vocabulary_size);
    todo!(
        "log-softmax both, weight the log-ratio by the reference probabilities, average over positions"
    )
}
