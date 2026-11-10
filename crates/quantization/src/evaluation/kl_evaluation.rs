//! The evaluation driver and the seam to the engine that runs the forwards.
//!
//! Two models cannot be held at once on the machines this is for, so the
//! reference runs first and its logits are kept while the candidate loads.
//! Logits never cross this boundary: the engine keeps the reference's on
//! device and returns only per-token divergences, which is what keeps this
//! crate free of an array type and lets the engine reduce on the GPU.

use std::path::Path;

use crate::error::QuantizationError;
use crate::progress::ProgressReporter;
use crate::sensitivity::CalibrationForwardSample;

use super::kl_statistics::KlEvaluationSummary;

#[cfg(test)]
#[path = "kl_evaluation_tests.rs"]
mod kl_evaluation_tests;

/// What an evaluation compares and on what.
pub struct KlEvaluationRequest<'request> {
    /// The model whose distribution is the truth: the bfloat16 base, or a
    /// uniform quant of it when the base does not fit in memory, in which
    /// case the figures are a lower bound on drift from full precision.
    pub reference_directory: &'request Path,
    /// The quant being evaluated.
    pub candidate_directory: &'request Path,
    pub prompts: &'request [CalibrationForwardSample],
}

/// An engine that can forward both models over the same prompts.
pub trait KlDivergenceEngine {
    /// Per-token KL(reference ‖ candidate) for each prompt, in prompt
    /// order: the reference is loaded, forwarded over every prompt and
    /// released before the candidate is loaded.
    fn per_token_divergences(
        &mut self,
        request: &KlEvaluationRequest<'_>,
        progress_reporter: &mut dyn ProgressReporter,
    ) -> Result<Vec<Vec<f64>>, QuantizationError>;
}

/// Runs the evaluation and summarizes it, OptiQ's `evaluate_kl`.
pub fn evaluate_kl_divergence(
    engine: &mut dyn KlDivergenceEngine,
    request: &KlEvaluationRequest<'_>,
    progress_reporter: &mut dyn ProgressReporter,
) -> Result<KlEvaluationSummary, QuantizationError> {
    if request.prompts.is_empty() {
        return Err(QuantizationError::EvaluationHasNoPrompts);
    }
    let per_prompt_divergences = engine.per_token_divergences(request, progress_reporter)?;
    match KlEvaluationSummary::from_per_prompt_divergences(&per_prompt_divergences) {
        Some(summary) => Ok(summary),
        None => Err(QuantizationError::EvaluationHasNoPrompts),
    }
}
