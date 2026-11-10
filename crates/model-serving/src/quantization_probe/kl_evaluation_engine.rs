//! Whole-artifact KL evaluation through the Qwen3.5 engine: the reference
//! model's logits are kept on device while the candidate loads, and only
//! per-token divergences leave.
//!
//! The model side of OptiQ's `evaluate_kl` (`optiq/eval/kl.py`), which loads
//! both models through mlx-lm; here both run through the engine that will
//! serve the artifact, so the figure describes the kernels a request uses.

use std::path::Path;

use astronomical_quantization::error::QuantizationError;
use astronomical_quantization::evaluation::{KlDivergenceEngine, KlEvaluationRequest};
use astronomical_quantization::progress::{ProgressReporter, StageProgressTracker};
use astronomical_quantization::sensitivity::{CalibrationForwardSample, SensitivityProbeModel};
use astronomical_runtime_integration::MlxMemoryLimits;
use mlx_bridge::MlxArray;

use super::qwen3_5_probe::{Qwen3_5SensitivityProbe, load_qwen3_5_probe_model};

/// The progress stage the CLI prints: one unit per forward, both models.
const EVALUATION_STAGE_NAME: &str = "KL evaluation";

/// Forwards a reference and a candidate Qwen3.5 model over the same prompts.
pub struct Qwen3_5KlDivergenceEngine {
    pub memory_limits: MlxMemoryLimits,
    /// The prompt chunk the forwards use; 2,048 tokens by the repository's
    /// reproducibility rule.
    pub prefill_chunk_tokens: usize,
}

impl Qwen3_5KlDivergenceEngine {
    fn load_model(
        &self,
        model_directory: &Path,
    ) -> Result<Qwen3_5SensitivityProbe, QuantizationError> {
        load_qwen3_5_probe_model(
            model_directory,
            None,
            self.memory_limits,
            self.prefill_chunk_tokens,
        )
    }

    /// The reference's logits for every prompt. The model is released when
    /// this returns; the logits stay, in the engine's half-width logit
    /// dtype, which is what makes holding all of them affordable.
    fn reference_logits(
        &self,
        reference_directory: &Path,
        prompts: &[CalibrationForwardSample],
        progress_tracker: &mut StageProgressTracker<'_>,
    ) -> Result<Vec<MlxArray>, QuantizationError> {
        let mut reference_model = self.load_model(reference_directory)?;
        let mut reference_logits = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            reference_logits.push(reference_model.forward(prompt)?);
            progress_tracker.advance("reference forward");
        }
        reference_model.release_transient_memory()?;
        Ok(reference_logits)
    }
}

impl KlDivergenceEngine for Qwen3_5KlDivergenceEngine {
    fn per_token_divergences(
        &mut self,
        request: &KlEvaluationRequest<'_>,
        progress_reporter: &mut dyn ProgressReporter,
    ) -> Result<Vec<Vec<f64>>, QuantizationError> {
        let mut progress_tracker = StageProgressTracker::new(
            EVALUATION_STAGE_NAME,
            2 * request.prompts.len() as u64,
            0,
            StageProgressTracker::DEFAULT_REPORT_INTERVAL,
            progress_reporter,
        );
        let reference_logits = self.reference_logits(
            request.reference_directory,
            request.prompts,
            &mut progress_tracker,
        )?;
        let mut candidate_model = self.load_model(request.candidate_directory)?;
        let mut per_prompt_divergences = Vec::with_capacity(request.prompts.len());
        for (prompt, prompt_reference_logits) in request.prompts.iter().zip(&reference_logits) {
            let candidate_logits = candidate_model.forward(prompt)?;
            per_prompt_divergences.push(
                candidate_model
                    .per_position_kl_divergences(prompt_reference_logits, &candidate_logits)?,
            );
            progress_tracker.advance("candidate forward");
        }
        progress_tracker.report_now("every prompt measured");
        Ok(per_prompt_divergences)
    }
}
