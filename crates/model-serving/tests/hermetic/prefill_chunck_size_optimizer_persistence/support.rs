use std::path::Path;

use super::{
    PrefillChunckSizeOptimizer, PrefillChunckSizeOptimizerContext,
    PrefillChunckSizeOptimizerDecisionReason, PrefillChunckSizeOptimizerObservation,
};

pub(super) const OPTIMIZER_STATE_FILE_NAME: &str = "prefill-chunck-size.json";
pub(super) const TRUSTED_OBSERVATION_COUNT: usize = 3;
pub(super) const SLIDING_WINDOW_OBSERVATION_COUNT: usize = 5;
pub(super) const DRIFT_TRIGGER_FACTOR: u64 = 2;
pub(super) const DEFAULT_CANDIDATES: [usize; 5] = [128, 256, 512, 1024, 2048];

pub(super) fn create_optimizer_with_default_candidates() -> PrefillChunckSizeOptimizer {
    PrefillChunckSizeOptimizer::new(
        DEFAULT_CANDIDATES.to_vec(),
        TRUSTED_OBSERVATION_COUNT,
        SLIDING_WINDOW_OBSERVATION_COUNT,
        DRIFT_TRIGGER_FACTOR,
    )
    .expect("optimizer should construct with valid default candidates")
}

pub(super) fn temporary_directory() -> tempfile::TempDir {
    tempfile::tempdir().expect("temporary directory should be created")
}

pub(super) fn context_at_bucket(bucket: u64) -> PrefillChunckSizeOptimizerContext {
    PrefillChunckSizeOptimizerContext::new(bucket)
}

pub(super) fn full_observation(
    actual_prefill_chunck_tokens: usize,
    elapsed_millis: u64,
) -> PrefillChunckSizeOptimizerObservation {
    PrefillChunckSizeOptimizerObservation::full_prefill_chunck(
        actual_prefill_chunck_tokens,
        elapsed_millis,
    )
}

/// Feeds one full observation to the optimizer for a given context bucket.
pub(super) fn record_full_observation(
    prefill_chunck_size_optimizer: &mut PrefillChunckSizeOptimizer,
    prompt_processing_context: PrefillChunckSizeOptimizerContext,
    candidate_prefill_chunck_tokens: usize,
    elapsed_millis: u64,
) {
    prefill_chunck_size_optimizer
        .tell(
            prompt_processing_context,
            candidate_prefill_chunck_tokens,
            full_observation(candidate_prefill_chunck_tokens, elapsed_millis),
        )
        .expect("full observation should be accepted");
}

/// Drives the optimizer through exploration for a context bucket, recording
/// interleaved observations for every candidate until all are trusted.
pub(super) fn explore_all_candidates(
    prefill_chunck_size_optimizer: &mut PrefillChunckSizeOptimizer,
    prompt_processing_context: PrefillChunckSizeOptimizerContext,
    elapsed_millis_per_observation: u64,
) {
    for _observation_round in 0..TRUSTED_OBSERVATION_COUNT {
        for &candidate_prefill_chunck_tokens in &DEFAULT_CANDIDATES {
            let decision = prefill_chunck_size_optimizer.ask(prompt_processing_context);
            assert_eq!(
                decision.candidate_prefill_chunck_tokens, candidate_prefill_chunck_tokens,
                "each exploration round should visit candidates in sorted order"
            );
            assert_eq!(
                decision.reason,
                PrefillChunckSizeOptimizerDecisionReason::Exploration,
                "should be in exploration phase"
            );
            record_full_observation(
                prefill_chunck_size_optimizer,
                prompt_processing_context,
                candidate_prefill_chunck_tokens,
                elapsed_millis_per_observation,
            );
        }
    }
}

/// Loads an optimizer from the state file, expecting Ok(Some(...)).
pub(super) fn load_expect_some(
    state_file_path: &Path,
    model_id: &str,
    model_revision: &str,
    candidate_prefill_chunck_tokens: &[usize],
    trusted_observation_count: usize,
    sliding_window_observation_count: usize,
    drift_trigger_factor: u64,
) -> PrefillChunckSizeOptimizer {
    PrefillChunckSizeOptimizer::load_from_path(
        state_file_path.to_path_buf(),
        model_id.to_string(),
        model_revision.to_string(),
        candidate_prefill_chunck_tokens.to_vec(),
        trusted_observation_count,
        sliding_window_observation_count,
        drift_trigger_factor,
    )
    .expect("load should not return an error")
    .expect("load should return Some(optimizer)")
}

/// Loads an optimizer from the state file, expecting Ok(None).
pub(super) fn load_expect_none(
    state_file_path: &Path,
    model_id: &str,
    model_revision: &str,
    candidate_prefill_chunck_tokens: &[usize],
    trusted_observation_count: usize,
    sliding_window_observation_count: usize,
    drift_trigger_factor: u64,
) {
    let load_outcome = PrefillChunckSizeOptimizer::load_from_path(
        state_file_path.to_path_buf(),
        model_id.to_string(),
        model_revision.to_string(),
        candidate_prefill_chunck_tokens.to_vec(),
        trusted_observation_count,
        sliding_window_observation_count,
        drift_trigger_factor,
    )
    .expect("load should not return an error");
    assert!(
        load_outcome.is_none(),
        "load should return None for invalid or mismatched state"
    );
}
