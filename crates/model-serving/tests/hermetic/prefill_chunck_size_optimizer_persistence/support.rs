use std::path::Path;

use super::{
    PrefillChunckSizeOptimizer, PrefillChunckSizeOptimizerContext,
    PrefillChunckSizeOptimizerObservation,
};

pub(super) const OPTIMIZER_STATE_FILE_NAME: &str = "prefill-chunck-size.json";
pub(super) const SLIDING_WINDOW_OBSERVATION_COUNT: usize = 5;
pub(super) const DEFAULT_CANDIDATES: [usize; 5] = [128, 256, 512, 1_024, 2_048];

pub(super) fn create_optimizer_with_default_candidates() -> PrefillChunckSizeOptimizer {
    PrefillChunckSizeOptimizer::new(
        DEFAULT_CANDIDATES.to_vec(),
        SLIDING_WINDOW_OBSERVATION_COUNT,
    )
    .expect("optimizer should construct with valid default candidates")
}

pub(super) fn temporary_directory() -> tempfile::TempDir {
    tempfile::tempdir().expect("temporary directory should be created")
}

pub(super) fn context_at_bucket(bucket: u64) -> PrefillChunckSizeOptimizerContext {
    PrefillChunckSizeOptimizerContext::new(bucket)
}

pub(super) fn record_transition_observation(
    prefill_chunck_size_optimizer: &mut PrefillChunckSizeOptimizer,
    prompt_processing_context: PrefillChunckSizeOptimizerContext,
    candidate_prefill_chunck_tokens: usize,
    actual_prefill_chunck_tokens: usize,
    elapsed_millis: u64,
    next_prompt_processing_context: PrefillChunckSizeOptimizerContext,
) {
    prefill_chunck_size_optimizer
        .tell(
            prompt_processing_context,
            candidate_prefill_chunck_tokens,
            PrefillChunckSizeOptimizerObservation::transition(
                actual_prefill_chunck_tokens,
                elapsed_millis,
                next_prompt_processing_context,
            ),
        )
        .expect("transition observation should be accepted");
}

pub(super) fn observe_all_candidates(
    prefill_chunck_size_optimizer: &mut PrefillChunckSizeOptimizer,
    prompt_processing_context: PrefillChunckSizeOptimizerContext,
    elapsed_millis_per_observation: u64,
) {
    for _candidate_prefill_chunck_tokens in DEFAULT_CANDIDATES {
        let decision = prefill_chunck_size_optimizer.ask(prompt_processing_context);
        record_transition_observation(
            prefill_chunck_size_optimizer,
            prompt_processing_context,
            decision.candidate_prefill_chunck_tokens,
            decision.candidate_prefill_chunck_tokens,
            elapsed_millis_per_observation,
            prompt_processing_context,
        );
    }
}

pub(super) fn load_expect_some(
    state_file_path: &Path,
    model_id: &str,
    model_revision: &str,
    candidate_prefill_chunck_tokens: &[usize],
    sliding_window_observation_count: usize,
) -> PrefillChunckSizeOptimizer {
    PrefillChunckSizeOptimizer::load_from_path(
        state_file_path.to_path_buf(),
        model_id.to_owned(),
        model_revision.to_owned(),
        candidate_prefill_chunck_tokens.to_vec(),
        sliding_window_observation_count,
    )
    .expect("load should not return an error")
    .expect("load should return an optimizer")
}

pub(super) fn load_expect_none(
    state_file_path: &Path,
    model_id: &str,
    model_revision: &str,
    candidate_prefill_chunck_tokens: &[usize],
    sliding_window_observation_count: usize,
) {
    let load_outcome = PrefillChunckSizeOptimizer::load_from_path(
        state_file_path.to_path_buf(),
        model_id.to_owned(),
        model_revision.to_owned(),
        candidate_prefill_chunck_tokens.to_vec(),
        sliding_window_observation_count,
    )
    .expect("load should not return an error");
    assert!(load_outcome.is_none());
}
