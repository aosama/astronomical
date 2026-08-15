use std::path::{Path, PathBuf};

use super::{
    PromptProcessingChunkMeasurement, PromptProcessingChunkSizeOptimizer,
    PromptProcessingMeasurementContext,
};

pub(super) const OPTIMIZER_STATE_FILE_NAME: &str =
    "prompt-processing-chunk-size-optimizer-state.json";
pub(super) const MAXIMUM_RETAINED_MEASUREMENTS: usize = 5;
pub(super) const DEFAULT_CANDIDATES: [usize; 5] = [128, 256, 512, 1_024, 2_048];

pub(super) fn create_optimizer_with_default_candidates() -> PromptProcessingChunkSizeOptimizer {
    PromptProcessingChunkSizeOptimizer::new(
        DEFAULT_CANDIDATES.to_vec(),
        MAXIMUM_RETAINED_MEASUREMENTS,
    )
    .expect("optimizer should construct with valid default candidates")
}

pub(super) fn temporary_directory() -> tempfile::TempDir {
    tempfile::tempdir().expect("temporary directory should be created")
}

pub(super) fn state_file_path_for_model(
    optimizer_directory: &Path,
    model_id: &str,
    model_revision: &str,
) -> PathBuf {
    PromptProcessingChunkSizeOptimizer::persisted_state_file_path(
        optimizer_directory,
        model_id,
        model_revision,
    )
}

pub(super) fn context_at_position_range(
    position_range_identifier: u64,
) -> PromptProcessingMeasurementContext {
    PromptProcessingMeasurementContext::isolated(position_range_identifier)
}

pub(super) fn record_chunk_measurement(
    chunk_size_optimizer: &mut PromptProcessingChunkSizeOptimizer,
    measurement_context: PromptProcessingMeasurementContext,
    selected_candidate_chunk_size_tokens: usize,
    processed_prompt_token_count: usize,
    forward_elapsed_millis: u64,
    next_measurement_context: PromptProcessingMeasurementContext,
) {
    chunk_size_optimizer
        .record_measurement(
            measurement_context,
            selected_candidate_chunk_size_tokens,
            PromptProcessingChunkMeasurement::transition(
                processed_prompt_token_count,
                forward_elapsed_millis,
                next_measurement_context,
            ),
        )
        .expect("chunk measurement should be accepted");
}

pub(super) fn measure_all_candidates(
    chunk_size_optimizer: &mut PromptProcessingChunkSizeOptimizer,
    measurement_context: PromptProcessingMeasurementContext,
    forward_elapsed_millis_per_measurement: u64,
) {
    for _candidate_chunk_size_tokens in DEFAULT_CANDIDATES {
        let selection = chunk_size_optimizer.select_candidate_chunk_size(measurement_context);
        record_chunk_measurement(
            chunk_size_optimizer,
            measurement_context,
            selection.selected_candidate_chunk_size_tokens,
            selection.selected_candidate_chunk_size_tokens,
            forward_elapsed_millis_per_measurement,
            measurement_context,
        );
    }
}

pub(super) fn load_expect_some(
    state_file_path: &Path,
    model_id: &str,
    model_revision: &str,
    candidate_chunk_size_tokens: &[usize],
    maximum_retained_measurements: usize,
) -> PromptProcessingChunkSizeOptimizer {
    PromptProcessingChunkSizeOptimizer::load_from_path(
        state_file_path.to_path_buf(),
        model_id.to_owned(),
        model_revision.to_owned(),
        candidate_chunk_size_tokens.to_vec(),
        maximum_retained_measurements,
    )
    .expect("load should not return an error")
    .expect("load should return an optimizer")
}

pub(super) fn load_expect_none(
    state_file_path: &Path,
    model_id: &str,
    model_revision: &str,
    candidate_chunk_size_tokens: &[usize],
    maximum_retained_measurements: usize,
) {
    let load_outcome = PromptProcessingChunkSizeOptimizer::load_from_path(
        state_file_path.to_path_buf(),
        model_id.to_owned(),
        model_revision.to_owned(),
        candidate_chunk_size_tokens.to_vec(),
        maximum_retained_measurements,
    )
    .expect("load should not return an error");
    assert!(load_outcome.is_none());
}
