//! Stable on-disk state for the prefill chunk-size optimizer.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::warn;

use super::context_statistics::{CandidatePrefillChunckStatistics, ContextCandidateStatistics};
use super::optimizer::{CandidatePrefillChunckObservation, PrefillChunckSizeOptimizer};
use super::{PrefillChunckSizeOptimizerContext, PrefillChunckSizeOptimizerError};

const FORMAT_VERSION: u32 = 4;
const STATE_FILE_NAME: &str = "prefill-chunck-size.json";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct PersistedOptimizerState {
    format_version: u32,
    model_id: String,
    model_revision: String,
    candidate_prefill_chunck_tokens: Vec<usize>,
    sliding_window_observation_count: usize,
    decision_sequence: u64,
    observation_sequence: u64,
    context_buckets: Vec<PersistedContextBucket>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct PersistedContextBucket {
    context_identifier: u64,
    fallback_context_identifier: u64,
    candidates: Vec<PersistedCandidateStatistics>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct PersistedCandidateStatistics {
    observations: Vec<PersistedObservation>,
    last_observed_decision_sequence: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct PersistedObservation {
    actual_prefill_chunck_tokens: usize,
    elapsed_millis: u64,
    next_context_identifier: u64,
    next_fallback_context_identifier: u64,
    observation_sequence: u64,
}

pub(crate) fn save_optimizer_to_directory(
    optimizer: &PrefillChunckSizeOptimizer,
    optimizer_directory: &Path,
    model_id: &str,
    model_revision: &str,
) -> Result<(), PrefillChunckSizeOptimizerError> {
    if !optimizer_directory.exists() {
        fs::create_dir_all(optimizer_directory).map_err(|io_error| {
            PrefillChunckSizeOptimizerError::OptimizerStateDirectoryCreationFailed {
                directory: optimizer_directory.to_path_buf(),
                source: io_error,
            }
        })?;
    }
    let optimizer_state =
        PersistedOptimizerState::from_optimizer(optimizer, model_id, model_revision);
    let serialized_optimizer_state =
        serde_json::to_string(&optimizer_state).map_err(|serialization_error| {
            PrefillChunckSizeOptimizerError::OptimizerStateSerializationFailed {
                source: serialization_error,
            }
        })?;
    let state_file_path = optimizer_directory.join(STATE_FILE_NAME);
    let temporary_state_file_path = state_file_path.with_extension("json.tmp");
    fs::write(&temporary_state_file_path, serialized_optimizer_state).map_err(|io_error| {
        PrefillChunckSizeOptimizerError::OptimizerStateWriteFailed {
            path: temporary_state_file_path.clone(),
            source: io_error,
        }
    })?;
    fs::rename(&temporary_state_file_path, &state_file_path).map_err(|io_error| {
        PrefillChunckSizeOptimizerError::OptimizerStateRenameFailed {
            from: temporary_state_file_path,
            to: state_file_path,
            source: io_error,
        }
    })?;
    Ok(())
}

pub(crate) fn load_optimizer_from_path(
    state_file_path: &Path,
    model_id: &str,
    model_revision: &str,
    candidate_prefill_chunck_tokens: Vec<usize>,
    sliding_window_observation_count: usize,
) -> Result<Option<PrefillChunckSizeOptimizer>, PrefillChunckSizeOptimizerError> {
    if !state_file_path.exists() {
        return Ok(None);
    }
    let serialized_optimizer_state = match fs::read_to_string(state_file_path) {
        Ok(serialized_optimizer_state) => serialized_optimizer_state,
        Err(io_error) => {
            warn!(
                path = %state_file_path.display(),
                error = %io_error,
                "Failed to read optimizer state file; starting fresh"
            );
            return Ok(None);
        }
    };
    if serialized_optimizer_state.is_empty() {
        warn!(path = %state_file_path.display(), "Optimizer state file is empty; starting fresh");
        return Ok(None);
    }
    let optimizer_state: PersistedOptimizerState =
        match serde_json::from_str(&serialized_optimizer_state) {
            Ok(optimizer_state) => optimizer_state,
            Err(parse_error) => {
                warn!(
                    path = %state_file_path.display(),
                    error = %parse_error,
                    "Failed to parse optimizer state file; starting fresh"
                );
                return Ok(None);
            }
        };
    if optimizer_state.format_version != FORMAT_VERSION
        || optimizer_state.model_id != model_id
        || optimizer_state.model_revision != model_revision
        || optimizer_state.candidate_prefill_chunck_tokens != candidate_prefill_chunck_tokens
        || optimizer_state.sliding_window_observation_count != sliding_window_observation_count
    {
        warn!(
            path = %state_file_path.display(),
            "Optimizer state file does not match the current model or policy; starting fresh"
        );
        return Ok(None);
    }
    Ok(Some(
        optimizer_state.into_optimizer(candidate_prefill_chunck_tokens),
    ))
}

impl PersistedOptimizerState {
    fn from_optimizer(
        optimizer: &PrefillChunckSizeOptimizer,
        model_id: &str,
        model_revision: &str,
    ) -> Self {
        let context_buckets = optimizer
            .context_statistics()
            .iter()
            .map(
                |(prompt_processing_context, context_statistics)| PersistedContextBucket {
                    context_identifier: prompt_processing_context.context_identifier(),
                    fallback_context_identifier: prompt_processing_context
                        .fallback_context_identifier(),
                    candidates: context_statistics
                        .candidate_statistics
                        .iter()
                        .map(|candidate_statistics| PersistedCandidateStatistics {
                            observations: candidate_statistics
                                .observations
                                .iter()
                                .map(|observation| PersistedObservation {
                                    actual_prefill_chunck_tokens: observation
                                        .actual_prefill_chunck_tokens,
                                    elapsed_millis: observation.elapsed_millis,
                                    next_context_identifier: observation
                                        .next_prompt_processing_context
                                        .context_identifier(),
                                    next_fallback_context_identifier: observation
                                        .next_prompt_processing_context
                                        .fallback_context_identifier(),
                                    observation_sequence: observation.observation_sequence,
                                })
                                .collect(),
                            last_observed_decision_sequence: candidate_statistics
                                .last_observed_decision_sequence,
                        })
                        .collect(),
                },
            )
            .collect();
        Self {
            format_version: FORMAT_VERSION,
            model_id: model_id.to_owned(),
            model_revision: model_revision.to_owned(),
            candidate_prefill_chunck_tokens: optimizer.candidate_prefill_chunck_tokens().to_vec(),
            sliding_window_observation_count: optimizer.sliding_window_observation_count(),
            decision_sequence: optimizer.decision_sequence(),
            observation_sequence: optimizer.observation_sequence(),
            context_buckets,
        }
    }

    fn into_optimizer(
        self,
        candidate_prefill_chunck_tokens: Vec<usize>,
    ) -> PrefillChunckSizeOptimizer {
        let context_statistics: BTreeMap<
            PrefillChunckSizeOptimizerContext,
            ContextCandidateStatistics,
        > = self
            .context_buckets
            .into_iter()
            .map(|persisted_context_bucket| {
                let prompt_processing_context =
                    PrefillChunckSizeOptimizerContext::new_with_fallback(
                        persisted_context_bucket.context_identifier,
                        persisted_context_bucket.fallback_context_identifier,
                    );
                let candidate_statistics = persisted_context_bucket
                    .candidates
                    .into_iter()
                    .map(
                        |persisted_candidate_statistics| CandidatePrefillChunckStatistics {
                            observations: persisted_candidate_statistics
                                .observations
                                .into_iter()
                                .map(|persisted_observation| CandidatePrefillChunckObservation {
                                    actual_prefill_chunck_tokens: persisted_observation
                                        .actual_prefill_chunck_tokens,
                                    elapsed_millis: persisted_observation.elapsed_millis,
                                    next_prompt_processing_context:
                                        PrefillChunckSizeOptimizerContext::new_with_fallback(
                                            persisted_observation.next_context_identifier,
                                            persisted_observation.next_fallback_context_identifier,
                                        ),
                                    observation_sequence: persisted_observation
                                        .observation_sequence,
                                })
                                .collect(),
                            last_observed_decision_sequence: persisted_candidate_statistics
                                .last_observed_decision_sequence,
                        },
                    )
                    .collect();
                (
                    prompt_processing_context,
                    ContextCandidateStatistics {
                        candidate_statistics,
                    },
                )
            })
            .collect();
        PrefillChunckSizeOptimizer::new_from_persisted_state(
            candidate_prefill_chunck_tokens,
            self.sliding_window_observation_count,
            self.decision_sequence,
            self.observation_sequence,
            context_statistics,
        )
    }
}
