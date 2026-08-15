//! Disposable, model-and-revision-scoped optimizer evidence for warm process starts.
//!
//! Loading accepts state only when identity, format, candidates, and retention
//! settings match. Missing or rejected state falls forward to a fresh optimizer.
//! Saving writes a temporary sibling before rename so readers never accept a
//! partially serialized state file.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::warn;

use super::context_statistics::{CandidateChunkStatistics, ContextCandidateStatistics};
use super::optimizer::{CandidateChunkMeasurement, PromptProcessingChunkSizeOptimizer};
use super::{PromptProcessingChunkSizeOptimizerError, PromptProcessingMeasurementContext};

/// Disposable schema version for prompt-processing terminology and transition evidence.
const FORMAT_VERSION: u32 = 5;
const STATE_FILE_NAME: &str = "prompt-processing-chunk-size-optimizer-state.json";
const MODEL_PROFILE_DIRECTORY_NAME: &str = "model-profiles";
const MODEL_PROFILE_NAMESPACE: &[u8] =
    b"astronomical-prompt-processing-chunk-optimizer-model-profile-v1";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct PersistedOptimizerState {
    format_version: u32,
    model_id: String,
    model_revision: String,
    candidate_chunk_size_tokens: Vec<usize>,
    maximum_retained_measurements_per_candidate_and_context: usize,
    selection_sequence: u64,
    measurement_sequence: u64,
    context_buckets: Vec<PersistedContextBucket>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct PersistedContextBucket {
    exact_measurement_context_identifier: u64,
    position_independent_execution_profile_identifier: u64,
    candidates: Vec<PersistedCandidateStatistics>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct PersistedCandidateStatistics {
    measurements: Vec<PersistedMeasurement>,
    last_measured_selection_sequence: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct PersistedMeasurement {
    processed_prompt_token_count: usize,
    forward_elapsed_millis: u64,
    next_exact_measurement_context_identifier: u64,
    next_position_independent_execution_profile_identifier: u64,
    measurement_sequence: u64,
}

pub(crate) fn save_optimizer_to_directory(
    optimizer: &PromptProcessingChunkSizeOptimizer,
    optimizer_directory: &Path,
    model_id: &str,
    model_revision: &str,
) -> Result<(), PromptProcessingChunkSizeOptimizerError> {
    let model_profile_directory =
        optimizer_model_profile_directory(optimizer_directory, model_id, model_revision);
    if !model_profile_directory.exists() {
        fs::create_dir_all(&model_profile_directory).map_err(|io_error| {
            PromptProcessingChunkSizeOptimizerError::OptimizerStateDirectoryCreationFailed {
                directory: model_profile_directory.clone(),
                source: io_error,
            }
        })?;
    }
    let optimizer_state =
        PersistedOptimizerState::from_optimizer(optimizer, model_id, model_revision);
    let serialized_optimizer_state =
        serde_json::to_string(&optimizer_state).map_err(|serialization_error| {
            PromptProcessingChunkSizeOptimizerError::OptimizerStateSerializationFailed {
                source: serialization_error,
            }
        })?;
    let state_file_path = model_profile_directory.join(STATE_FILE_NAME);
    let temporary_state_file_path = state_file_path.with_extension("json.tmp");
    fs::write(&temporary_state_file_path, serialized_optimizer_state).map_err(|io_error| {
        PromptProcessingChunkSizeOptimizerError::OptimizerStateWriteFailed {
            path: temporary_state_file_path.clone(),
            source: io_error,
        }
    })?;
    fs::rename(&temporary_state_file_path, &state_file_path).map_err(|io_error| {
        PromptProcessingChunkSizeOptimizerError::OptimizerStateRenameFailed {
            from: temporary_state_file_path,
            to: state_file_path,
            source: io_error,
        }
    })?;
    Ok(())
}

/// Resolves one opaque model-and-revision namespace beneath the shared
/// optimizer root so switching models never replaces another model's evidence.
pub(crate) fn optimizer_state_file_path(
    optimizer_directory: &Path,
    model_id: &str,
    model_revision: &str,
) -> PathBuf {
    optimizer_model_profile_directory(optimizer_directory, model_id, model_revision)
        .join(STATE_FILE_NAME)
}

fn optimizer_model_profile_directory(
    optimizer_directory: &Path,
    model_id: &str,
    model_revision: &str,
) -> PathBuf {
    // Hash each variable-length identity independently before combining it.
    // Fixed-width digests prevent concatenation ambiguity and keep model IDs
    // containing path separators out of filesystem path components.
    let model_id_digest = Sha256::digest(model_id.as_bytes());
    let model_revision_digest = Sha256::digest(model_revision.as_bytes());
    let mut model_profile_digest = Sha256::new();
    model_profile_digest.update(MODEL_PROFILE_NAMESPACE);
    model_profile_digest.update(model_id_digest);
    model_profile_digest.update(model_revision_digest);
    let model_profile_identifier = hex_encode(model_profile_digest.finalize().into());
    optimizer_directory
        .join(MODEL_PROFILE_DIRECTORY_NAME)
        .join(model_profile_identifier)
}

fn hex_encode(bytes: [u8; 32]) -> String {
    bytes
        .iter()
        .map(|fingerprint_byte| format!("{fingerprint_byte:02x}"))
        .collect()
}

pub(crate) fn load_optimizer_from_path(
    state_file_path: &Path,
    model_id: &str,
    model_revision: &str,
    candidate_chunk_size_tokens: Vec<usize>,
    maximum_retained_measurements_per_candidate_and_context: usize,
) -> Result<Option<PromptProcessingChunkSizeOptimizer>, PromptProcessingChunkSizeOptimizerError> {
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
        || optimizer_state.candidate_chunk_size_tokens != candidate_chunk_size_tokens
        || optimizer_state.maximum_retained_measurements_per_candidate_and_context
            != maximum_retained_measurements_per_candidate_and_context
    {
        warn!(
            path = %state_file_path.display(),
            "Optimizer state file does not match the current model or policy; starting fresh"
        );
        return Ok(None);
    }
    Ok(Some(
        optimizer_state.into_optimizer(candidate_chunk_size_tokens),
    ))
}

impl PersistedOptimizerState {
    fn from_optimizer(
        optimizer: &PromptProcessingChunkSizeOptimizer,
        model_id: &str,
        model_revision: &str,
    ) -> Self {
        let context_buckets = optimizer
            .context_statistics()
            .iter()
            .map(
                |(measurement_context, context_statistics)| PersistedContextBucket {
                    exact_measurement_context_identifier: measurement_context
                        .exact_measurement_context_identifier(),
                    position_independent_execution_profile_identifier: measurement_context
                        .position_independent_execution_profile_identifier(),
                    candidates: context_statistics
                        .candidate_statistics
                        .iter()
                        .map(|candidate_statistics| PersistedCandidateStatistics {
                            measurements: candidate_statistics
                                .measurements
                                .iter()
                                .map(|measurement| PersistedMeasurement {
                                    processed_prompt_token_count: measurement
                                        .processed_prompt_token_count,
                                    forward_elapsed_millis: measurement.forward_elapsed_millis,
                                    next_exact_measurement_context_identifier: measurement
                                        .next_measurement_context
                                        .exact_measurement_context_identifier(),
                                    next_position_independent_execution_profile_identifier:
                                        measurement
                                            .next_measurement_context
                                            .position_independent_execution_profile_identifier(),
                                    measurement_sequence: measurement.measurement_sequence,
                                })
                                .collect(),
                            last_measured_selection_sequence: candidate_statistics
                                .last_measured_selection_sequence,
                        })
                        .collect(),
                },
            )
            .collect();
        Self {
            format_version: FORMAT_VERSION,
            model_id: model_id.to_owned(),
            model_revision: model_revision.to_owned(),
            candidate_chunk_size_tokens: optimizer.candidate_chunk_size_tokens().to_vec(),
            maximum_retained_measurements_per_candidate_and_context: optimizer
                .maximum_retained_measurements_per_candidate_and_context(),
            selection_sequence: optimizer.selection_sequence(),
            measurement_sequence: optimizer.measurement_sequence(),
            context_buckets,
        }
    }

    fn into_optimizer(
        self,
        candidate_chunk_size_tokens: Vec<usize>,
    ) -> PromptProcessingChunkSizeOptimizer {
        let context_statistics: BTreeMap<
            PromptProcessingMeasurementContext,
            ContextCandidateStatistics,
        > = self
            .context_buckets
            .into_iter()
            .map(|persisted_context_bucket| {
                let measurement_context =
                    PromptProcessingMeasurementContext::with_position_independent_execution_profile(
                        persisted_context_bucket.exact_measurement_context_identifier,
                        persisted_context_bucket.position_independent_execution_profile_identifier,
                    );
                let candidate_statistics = persisted_context_bucket
                    .candidates
                    .into_iter()
                    .map(
                        |persisted_candidate_statistics| CandidateChunkStatistics {
                            measurements: persisted_candidate_statistics
                                .measurements
                                .into_iter()
                                .map(|persisted_measurement| CandidateChunkMeasurement {
                                    processed_prompt_token_count: persisted_measurement
                                        .processed_prompt_token_count,
                                    forward_elapsed_millis: persisted_measurement
                                        .forward_elapsed_millis,
                                    next_measurement_context:
                                        PromptProcessingMeasurementContext::with_position_independent_execution_profile(
                                            persisted_measurement
                                                .next_exact_measurement_context_identifier,
                                            persisted_measurement
                                                .next_position_independent_execution_profile_identifier,
                                        ),
                                    measurement_sequence: persisted_measurement
                                        .measurement_sequence,
                                })
                                .collect(),
                            last_measured_selection_sequence: persisted_candidate_statistics
                                .last_measured_selection_sequence,
                        },
                    )
                    .collect();
                (
                    measurement_context,
                    ContextCandidateStatistics {
                        candidate_statistics,
                    },
                )
            })
            .collect();
        PromptProcessingChunkSizeOptimizer::new_from_persisted_state(
            candidate_chunk_size_tokens,
            self.maximum_retained_measurements_per_candidate_and_context,
            self.selection_sequence,
            self.measurement_sequence,
            context_statistics,
        )
    }
}
