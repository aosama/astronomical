//! Stable on-disk schema for persisting `PrefillChunckSizeOptimizer` state.
//!
//! This module defines `PersistedOptimizerState` and its nested types as a
//! serialization-friendly representation that is decoupled from the optimizer's
//! internal types. This allows the internal representation to evolve
//! independently of the on-disk format.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::warn;

use super::optimizer::{
    CandidatePrefillChunckObservation, CandidatePrefillChunckStatistics,
    ContextCandidateStatistics, PrefillChunckSizeOptimizer,
};
use super::{PrefillChunckSizeOptimizerContext, PrefillChunckSizeOptimizerError};

const FORMAT_VERSION: u32 = 3;
const STATE_FILE_NAME: &str = "prefill-chunck-size.json";

/// Stable on-disk representation of optimizer state.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct PersistedOptimizerState {
    format_version: u32,
    model_id: String,
    model_revision: String,
    candidate_prefill_chunck_tokens: Vec<usize>,
    trusted_observation_count: usize,
    sliding_window_observation_count: usize,
    drift_trigger_factor: u64,
    context_buckets: BTreeMap<String, PersistedContextBucket>,
}

/// Stable on-disk representation of per-context optimizer statistics.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct PersistedContextBucket {
    candidates: Vec<PersistedCandidateStatistics>,
    is_re_exploring: bool,
    re_exploration_remaining: usize,
    exploration_cursor: usize,
}

/// Stable on-disk representation of per-candidate observation statistics.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct PersistedCandidateStatistics {
    observations: Vec<PersistedObservation>,
}

/// Stable on-disk representation of a single candidate observation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct PersistedObservation {
    actual_prefill_chunck_tokens: usize,
    elapsed_millis: u64,
}

impl From<&CandidatePrefillChunckObservation> for PersistedObservation {
    fn from(observation: &CandidatePrefillChunckObservation) -> Self {
        Self {
            actual_prefill_chunck_tokens: observation.actual_prefill_chunck_tokens,
            elapsed_millis: observation.elapsed_millis,
        }
    }
}

impl From<PersistedObservation> for CandidatePrefillChunckObservation {
    fn from(observation: PersistedObservation) -> Self {
        Self {
            actual_prefill_chunck_tokens: observation.actual_prefill_chunck_tokens,
            elapsed_millis: observation.elapsed_millis,
        }
    }
}

impl From<&CandidatePrefillChunckStatistics> for PersistedCandidateStatistics {
    fn from(statistics: &CandidatePrefillChunckStatistics) -> Self {
        Self {
            observations: statistics
                .observations
                .iter()
                .map(PersistedObservation::from)
                .collect(),
        }
    }
}

impl From<PersistedCandidateStatistics> for CandidatePrefillChunckStatistics {
    fn from(persisted: PersistedCandidateStatistics) -> Self {
        Self {
            observations: persisted
                .observations
                .into_iter()
                .map(CandidatePrefillChunckObservation::from)
                .collect(),
        }
    }
}

impl From<&ContextCandidateStatistics> for PersistedContextBucket {
    fn from(statistics: &ContextCandidateStatistics) -> Self {
        Self {
            candidates: statistics
                .candidate_statistics
                .iter()
                .map(PersistedCandidateStatistics::from)
                .collect(),
            is_re_exploring: statistics.is_re_exploring,
            re_exploration_remaining: statistics.re_exploration_remaining,
            exploration_cursor: statistics.exploration_cursor,
        }
    }
}

impl From<PersistedContextBucket> for ContextCandidateStatistics {
    fn from(persisted: PersistedContextBucket) -> Self {
        Self {
            candidate_statistics: persisted
                .candidates
                .into_iter()
                .map(CandidatePrefillChunckStatistics::from)
                .collect(),
            is_re_exploring: persisted.is_re_exploring,
            re_exploration_remaining: persisted.re_exploration_remaining,
            exploration_cursor: persisted.exploration_cursor,
        }
    }
}

/// Validates that the persisted state matches the current model and optimizer
/// configuration. Returns `true` if the state is compatible, `false` otherwise.
fn state_matches_current_configuration(
    state: &PersistedOptimizerState,
    model_id: &str,
    model_revision: &str,
    candidate_prefill_chunck_tokens: &[usize],
    trusted_observation_count: usize,
    sliding_window_observation_count: usize,
    drift_trigger_factor: u64,
) -> bool {
    state.model_id == model_id
        && state.model_revision == model_revision
        && state.candidate_prefill_chunck_tokens == candidate_prefill_chunck_tokens
        && state.trusted_observation_count == trusted_observation_count
        && state.sliding_window_observation_count == sliding_window_observation_count
        && state.drift_trigger_factor == drift_trigger_factor
}

/// Saves optimizer state to the given directory using atomic write (temp file + rename).
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

    let state = PersistedOptimizerState::from_optimizer(optimizer, model_id, model_revision);
    let json = serde_json::to_string(&state).map_err(|serialization_error| {
        PrefillChunckSizeOptimizerError::OptimizerStateSerializationFailed {
            source: serialization_error,
        }
    })?;

    let state_file_path = optimizer_directory.join(STATE_FILE_NAME);
    let temp_file_path = state_file_path.with_extension("json.tmp");

    fs::write(&temp_file_path, &json).map_err(|io_error| {
        PrefillChunckSizeOptimizerError::OptimizerStateWriteFailed {
            path: temp_file_path.clone(),
            source: io_error,
        }
    })?;

    fs::rename(&temp_file_path, &state_file_path).map_err(|io_error| {
        PrefillChunckSizeOptimizerError::OptimizerStateRenameFailed {
            from: temp_file_path.clone(),
            to: state_file_path.clone(),
            source: io_error,
        }
    })?;

    Ok(())
}

/// Loads optimizer state from the given file path. Returns `Ok(None)` if the
/// file doesn't exist, is corrupt, or doesn't match the current model/configuration.
/// All I/O errors are logged at warn level and result in `Ok(None)` — the
/// optimizer is an accelerator, not a correctness gate.
pub(crate) fn load_optimizer_from_path(
    state_file_path: &Path,
    model_id: &str,
    model_revision: &str,
    candidate_prefill_chunck_tokens: Vec<usize>,
    trusted_observation_count: usize,
    sliding_window_observation_count: usize,
    drift_trigger_factor: u64,
) -> Result<Option<PrefillChunckSizeOptimizer>, PrefillChunckSizeOptimizerError> {
    if !state_file_path.exists() {
        return Ok(None);
    }

    let file_content = match fs::read_to_string(state_file_path) {
        Ok(content) => content,
        Err(io_error) => {
            warn!(
                path = %state_file_path.display(),
                error = %io_error,
                "Failed to read optimizer state file; starting fresh"
            );
            return Ok(None);
        }
    };

    if file_content.is_empty() {
        warn!(
            path = %state_file_path.display(),
            "Optimizer state file is empty; starting fresh"
        );
        return Ok(None);
    }

    let state: PersistedOptimizerState = match serde_json::from_str(&file_content) {
        Ok(parsed) => parsed,
        Err(parse_error) => {
            warn!(
                path = %state_file_path.display(),
                error = %parse_error,
                "Failed to parse optimizer state file; starting fresh"
            );
            return Ok(None);
        }
    };

    if state.format_version != FORMAT_VERSION {
        warn!(
            path = %state_file_path.display(),
            expected_version = FORMAT_VERSION,
            actual_version = state.format_version,
            "Optimizer state file has unknown format version; starting fresh"
        );
        return Ok(None);
    }

    if !state_matches_current_configuration(
        &state,
        model_id,
        model_revision,
        &candidate_prefill_chunck_tokens,
        trusted_observation_count,
        sliding_window_observation_count,
        drift_trigger_factor,
    ) {
        warn!(
            path = %state_file_path.display(),
            "Optimizer state file does not match current model or configuration; starting fresh"
        );
        return Ok(None);
    }

    let optimizer = state.into_optimizer(
        candidate_prefill_chunck_tokens,
        trusted_observation_count,
        sliding_window_observation_count,
        drift_trigger_factor,
    );

    Ok(Some(optimizer))
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
            .map(|(context, statistics)| {
                let bucket_key = context.context_identifier().to_string();
                (bucket_key, PersistedContextBucket::from(statistics))
            })
            .collect();

        Self {
            format_version: FORMAT_VERSION,
            model_id: model_id.to_string(),
            model_revision: model_revision.to_string(),
            candidate_prefill_chunck_tokens: optimizer.candidate_prefill_chunck_tokens().to_vec(),
            trusted_observation_count: optimizer.trusted_observation_count(),
            sliding_window_observation_count: optimizer.sliding_window_observation_count(),
            drift_trigger_factor: optimizer.drift_trigger_factor(),
            context_buckets,
        }
    }

    fn into_optimizer(
        self,
        candidate_prefill_chunck_tokens: Vec<usize>,
        trusted_observation_count: usize,
        sliding_window_observation_count: usize,
        drift_trigger_factor: u64,
    ) -> PrefillChunckSizeOptimizer {
        let context_statistics: BTreeMap<
            PrefillChunckSizeOptimizerContext,
            ContextCandidateStatistics,
        > = self
            .context_buckets
            .into_iter()
            .filter_map(|(bucket_key, persisted_bucket)| {
                let context_identifier: u64 = bucket_key.parse().ok()?;
                let context = PrefillChunckSizeOptimizerContext::new(context_identifier);
                let statistics = ContextCandidateStatistics::from(persisted_bucket);
                Some((context, statistics))
            })
            .collect();

        PrefillChunckSizeOptimizer::new_from_persisted_state(
            candidate_prefill_chunck_tokens,
            trusted_observation_count,
            sliding_window_observation_count,
            drift_trigger_factor,
            context_statistics,
        )
    }
}
