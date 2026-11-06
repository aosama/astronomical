//! What the loaded model's Neural Engine prefill is actually doing.

use serde::{Deserialize, Serialize};

/// Reported from what happened to the loaded model's programs, so a user who enabled
/// the setting can see whether it took effect and, if not, why serving stayed on the
/// GPU. A failure later in a request moves prefill to the GPU and logs a warning, but
/// does not change this status, which stays as it was until the next load.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum NeuralEnginePrefillStatus {
    Off,
    /// The model is already serving on the GPU while its programs are prepared
    /// beside it. Counts every program of the set, so a reader sees one figure
    /// rise to its total instead of a count that restarts per program family.
    Preparing {
        completed_program_count: u32,
        total_program_count: u32,
        elapsed_ms: u64,
        /// Absent until one program has finished, because there is no rate to
        /// extrapolate from before that. A hint for a person, never a deadline.
        estimated_remaining_ms: Option<u64>,
    },
    Active {
        layer_program_count: u32,
        neural_engine_tokens_per_chunk: u32,
        chunk_token_count: u32,
        estimated_memory_bytes: u64,
    },
    /// Enabled but not running. `reason` is bounded and carries no local path.
    Unavailable {
        reason: String,
    },
}

#[cfg(test)]
#[path = "neural_engine_prefill_status_tests.rs"]
mod tests;
