//! Resumable sweep state on disk.
//!
//! A sweep on a model that streams from SSD runs for hours, so a killed run
//! must resume. The checkpoint is a JSON array of serialized
//! [`SensitivityResult`]s in OptiQ's `sensitivity_checkpoint.json` shape
//! (`layer_name`, `sensitivities` keyed by the bit width as a string,
//! `param_count`), so a checkpoint written by the Python tool resumes here and
//! vice versa.

use std::path::Path;

use crate::error::QuantizationError;

use super::SensitivityResult;

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod checkpoint_tests;

/// Loads the checkpoint, ignoring an unreadable file (a corrupt checkpoint
/// costs a re-sweep, never a refusal to run).
///
/// Work to port: read the JSON array; map each entry's `layer_name` →
/// `module_name`, `sensitivities` (string keys) → `kl_divergence_by_bit_width`
/// (validated `BitWidth` keys), `param_count` → `logical_parameter_count`.
/// Return an empty vector when the file is absent or malformed, logging why.
#[must_use]
pub fn load_checkpoint(checkpoint_path: &Path) -> Vec<SensitivityResult> {
    tracing::debug!(checkpoint_path = %checkpoint_path.display(), "loading the sensitivity checkpoint");
    todo!("read and translate the OptiQ checkpoint document, tolerating absence and corruption")
}

/// Persists results by merging with the existing file, so it only grows.
///
/// Work to port: a resumed run re-appends already-complete modules to its
/// in-memory results only as the loop passes them, so a periodic save (or a
/// kill) mid-loop could otherwise write fewer entries than the file held and
/// drop scored modules. Merge by module name (existing ∪ current), then write
/// to `<path>.tmp` and rename over the original so a crash mid-save cannot
/// corrupt it.
pub fn merge_and_save_checkpoint(
    checkpoint_path: &Path,
    results: &[SensitivityResult],
) -> Result<(), QuantizationError> {
    let _ = (checkpoint_path, results);
    todo!("merge by module name, serialize in the OptiQ shape, write temp then rename")
}

/// Keeps only the checkpointed entries that cover every requested width;
/// an entry from a sweep with a different ladder must be re-probed.
#[must_use]
pub fn complete_entries_for(
    checkpointed: Vec<SensitivityResult>,
    candidate_bits: &[crate::bit_width::BitWidth],
) -> Vec<SensitivityResult> {
    let _ = (checkpointed, candidate_bits);
    todo!("retain entries whose key set is a superset of candidate_bits")
}
