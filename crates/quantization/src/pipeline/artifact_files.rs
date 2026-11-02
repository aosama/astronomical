//! The pipeline's last stage: the documents beside the written shards and
//! the files copied verbatim from the source. Pure file work, so it runs
//! and is tested on any host.
//!
//! Ported from OptiQ's `_describe_artifact` and `_persist_sensitivity` (`optiq/models/llm.py`) and the copy step of `requantize_shardwise` (`optiq/core/shardwise.py`).

use std::collections::BTreeMap;
use std::path::Path;

use crate::allocation::AllocationResult;
use crate::artifact::{
    OptiQArtifactMetadata, OptiQSensitivityDocument, QuantizationConfigDocument,
};
use crate::bit_width::GroupSize;
use crate::error::QuantizationError;
use crate::sensitivity::{SensitivityMethod, SensitivityTable};

#[cfg(test)]
#[path = "artifact_files_tests.rs"]
mod artifact_files_tests;

/// Everything the document writers need.
pub struct ArtifactDocumentsRequest<'request> {
    pub source_directory: &'request Path,
    pub artifact_directory: &'request Path,
    pub base_model_name: &'request str,
    pub method: SensitivityMethod,
    pub allocation: &'request AllocationResult,
    pub sensitivity_table: &'request SensitivityTable,
    pub candidate_bits: &'request [u32],
    pub group_size: GroupSize,
    /// The `bits` default of the quantization document, OptiQ's `q_bits`
    /// (the ladder's lowest width).
    pub default_bits: u32,
    /// What the requantizer wrote, which the documents describe rather than
    /// the plan.
    pub written_quantization_by_module: &'request BTreeMap<String, (u32, u32)>,
}

/// Writes `config.json` with both quantization keys, `optiq/metadata.json`
/// reconciled against what was written, and `optiq/sensitivity.json`.
///
/// Returns whether the sensitivity document was written. Its failure is
/// logged and never fatal: the artifact loads without it, and failing an
/// hours-long conversion over a 50 KB sidecar would be the wrong trade.
pub fn write_artifact_documents(
    request: &ArtifactDocumentsRequest<'_>,
) -> Result<bool, QuantizationError> {
    QuantizationConfigDocument::from_written_modules(
        request.group_size.elements(),
        request.default_bits,
        request.written_quantization_by_module,
    )
    .write_into_config(
        &request.source_directory.join("config.json"),
        &request.artifact_directory.join("config.json"),
    )?;
    let mut metadata = OptiQArtifactMetadata::from_allocation(
        request.allocation,
        request.base_model_name,
        request.method,
    );
    metadata.describe_written_artifact(request.written_quantization_by_module);
    metadata.write(request.artifact_directory)?;
    let sensitivity_document = OptiQSensitivityDocument::from_table(
        request.sensitivity_table,
        request.base_model_name,
        request.candidate_bits,
        request.group_size,
    );
    match sensitivity_document.write(request.artifact_directory) {
        Ok(()) => Ok(true),
        Err(write_error) => {
            tracing::warn!(%write_error, "the sensitivity document was not written; the artifact is complete without it");
            Ok(false)
        }
    }
}

/// Copies every top-level source file matching one of `copy_patterns`
/// (`prefix*`, `*suffix`, or an exact name) into the artifact directory,
/// returning how many were copied. Directories are never copied.
pub fn copy_source_files(
    source_directory: &Path,
    artifact_directory: &Path,
    copy_patterns: &[&str],
) -> Result<usize, QuantizationError> {
    let directory_entries = match std::fs::read_dir(source_directory) {
        Ok(directory_entries) => directory_entries,
        Err(read_error) => return Err(copy_failure(artifact_directory, &read_error)),
    };
    let mut copied_file_count = 0;
    for directory_entry in directory_entries.flatten() {
        let entry_path = directory_entry.path();
        let file_name = directory_entry.file_name().to_string_lossy().into_owned();
        if !entry_path.is_file() || !matches_any_copy_pattern(&file_name, copy_patterns) {
            continue;
        }
        if let Err(copy_error) = std::fs::copy(&entry_path, artifact_directory.join(&file_name)) {
            return Err(copy_failure(artifact_directory, &copy_error));
        }
        copied_file_count += 1;
    }
    Ok(copied_file_count)
}

/// The shell-glob subset the copy list uses: one leading or trailing `*`.
#[must_use]
pub fn matches_any_copy_pattern(file_name: &str, copy_patterns: &[&str]) -> bool {
    for copy_pattern in copy_patterns {
        let is_match = if let Some(required_prefix) = copy_pattern.strip_suffix('*') {
            file_name.starts_with(required_prefix)
        } else if let Some(required_suffix) = copy_pattern.strip_prefix('*') {
            file_name.ends_with(required_suffix)
        } else {
            file_name == *copy_pattern
        };
        if is_match {
            return true;
        }
    }
    false
}

fn copy_failure(artifact_directory: &Path, failure: &std::io::Error) -> QuantizationError {
    QuantizationError::ArtifactWriteFailure {
        output_directory: artifact_directory.to_path_buf(),
        description: format!("copying source files failed: {failure}"),
    }
}
