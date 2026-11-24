use std::collections::HashMap;
use std::path::Path;

use crate::artifact_validation::{RequiredFileProfile, ValidatedRequiredFile};
use crate::qwen3_5_core::quantizations::producers::detect_quantization_producer_in_directory;
use crate::qwen3_5_core::quantizations::producers::optiq::metadata_location::{
    LEGACY_OPTIQ_METADATA_RELATIVE_PATH, OPTIQ_METADATA_RELATIVE_PATH,
    resolve_optiq_metadata_relative_path,
};

use super::artifact_helpers::{read_required_file_bytes, required_file};
use super::{OptiQMetadata, Qwen3_5ArtifactValidationError, Qwen3_5Config};

/// Adds the OptiQ metadata file to the required-file profiles when the
/// artifact carries one, preferring `optiq/metadata.json` over the legacy
/// root name so current artifacts get the config cross-check too.
///
/// The producer is logged here, at the one place every artifact load passes
/// through, so a mismatch report can be read against who wrote the artifact.
pub(super) fn register_optional_optiq_metadata(
    required_file_profiles: &mut Vec<RequiredFileProfile>,
    model_directory: &Path,
) {
    tracing::debug!(
        quantization_producer = ?detect_quantization_producer_in_directory(model_directory),
        "detected quantization producer"
    );
    if let Some(metadata_relative_path) = resolve_optiq_metadata_relative_path(model_directory) {
        required_file_profiles.push(required_file(metadata_relative_path));
    }
}

/// Cross-checks the registered OptiQ metadata against the parsed config.
/// Artifacts without OptiQ metadata (plain mlx-lm, oQ) pass untouched.
pub(super) fn validate_optional_optiq_metadata(
    required_files: &HashMap<String, ValidatedRequiredFile>,
    config: &Qwen3_5Config,
) -> Result<(), Qwen3_5ArtifactValidationError> {
    for metadata_relative_path in [
        OPTIQ_METADATA_RELATIVE_PATH,
        LEGACY_OPTIQ_METADATA_RELATIVE_PATH,
    ] {
        let Some(metadata_required_file) = required_files.get(metadata_relative_path) else {
            continue;
        };
        let metadata_bytes = read_required_file_bytes(metadata_required_file)?;
        OptiQMetadata::from_json_bytes(&metadata_bytes)?.validate_against_config(config)?;
        return Ok(());
    }
    Ok(())
}

#[cfg(test)]
mod optiq_metadata_validation_tests;
