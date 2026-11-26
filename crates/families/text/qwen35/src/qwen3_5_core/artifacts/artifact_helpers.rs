use std::collections::HashMap;
use std::os::unix::fs::FileExt;

use crate::artifact_validation::{
    ArtifactValidationError, RequiredFileProfile, ValidatedRequiredFile,
};

use super::MAXIMUM_INDEX_BYTES;

pub(super) fn required_file(file_name: &str) -> RequiredFileProfile {
    RequiredFileProfile {
        file_name: file_name.to_owned(),
        size_bytes: 0,
    }
}

pub(super) fn captured_required_file_bytes<'a>(
    required_files: &'a HashMap<String, ValidatedRequiredFile>,
    file_name: &str,
) -> Result<&'a [u8], ArtifactValidationError> {
    required_files
        .get(file_name)
        .and_then(ValidatedRequiredFile::captured_bytes)
        .ok_or_else(|| ArtifactValidationError::ProfileMissingRequiredFile {
            file_name: file_name.to_owned(),
        })
}

pub(super) fn read_required_file_bytes(
    required_file: &ValidatedRequiredFile,
) -> Result<Vec<u8>, ArtifactValidationError> {
    let file_size = usize::try_from(required_file.size_bytes()).map_err(|_| {
        ArtifactValidationError::CapturedRequiredFileTooLarge {
            file_name: required_file.file_name().to_owned(),
            actual_size_bytes: required_file.size_bytes(),
            maximum_size_bytes: MAXIMUM_INDEX_BYTES as u64,
        }
    })?;
    if file_size > MAXIMUM_INDEX_BYTES {
        return Err(ArtifactValidationError::CapturedRequiredFileTooLarge {
            file_name: required_file.file_name().to_owned(),
            actual_size_bytes: required_file.size_bytes(),
            maximum_size_bytes: MAXIMUM_INDEX_BYTES as u64,
        });
    }
    let mut file_bytes = vec![0_u8; file_size];
    let mut completed_bytes = 0_usize;
    while completed_bytes < file_bytes.len() {
        let bytes_read = required_file
            .file()
            .read_at(&mut file_bytes[completed_bytes..], completed_bytes as u64)
            .map_err(
                |source| ArtifactValidationError::ReadRequiredFileForStructuralValidation {
                    file_name: required_file.file_name().to_owned(),
                    source,
                },
            )?;
        if bytes_read == 0 {
            return Err(
                ArtifactValidationError::ReadRequiredFileForStructuralValidation {
                    file_name: required_file.file_name().to_owned(),
                    source: std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "validated required file ended before its expected size",
                    ),
                },
            );
        }
        completed_bytes += bytes_read;
    }
    Ok(file_bytes)
}

/// The model's ID: its Hugging Face snapshot's repository, or its folder's
/// name in a plain library.
pub(super) fn artifact_model_id(model_directory: &std::path::Path) -> String {
    crate::artifact_validation::hugging_face_snapshot_model_id(model_directory).unwrap_or_else(
        || {
            model_directory
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unknown".to_owned())
        },
    )
}

/// One safetensors file a shard index names, opened and its header parsed.
pub(super) fn parsed_index_source(
    model_directory: &std::path::Path,
    file_name: &str,
    source_id: crate::artifact_validation::TensorSourceId,
) -> Result<
    crate::artifact_validation::ValidatedSafetensorsSource,
    crate::artifact_validation::ArtifactValidationError,
> {
    let required_file_entry = crate::artifact_validation::validate_required_file(
        model_directory,
        &required_file(file_name),
    )?;
    crate::artifact_validation::ValidatedSafetensorsSource::parse(source_id, required_file_entry)
}

/// Adds `generation_config.json` to the files a model must validate when it
/// ships one: optional, but checked like the rest when present.
pub(super) fn push_optional_generation_config(
    required_file_profiles: &mut Vec<RequiredFileProfile>,
    model_directory: &std::path::Path,
) {
    if model_directory.join("generation_config.json").is_file() {
        required_file_profiles.push(required_file("generation_config.json"));
    }
}

/// The shard index of an index document an artifact without one is given,
/// once the configuration has learned from its tensor names which modules
/// are stored unquantized.
pub(super) fn shard_index_from_synthesized_document(
    index_bytes: &[u8],
    config: &mut super::Qwen3_5Config,
) -> Result<super::Qwen3_5ShardIndex, super::Qwen3_5ArtifactValidationError> {
    let canonical_model_tensor_names =
        super::Qwen3_5ShardIndex::extract_language_and_vision_tensor_names_from_json(index_bytes)?;
    config.resolve_unquantized_modules_from_shard_index(&canonical_model_tensor_names);
    let language_tensor_profiles = super::tensor_spec::qwen3_5_language_tensor_profiles(config)?;
    Ok(super::Qwen3_5ShardIndex::from_json_bytes(
        index_bytes,
        &language_tensor_profiles,
    )?)
}
