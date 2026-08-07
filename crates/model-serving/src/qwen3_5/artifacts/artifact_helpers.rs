use std::collections::{HashMap, HashSet};

use crate::artifact_validation::{
    ArtifactValidationError, RequiredFileProfile, ValidatedRequiredFile,
    read_validated_required_file_bytes,
};

use super::{MAXIMUM_INDEX_BYTES, Qwen3_5ShardIndex};

pub(super) fn required_file(file_name: &str) -> RequiredFileProfile {
    RequiredFileProfile {
        file_name: file_name.to_owned(),
        size_bytes: 0,
    }
}

pub(super) fn recognized_tensor_names(shard_index: &Qwen3_5ShardIndex) -> HashSet<&str> {
    shard_index
        .language_tensor_name_to_shard_file_name()
        .keys()
        .map(String::as_str)
        .chain(
            shard_index
                .mtp_tensor_name_to_shard_file_name()
                .keys()
                .map(String::as_str),
        )
        .chain(
            shard_index
                .vision_tensor_name_to_shard_file_name()
                .keys()
                .map(String::as_str),
        )
        .collect()
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
    read_validated_required_file_bytes(required_file, MAXIMUM_INDEX_BYTES as u64)
}
