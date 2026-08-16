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
                        "validated required file ended before its certified size",
                    ),
                },
            );
        }
        completed_bytes += bytes_read;
    }
    Ok(file_bytes)
}
