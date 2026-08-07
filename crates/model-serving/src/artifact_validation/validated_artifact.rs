use std::fs::{self, File};
use std::os::unix::fs::{FileExt, MetadataExt};

use super::ArtifactValidationError;

/// An open required-file descriptor whose identity was checked during validation.
#[derive(Debug)]
pub(crate) struct ValidatedRequiredFile {
    file: File,
    file_identity: ValidatedFileIdentity,
    file_name: String,
    size_bytes: u64,
    captured_bytes: Option<Vec<u8>>,
}

/// A validated model-weight descriptor ready for ownership transfer to MLX.
#[derive(Debug)]
pub struct ValidatedWeightsFile {
    file: File,
    size_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ValidatedFileIdentity {
    device_id: u64,
    inode: u64,
    size_bytes: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
}

impl ValidatedRequiredFile {
    pub(crate) fn new(
        file: File,
        file_identity: ValidatedFileIdentity,
        file_name: String,
        size_bytes: u64,
        captured_bytes: Option<Vec<u8>>,
    ) -> Self {
        Self {
            file,
            file_identity,
            file_name,
            size_bytes,
            captured_bytes,
        }
    }

    pub(crate) fn file_name(&self) -> &str {
        &self.file_name
    }

    pub(crate) const fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    pub(crate) fn captured_bytes(&self) -> Option<&[u8]> {
        self.captured_bytes.as_deref()
    }

    pub(crate) const fn file(&self) -> &File {
        &self.file
    }

    pub(crate) fn into_validated_weights_file(
        self,
    ) -> Result<ValidatedWeightsFile, ArtifactValidationError> {
        let file_metadata = self.file.metadata().map_err(|source| {
            ArtifactValidationError::InspectRequiredFile {
                file_name: self.file_name.clone(),
                source,
            }
        })?;
        if validated_file_identity(&file_metadata) != self.file_identity {
            return Err(ArtifactValidationError::ValidatedFileIdentityChanged {
                file_name: self.file_name,
            });
        }
        Ok(ValidatedWeightsFile {
            file: self.file,
            size_bytes: file_metadata.len(),
        })
    }
}

impl ValidatedWeightsFile {
    /// Transfers the validated read-only descriptor to its runtime owner.
    pub fn into_file(self) -> File {
        self.file
    }

    /// Returns the byte length of the validated file.
    pub const fn size_bytes(&self) -> u64 {
        self.size_bytes
    }
}

/// Reads a bounded structural file through its already-validated descriptor.
pub(crate) fn read_validated_required_file_bytes(
    validated_required_file: &ValidatedRequiredFile,
    maximum_size_bytes: u64,
) -> Result<Vec<u8>, ArtifactValidationError> {
    if validated_required_file.size_bytes() > maximum_size_bytes {
        return Err(ArtifactValidationError::CapturedRequiredFileTooLarge {
            file_name: validated_required_file.file_name().to_owned(),
            actual_size_bytes: validated_required_file.size_bytes(),
            maximum_size_bytes,
        });
    }
    let file_size = usize::try_from(validated_required_file.size_bytes()).map_err(|_| {
        ArtifactValidationError::CapturedRequiredFileTooLarge {
            file_name: validated_required_file.file_name().to_owned(),
            actual_size_bytes: validated_required_file.size_bytes(),
            maximum_size_bytes,
        }
    })?;
    let mut file_bytes = vec![0_u8; file_size];
    let mut completed_bytes = 0_usize;
    while completed_bytes < file_bytes.len() {
        let bytes_read = validated_required_file
            .file()
            .read_at(&mut file_bytes[completed_bytes..], completed_bytes as u64)
            .map_err(
                |source| ArtifactValidationError::ReadRequiredFileForStructuralValidation {
                    file_name: validated_required_file.file_name().to_owned(),
                    source,
                },
            )?;
        if bytes_read == 0 {
            return Err(
                ArtifactValidationError::ReadRequiredFileForStructuralValidation {
                    file_name: validated_required_file.file_name().to_owned(),
                    source: std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "validated file ended before its inspected size",
                    ),
                },
            );
        }
        completed_bytes += bytes_read;
    }
    Ok(file_bytes)
}

pub(crate) fn validated_file_identity(file_metadata: &fs::Metadata) -> ValidatedFileIdentity {
    ValidatedFileIdentity {
        device_id: file_metadata.dev(),
        inode: file_metadata.ino(),
        size_bytes: file_metadata.len(),
        modified_seconds: file_metadata.mtime(),
        modified_nanoseconds: file_metadata.mtime_nsec(),
    }
}
