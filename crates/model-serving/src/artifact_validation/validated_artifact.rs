use std::fs::{self, File};
use std::os::unix::fs::MetadataExt;

use ::safetensors::Dtype;

use super::ArtifactValidationError;
use super::raw_safetensors_inventory::read_raw_safetensors_inventory;

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
    validated_required_file: ValidatedRequiredFile,
}

/// Integration-test projection of one crate-private raw SafeTensors inventory.
#[doc(hidden)]
#[derive(Debug)]
pub struct RawSafetensorsInventoryForTests {
    /// Deterministic raw tensor descriptors with metadata excluded.
    pub tensor_descriptors: Vec<RawSafetensorsTensorDescriptorForTests>,
    /// Checked bytes covered by all tensor declarations in the shard.
    pub shard_payload_bytes: u64,
}

/// Integration-test projection of one crate-private raw tensor descriptor.
#[doc(hidden)]
#[derive(Debug)]
pub struct RawSafetensorsTensorDescriptorForTests {
    /// Unmodified tensor key from the raw header.
    pub tensor_name: String,
    /// Actual SafeTensors storage dtype.
    pub dtype: Dtype,
    /// Actual shape from the raw header.
    pub shape: Vec<usize>,
    /// Inclusive absolute source-file byte offset.
    pub data_start_offset_bytes: u64,
    /// Exclusive absolute source-file byte offset.
    pub data_end_offset_bytes: u64,
    /// Checked tensor payload byte count.
    pub tensor_payload_bytes: u64,
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
            validated_required_file: self,
        })
    }
}

impl ValidatedWeightsFile {
    /// Transfers the validated read-only descriptor to its runtime owner.
    pub fn into_file(self) -> File {
        self.validated_required_file.file
    }

    /// Returns the byte length of the validated file.
    pub const fn size_bytes(&self) -> u64 {
        self.validated_required_file.size_bytes
    }

    /// Returns a strict family-neutral inventory from this retained descriptor.
    pub(crate) fn read_raw_safetensors_inventory(
        &self,
    ) -> Result<super::RawSafetensorsInventory, ArtifactValidationError> {
        read_raw_safetensors_inventory(&self.validated_required_file)
    }

    /// Integration-test seam for the crate-private raw inventory owner.
    #[doc(hidden)]
    pub fn read_raw_safetensors_inventory_for_tests(
        &self,
    ) -> Result<RawSafetensorsInventoryForTests, ArtifactValidationError> {
        let raw_inventory = self.read_raw_safetensors_inventory()?;
        let tensor_descriptors = raw_inventory
            .tensor_descriptors
            .into_iter()
            .map(|tensor_descriptor| RawSafetensorsTensorDescriptorForTests {
                tensor_name: tensor_descriptor.tensor_name,
                dtype: tensor_descriptor.dtype,
                shape: tensor_descriptor.shape,
                data_start_offset_bytes: tensor_descriptor.data_start_offset_bytes,
                data_end_offset_bytes: tensor_descriptor.data_end_offset_bytes,
                tensor_payload_bytes: tensor_descriptor.tensor_payload_bytes,
            })
            .collect();
        Ok(RawSafetensorsInventoryForTests {
            tensor_descriptors,
            shard_payload_bytes: raw_inventory.shard_payload_bytes,
        })
    }

    /// Integration-test seam for the shared bounded retained-descriptor reader.
    #[doc(hidden)]
    pub fn read_bounded_bytes_for_tests(
        &self,
        maximum_size_bytes: u64,
    ) -> Result<Vec<u8>, ArtifactValidationError> {
        // Keep tests on the production reader so path-retention and typed-error
        // behavior are exercised together rather than copied into test code.
        super::required_files::read_bounded_required_file_bytes(
            &self.validated_required_file,
            maximum_size_bytes,
        )
    }
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
