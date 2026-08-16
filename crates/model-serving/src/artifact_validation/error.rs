use std::{io, path::PathBuf};

use ::safetensors::Dtype;
use thiserror::Error;

use super::TensorDtype;

/// Errors produced while failing closed on an invalid model artifact.
#[derive(Debug, Error)]
pub enum ArtifactValidationError {
    /// The supplied path is not a directory.
    #[error("model artifact directory does not exist or is not a directory: {model_directory:?}")]
    ModelDirectoryNotFound {
        /// Directory that was requested for validation.
        model_directory: PathBuf,
    },

    /// The artifact profile does not include a file needed by validation.
    #[error("artifact profile is missing required file {file_name}")]
    ProfileMissingRequiredFile {
        /// Required file name that was absent from the profile.
        file_name: String,
    },

    /// A profile file name is not one plain name relative to the model directory.
    #[error("artifact profile entry {file_name:?} must be one plain file name")]
    InvalidProfileFileName {
        /// Unsafe or ambiguous file name supplied by the profile.
        file_name: String,
    },

    /// A profile repeats a required file name and would make ownership ambiguous.
    #[error("artifact profile contains duplicate required file name {file_name}")]
    DuplicateProfileFileName {
        /// Required file name repeated by the profile.
        file_name: String,
    },

    /// A required file could not be inspected.
    #[error("failed to inspect required model file {file_name}")]
    InspectRequiredFile {
        /// Required file name from the artifact profile.
        file_name: String,
        /// Underlying filesystem error.
        #[source]
        source: io::Error,
    },

    /// A required file has an unexpected byte length.
    #[error(
        "required model file {file_name} has {actual_size_bytes} bytes, expected {expected_size_bytes} bytes"
    )]
    RequiredFileSizeMismatch {
        /// Required file name from the artifact profile.
        file_name: String,
        /// Size from the profile.
        expected_size_bytes: u64,
        /// Size observed on disk.
        actual_size_bytes: u64,
    },

    /// A required file is a symlink and would escape byte-identity validation.
    #[error("required model file {file_name} is a symlink")]
    RequiredFileIsSymlink {
        /// Required file name from the artifact profile.
        file_name: String,
    },

    /// A Hugging Face snapshot symlink resolves outside its cache entry's blobs directory.
    #[error(
        "Hugging Face snapshot file {file_name} resolves to {resolved_target_path:?}, outside expected blob directory {expected_blob_directory:?}"
    )]
    HuggingFaceSnapshotSymlinkEscapesBlobDirectory {
        /// Required file name from the artifact profile.
        file_name: String,
        /// Canonical target reached through the snapshot symlink.
        resolved_target_path: PathBuf,
        /// Canonical blob directory permitted for this cache entry.
        expected_blob_directory: PathBuf,
    },

    /// A required path is not a regular file.
    #[error("required model file {file_name} is not a regular file")]
    RequiredFileIsNotRegular {
        /// Required file name from the artifact profile.
        file_name: String,
    },

    /// A required file could not be read for bounded capture.
    #[error("failed to read required model file {file_name} for bounded capture")]
    ReadRequiredFileForCapture {
        /// Required file name from the artifact profile.
        file_name: String,
        /// Underlying filesystem error.
        #[source]
        source: io::Error,
    },

    /// A retained required-file descriptor could not supply its validated bytes.
    #[error("failed to read validated required model file {file_name} within its byte limit")]
    ReadBoundedRequiredFile {
        /// Required file name from the artifact profile.
        file_name: String,
        /// Underlying conversion, allocation, or filesystem error.
        #[source]
        source: io::Error,
    },

    /// A required file could not be read for bounded structural validation.
    #[error("failed to read required model file {file_name} for bounded structural validation")]
    ReadRequiredFileForStructuralValidation {
        /// Required file name from the artifact profile.
        file_name: String,
        /// Underlying filesystem error.
        #[source]
        source: io::Error,
    },

    /// A validated file changed identity between validation and reopening.
    #[error("validated required file {file_name} changed identity after validation")]
    ValidatedFileIdentityChanged {
        /// Required file name that changed after validation.
        file_name: String,
    },

    /// A config or tokenizer file exceeds the bounded capture limit.
    #[error(
        "required model file {file_name} is {actual_size_bytes} bytes, exceeding captured-file limit {maximum_size_bytes}"
    )]
    CapturedRequiredFileTooLarge {
        /// Profile-relative file name.
        file_name: String,
        /// Actual file size in bytes.
        actual_size_bytes: u64,
        /// Maximum captured size in bytes.
        maximum_size_bytes: u64,
    },

    /// A retained required file exceeds its caller's explicit read limit.
    #[error(
        "required model file {file_name} is {actual_size_bytes} bytes, exceeding bounded-read limit {maximum_size_bytes}"
    )]
    BoundedRequiredFileTooLarge {
        /// Profile-relative file name.
        file_name: String,
        /// Validated file size in bytes.
        actual_size_bytes: u64,
        /// Maximum size accepted by the caller.
        maximum_size_bytes: u64,
    },

    /// The safetensors length prefix could not be read from the weights file.
    #[error("failed to read safetensors length prefix from weight file {file_name}")]
    ReadSafetensorsLengthPrefix {
        /// Weight file name from the artifact profile.
        file_name: String,
        /// Underlying filesystem error.
        #[source]
        source: io::Error,
    },

    /// The safetensors header length exceeds the maximum accepted bound.
    #[error(
        "safetensors header length {header_length_bytes} exceeds maximum {maximum_header_length_bytes} in weight file {file_name}"
    )]
    SafetensorsHeaderLengthTooLarge {
        /// Weight file name from the artifact profile.
        file_name: String,
        /// Declared header length in bytes.
        header_length_bytes: u64,
        /// Maximum accepted header length in bytes.
        maximum_header_length_bytes: u64,
    },

    /// The safetensors file is too short for its declared header length.
    #[error(
        "safetensors file {file_name} is {actual_file_size_bytes} bytes, shorter than the {expected_minimum_bytes} bytes required by its header length"
    )]
    TruncatedSafetensorsFile {
        /// Weight file name from the artifact profile.
        file_name: String,
        /// Minimum file size required by the length prefix and header.
        expected_minimum_bytes: u64,
        /// Actual file size observed on disk.
        actual_file_size_bytes: u64,
    },

    /// The safetensors header bytes could not be read from the weights file.
    #[error("failed to read safetensors header from weight file {file_name}")]
    ReadSafetensorsHeader {
        /// Weight file name from the artifact profile.
        file_name: String,
        /// Underlying filesystem error.
        #[source]
        source: io::Error,
    },

    /// The safetensors header JSON could not be parsed.
    #[error("invalid safetensors header JSON in weight file {file_name}")]
    InvalidSafetensorsHeader {
        /// Weight file name from the artifact profile.
        file_name: String,
        /// JSON decoding error.
        #[source]
        source: serde_json::Error,
    },

    /// A safetensors tensor name is empty or exceeds the bounded header allowance.
    #[error(
        "safetensors tensor name has {tensor_name_length_bytes} bytes, expected 1..={maximum_tensor_name_length_bytes} bytes in weight file {file_name}"
    )]
    InvalidSafetensorsTensorName {
        /// Weight file name from the artifact profile.
        file_name: String,
        /// UTF-8 byte length of the invalid tensor name.
        tensor_name_length_bytes: u64,
        /// Maximum name length already bounded by the raw header limit.
        maximum_tensor_name_length_bytes: u64,
    },

    /// A safetensors tensor data offset extends beyond the weights file.
    #[error(
        "safetensors tensor {tensor_name} data end offset {data_end_offset} exceeds file size {file_size_bytes} in weight file {file_name}"
    )]
    SafetensorsOffsetBeyondFile {
        /// Weight file name from the artifact profile.
        file_name: String,
        /// Tensor name whose offset is out of bounds.
        tensor_name: String,
        /// Declared end offset of the tensor data within the data section.
        data_end_offset: u64,
        /// Total file size in bytes.
        file_size_bytes: u64,
    },

    /// The safetensors file contains bytes outside all declared tensor payloads.
    #[error(
        "safetensors payload length mismatch in {file_name}: declared {declared_payload_bytes} bytes, file contains {actual_payload_bytes} bytes"
    )]
    SafetensorsPayloadLengthMismatch {
        /// Weight file name from the artifact profile.
        file_name: String,
        /// Payload bytes covered by tensor metadata.
        declared_payload_bytes: u64,
        /// Actual bytes after the safetensors header.
        actual_payload_bytes: u64,
    },

    /// A safetensors tensor has invalid data offsets where the start exceeds the end.
    #[error(
        "safetensors tensor {tensor_name} has invalid data offsets: start {data_start_offset} exceeds end {data_end_offset} in weight file {file_name}"
    )]
    SafetensorsInvalidDataOffsets {
        /// Weight file name from the artifact profile.
        file_name: String,
        /// Tensor name with invalid offsets.
        tensor_name: String,
        /// Declared start offset of the tensor data.
        data_start_offset: u64,
        /// Declared end offset of the tensor data.
        data_end_offset: u64,
    },

    /// A safetensors tensor declares an unknown dtype string.
    #[error(
        "safetensors tensor {tensor_name} has unknown dtype {dtype_string} in weight file {file_name}"
    )]
    UnknownSafetensorsDtype {
        /// Weight file name from the artifact profile.
        file_name: String,
        /// Tensor name with the unknown dtype.
        tensor_name: String,
        /// The unrecognized dtype string from the header.
        dtype_string: String,
    },

    /// A certified tensor is absent from the safetensors file.
    #[error("tensor {tensor_name} is missing from the safetensors weight file {file_name}")]
    TensorMissing {
        /// Expected tensor name.
        tensor_name: String,
        /// Weight file name from the artifact profile.
        file_name: String,
    },

    /// The safetensors file contains a tensor absent from the certified profile.
    #[error("extra tensor {tensor_name} in the safetensors weight file")]
    UnexpectedTensor {
        /// Unexpected tensor name.
        tensor_name: String,
    },

    /// A tensor has a different dtype than the certified profile allows.
    #[error("tensor {tensor_name} dtype mismatch: expected {expected_dtype:?}, got {actual_dtype}")]
    TensorDtypeMismatch {
        /// Tensor name.
        tensor_name: String,
        /// Expected dtype.
        expected_dtype: TensorDtype,
        /// Actual safetensors dtype.
        actual_dtype: Dtype,
    },

    /// A tensor has a different shape than the certified profile allows.
    #[error(
        "tensor {tensor_name} shape mismatch: expected {expected_shape:?}, got {actual_shape:?}"
    )]
    TensorShapeMismatch {
        /// Tensor name.
        tensor_name: String,
        /// Expected shape.
        expected_shape: Vec<usize>,
        /// Actual shape.
        actual_shape: Vec<usize>,
    },

    /// Tensor payload sizes overflowed the validator's accounting type.
    #[error("safetensors payload byte count overflowed u64")]
    TensorPayloadSizeOverflow,
}
