/// One persistent prompt-cache block could not be read or did not match the expected Qwen3.5-MoE layout.
///
/// These errors describe an untrusted on-disk artifact rather than a model
/// inference failure. The disk store uses them to reject and remove a bad
/// block, then lets the request use cold prompt processing.
#[derive(Debug, thiserror::Error)]
pub enum PersistentPromptCacheBlockError {
    #[error(
        "failed to read persistent prompt-cache block metadata at {persistent_prompt_cache_block_path:?}"
    )]
    ReadFileMetadata {
        persistent_prompt_cache_block_path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "failed to read persistent prompt-cache block header bytes at {persistent_prompt_cache_block_path:?}"
    )]
    ReadHeaderBytes {
        persistent_prompt_cache_block_path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "persistent prompt-cache block read offset overflowed at {persistent_prompt_cache_block_path:?}"
    )]
    ReadOffsetOverflow {
        persistent_prompt_cache_block_path: std::path::PathBuf,
    },
    #[error(
        "persistent prompt-cache block header at {persistent_prompt_cache_block_path:?} is {header_length_bytes} bytes, maximum {maximum_header_length_bytes}"
    )]
    HeaderLengthTooLarge {
        persistent_prompt_cache_block_path: std::path::PathBuf,
        header_length_bytes: u64,
        maximum_header_length_bytes: u64,
    },
    #[error(
        "persistent prompt-cache block at {persistent_prompt_cache_block_path:?} is truncated: expected {expected_minimum_bytes}, got {actual_file_size_bytes}"
    )]
    TruncatedFile {
        persistent_prompt_cache_block_path: std::path::PathBuf,
        expected_minimum_bytes: u64,
        actual_file_size_bytes: u64,
    },
    #[error(
        "persistent prompt-cache block header at {persistent_prompt_cache_block_path:?} is not valid JSON"
    )]
    InvalidHeaderJson {
        persistent_prompt_cache_block_path: std::path::PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error(
        "persistent prompt-cache block at {persistent_prompt_cache_block_path:?} is missing metadata field {field_name}"
    )]
    MissingMetadata {
        persistent_prompt_cache_block_path: std::path::PathBuf,
        field_name: &'static str,
    },
    #[error(
        "persistent prompt-cache block metadata field {field_name} at {persistent_prompt_cache_block_path:?} is invalid"
    )]
    InvalidMetadata {
        persistent_prompt_cache_block_path: std::path::PathBuf,
        field_name: &'static str,
        #[source]
        source: std::num::ParseIntError,
    },
    #[error(
        "persistent prompt-cache block format version is {actual_format_version}, expected {expected_format_version}"
    )]
    UnsupportedFormatVersion {
        actual_format_version: String,
        expected_format_version: String,
    },
    #[error("persistent prompt-cache block is for foreign model {actual_model_id}")]
    ForeignModel { actual_model_id: String },
    #[error("persistent prompt-cache block is for foreign model revision {actual_model_revision}")]
    ForeignModelRevision { actual_model_revision: String },
    #[error(
        "persistent prompt-cache block token count is {actual_block_token_count}, expected {expected_block_token_count}"
    )]
    BlockTokenCountMismatch {
        actual_block_token_count: usize,
        expected_block_token_count: usize,
    },
    #[error(
        "persistent prompt-cache block at {persistent_prompt_cache_block_path:?} is missing tensor {tensor_name}"
    )]
    MissingTensor {
        persistent_prompt_cache_block_path: std::path::PathBuf,
        tensor_name: String,
    },
    #[error(
        "persistent prompt-cache block tensor {tensor_name} at {persistent_prompt_cache_block_path:?} has dtype {actual_dtype}, expected {expected_dtype}"
    )]
    TensorDtypeMismatch {
        persistent_prompt_cache_block_path: std::path::PathBuf,
        tensor_name: String,
        expected_dtype: &'static str,
        actual_dtype: String,
    },
    #[error(
        "persistent prompt-cache block tensor {tensor_name} at {persistent_prompt_cache_block_path:?} has shape {actual_shape:?}, expected {expected_shape:?}"
    )]
    TensorShapeMismatch {
        persistent_prompt_cache_block_path: std::path::PathBuf,
        tensor_name: String,
        expected_shape: Vec<usize>,
        actual_shape: Vec<usize>,
    },
    #[error(
        "persistent prompt-cache block at {persistent_prompt_cache_block_path:?} has {actual_tensor_count} tensors, expected {expected_tensor_count}"
    )]
    UnexpectedTensorCount {
        persistent_prompt_cache_block_path: std::path::PathBuf,
        expected_tensor_count: usize,
        actual_tensor_count: usize,
    },
    #[error(
        "persistent prompt-cache block tensor {tensor_name} at {persistent_prompt_cache_block_path:?} has invalid data offsets [{start_offset}, {end_offset}]"
    )]
    InvalidDataOffsets {
        persistent_prompt_cache_block_path: std::path::PathBuf,
        tensor_name: String,
        start_offset: u64,
        end_offset: u64,
    },
    #[error(
        "persistent prompt-cache block tensor {tensor_name} at {persistent_prompt_cache_block_path:?} ends at {end_offset} bytes, beyond the {file_size_bytes} byte file"
    )]
    OffsetBeyondFile {
        persistent_prompt_cache_block_path: std::path::PathBuf,
        tensor_name: String,
        end_offset: u64,
        file_size_bytes: u64,
    },
}
