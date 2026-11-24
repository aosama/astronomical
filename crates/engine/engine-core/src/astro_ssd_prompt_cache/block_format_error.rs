/// One persistent model-state block could not be read or did not match its storage contract.
///
/// These errors describe an untrusted on-disk artifact rather than a model
/// inference failure. The disk store uses them to reject and remove a bad
/// block, then lets the request use cold prompt processing.
#[derive(Debug, thiserror::Error)]
pub enum AstroSsdPromptCacheBlockError {
    #[error(
        "failed to read persistent astro-ssd-prompt-cache block metadata at {astro_ssd_prompt_cache_block_path:?}"
    )]
    ReadFileMetadata {
        astro_ssd_prompt_cache_block_path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "failed to read persistent astro-ssd-prompt-cache block header bytes at {astro_ssd_prompt_cache_block_path:?}"
    )]
    ReadHeaderBytes {
        astro_ssd_prompt_cache_block_path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "persistent astro-ssd-prompt-cache block read offset overflowed at {astro_ssd_prompt_cache_block_path:?}"
    )]
    ReadOffsetOverflow {
        astro_ssd_prompt_cache_block_path: std::path::PathBuf,
    },
    #[error(
        "persistent astro-ssd-prompt-cache block header at {astro_ssd_prompt_cache_block_path:?} is {header_length_bytes} bytes, maximum {maximum_header_length_bytes}"
    )]
    HeaderLengthTooLarge {
        astro_ssd_prompt_cache_block_path: std::path::PathBuf,
        header_length_bytes: u64,
        maximum_header_length_bytes: u64,
    },
    #[error(
        "persistent astro-ssd-prompt-cache block at {astro_ssd_prompt_cache_block_path:?} is truncated: expected {expected_minimum_bytes}, got {actual_file_size_bytes}"
    )]
    TruncatedFile {
        astro_ssd_prompt_cache_block_path: std::path::PathBuf,
        expected_minimum_bytes: u64,
        actual_file_size_bytes: u64,
    },
    #[error(
        "persistent astro-ssd-prompt-cache block header at {astro_ssd_prompt_cache_block_path:?} is not valid JSON"
    )]
    InvalidHeaderJson {
        astro_ssd_prompt_cache_block_path: std::path::PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error(
        "persistent astro-ssd-prompt-cache block at {astro_ssd_prompt_cache_block_path:?} is missing metadata field {field_name}"
    )]
    MissingMetadata {
        astro_ssd_prompt_cache_block_path: std::path::PathBuf,
        field_name: &'static str,
    },
    #[error(
        "persistent astro-ssd-prompt-cache block metadata field {field_name} at {astro_ssd_prompt_cache_block_path:?} is invalid"
    )]
    InvalidMetadata {
        astro_ssd_prompt_cache_block_path: std::path::PathBuf,
        field_name: &'static str,
        #[source]
        source: std::num::ParseIntError,
    },
    #[error(
        "persistent astro-ssd-prompt-cache block format version is {actual_format_version}, expected {expected_format_version}"
    )]
    UnsupportedFormatVersion {
        actual_format_version: String,
        expected_format_version: String,
    },
    #[error(
        "persistent astro-ssd-prompt-cache block token count is {actual_block_token_count}, expected {expected_block_token_count}"
    )]
    BlockTokenCountMismatch {
        actual_block_token_count: usize,
        expected_block_token_count: usize,
    },
    #[error(
        "persistent astro-ssd-prompt-cache block at {astro_ssd_prompt_cache_block_path:?} is missing tensor {tensor_name}"
    )]
    MissingTensor {
        astro_ssd_prompt_cache_block_path: std::path::PathBuf,
        tensor_name: String,
    },
    #[error(
        "persistent astro-ssd-prompt-cache block tensor {tensor_name} at {astro_ssd_prompt_cache_block_path:?} has dtype {actual_dtype}, expected {expected_dtype}"
    )]
    TensorDtypeMismatch {
        astro_ssd_prompt_cache_block_path: std::path::PathBuf,
        tensor_name: String,
        expected_dtype: &'static str,
        actual_dtype: String,
    },
    #[error(
        "persistent astro-ssd-prompt-cache block tensor {tensor_name} at {astro_ssd_prompt_cache_block_path:?} has shape {actual_shape:?}, expected {expected_shape:?}"
    )]
    TensorShapeMismatch {
        astro_ssd_prompt_cache_block_path: std::path::PathBuf,
        tensor_name: String,
        expected_shape: Vec<usize>,
        actual_shape: Vec<usize>,
    },
    #[error(
        "persistent astro-ssd-prompt-cache block tensor {tensor_name} at {astro_ssd_prompt_cache_block_path:?} declares {actual_payload_byte_count} payload bytes, expected {expected_payload_byte_count}"
    )]
    TensorPayloadSizeMismatch {
        astro_ssd_prompt_cache_block_path: std::path::PathBuf,
        tensor_name: String,
        expected_payload_byte_count: u64,
        actual_payload_byte_count: u64,
    },
    #[error(
        "persistent astro-ssd-prompt-cache block at {astro_ssd_prompt_cache_block_path:?} has {actual_tensor_count} tensors, expected {expected_tensor_count}"
    )]
    UnexpectedTensorCount {
        astro_ssd_prompt_cache_block_path: std::path::PathBuf,
        expected_tensor_count: usize,
        actual_tensor_count: usize,
    },
    #[error(
        "persistent astro-ssd-prompt-cache block tensor {tensor_name} at {astro_ssd_prompt_cache_block_path:?} has invalid data offsets [{start_offset}, {end_offset}]"
    )]
    InvalidDataOffsets {
        astro_ssd_prompt_cache_block_path: std::path::PathBuf,
        tensor_name: String,
        start_offset: u64,
        end_offset: u64,
    },
    #[error(
        "persistent astro-ssd-prompt-cache block tensor {tensor_name} at {astro_ssd_prompt_cache_block_path:?} ends at {end_offset} bytes, beyond the {file_size_bytes} byte file"
    )]
    OffsetBeyondFile {
        astro_ssd_prompt_cache_block_path: std::path::PathBuf,
        tensor_name: String,
        end_offset: u64,
        file_size_bytes: u64,
    },
    #[error(
        "persistent astro-ssd-prompt-cache model-specific artifact at {astro_ssd_prompt_cache_block_path:?} is invalid: {description}"
    )]
    InvalidModelSpecificArtifact {
        astro_ssd_prompt_cache_block_path: std::path::PathBuf,
        description: String,
    },
}
