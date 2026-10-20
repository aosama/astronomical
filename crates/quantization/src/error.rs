//! The one error type every stage of the quantization pipeline returns.
//!
//! Each variant names the stage that failed and carries enough context for an
//! operator to act without reading the source. The variants are deliberately
//! specific about the silent failure modes OptiQ documents from the field: a
//! sweep that measured almost nothing, a target that cannot be reached because
//! the block floor overrides it, a tensor whose last dimension does not divide
//! the group size.

use std::path::PathBuf;

use thiserror::Error;

/// Failures across module inventory, calibration, sensitivity, allocation,
/// encoding, requantization, and artifact writing.
#[derive(Debug, Error)]
pub enum QuantizationError {
    #[error("the source directory {source_directory} has no *.safetensors shards")]
    NoSafetensorsShards { source_directory: PathBuf },

    #[error("the safetensors header of {shard_path} could not be parsed: {description}")]
    SafetensorsHeaderUnreadable {
        shard_path: PathBuf,
        description: String,
    },

    #[error("config.json at {config_path} could not be read or parsed: {description}")]
    ModelConfigUnreadable {
        config_path: PathBuf,
        description: String,
    },

    #[error(
        "bit width {bits} is not an MLX affine quantization width (2, 3, 4, 5, 6, 8) or the keep-bfloat16 tier (16)"
    )]
    UnsupportedBitWidth { bits: u32 },

    #[error("group size {group_size} is not an MLX affine quantization group size (32, 64, 128)")]
    UnsupportedGroupSize { group_size: u32 },

    #[error(
        "candidate bit widths must be non-empty, ascending, and unique; got {candidate_bit_widths:?}"
    )]
    InvalidCandidateBitWidths { candidate_bit_widths: Vec<u32> },

    #[error("the calibration mix at {mix_path} is unreadable: {description}")]
    CalibrationMixUnreadable {
        mix_path: PathBuf,
        description: String,
    },

    #[error(
        "the calibration mix produced only {available_token_count} tokens, fewer than one sequence of {sequence_length_tokens}"
    )]
    CalibrationMixTooShort {
        available_token_count: usize,
        sequence_length_tokens: usize,
    },

    #[error("the tokenizer failed to encode calibration text: {description}")]
    TokenizerFailure { description: String },

    /// OptiQ's `_require_measured_majority` guard. A name-matching failure
    /// between the running model and the bf16 shards is silent by
    /// construction, so a sweep that measured almost nothing must refuse to
    /// hand the allocator a table built from data it never collected.
    #[error(
        "the sensitivity sweep measured only {measured_module_count} of {total_module_count} modules ({skipped_parameter_share_percent:.0}% of quantizable weights had no source); refusing to allocate from data that was never collected"
    )]
    SensitivitySweepMeasuredTooLittle {
        measured_module_count: usize,
        total_module_count: usize,
        skipped_parameter_share_percent: f64,
    },

    #[error(
        "the keep-bfloat16 tier (16 bits) cannot be probed against a uniform quantized reference; use the bf16 reference"
    )]
    KeepBfloat16TierNeedsBfloat16Reference,

    #[error("the sensitivity probe model failed: {description}")]
    ProbeModelFailure { description: String },

    #[error("the sensitivity document at {document_path} could not be read: {description}")]
    SensitivityDocumentUnreadable {
        document_path: PathBuf,
        description: String,
    },

    #[error("the sensitivity checkpoint at {checkpoint_path} could not be written: {description}")]
    CheckpointUnwritable {
        checkpoint_path: PathBuf,
        description: String,
    },

    #[error(
        "tensor {tensor_name} has shape {shape:?}; only 2-D and 3-D tensors whose last dimension divides group size {group_size} are group-quantizable"
    )]
    TensorNotGroupQuantizable {
        tensor_name: String,
        shape: Vec<usize>,
        group_size: u32,
    },

    #[error("the MLX runtime failed during {operation}: {description}")]
    MlxRuntimeFailure {
        operation: &'static str,
        description: String,
    },

    #[error("writing the output artifact under {output_directory} failed: {description}")]
    ArtifactWriteFailure {
        output_directory: PathBuf,
        description: String,
    },

    #[error(
        "the output directory {output_directory} already exists and overwrite was not requested"
    )]
    OutputDirectoryExists { output_directory: PathBuf },
}
