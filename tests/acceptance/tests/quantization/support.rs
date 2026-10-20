//! Artifact discovery, runtime, and prompt helpers for the quantization
//! journeys. Never a hardcoded developer path.

use std::path::PathBuf;

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

/// The installed bf16 (or FP8) source snapshot: `config.json`, shards,
/// `tokenizer.json`. A small Qwen3.5 dense base keeps the exact sweep inside
/// the runner budget.
pub(crate) const SOURCE_MODEL_DIRECTORY_ENV: &str = "ASTRONOMICAL_QUANTIZATION_SOURCE_MODEL_DIR";
/// An artifact the Python tool (`optiq convert`, mlx-optiq 0.6.5) produced
/// from the same source with `--method static --candidate-bits 2,4,8
/// --target-bpw 4.5 --no-low-bit-search`, for the parity journey.
pub(crate) const PYTHON_REFERENCE_DIRECTORY_ENV: &str =
    "ASTRONOMICAL_QUANTIZATION_PYTHON_REFERENCE_DIR";

/// Generous for a small base plus the encoder's working copies, far below
/// any laptop ceiling.
const JOURNEY_ACTIVE_MEMORY_LIMIT_BYTES: usize = 24 * 1024 * 1024 * 1024;

/// The repository's prompt-processing chunk for reproducible results.
pub(crate) const PREFILL_CHUNK_TOKENS: usize = 2_048;

/// The Romeo and Juliet source every calibration and generation prompt is
/// cut from.
pub(crate) const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

pub(crate) fn source_model_directory() -> PathBuf {
    match std::env::var_os(SOURCE_MODEL_DIRECTORY_ENV) {
        Some(value) if !value.is_empty() => PathBuf::from(value),
        _ => panic!(
            "set {SOURCE_MODEL_DIRECTORY_ENV} to a local bf16 Qwen3.5 snapshot directory \
             (config.json, *.safetensors, tokenizer.json)"
        ),
    }
}

pub(crate) fn python_reference_directory() -> PathBuf {
    match std::env::var_os(PYTHON_REFERENCE_DIRECTORY_ENV) {
        Some(value) if !value.is_empty() => PathBuf::from(value),
        _ => panic!(
            "set {PYTHON_REFERENCE_DIRECTORY_ENV} to the artifact mlx-optiq 0.6.5 produced from \
             the same source (optiq convert --method static --candidate-bits 2,4,8 \
             --target-bpw 4.5 --no-low-bit-search)"
        ),
    }
}

pub(crate) fn journey_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            JOURNEY_ACTIVE_MEMORY_LIMIT_BYTES,
            JOURNEY_ACTIVE_MEMORY_LIMIT_BYTES,
        )
        .expect("the journey memory limits are valid"),
    )
    .expect("the MLX runtime should initialize")
}

/// A fresh output directory under the system temporary directory, named
/// uniquely per journey; the journey removes it when done.
pub(crate) fn fresh_output_directory(journey_name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("the clock is after the epoch")
        .as_nanos();
    let directory =
        std::env::temp_dir().join(format!("astronomical-quantization-{journey_name}-{nanos}"));
    std::fs::create_dir_all(&directory).expect("the output directory should be creatable");
    directory
}
