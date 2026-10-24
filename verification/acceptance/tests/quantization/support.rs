//! Artifact discovery, runtime, and prompt helpers for the quantization
//! journeys. Never a hardcoded developer path.

use std::path::PathBuf;

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use astronomical_test_harness::test_run::TestRunContext;

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
    TestRunContext::current().required_model_directory()
}

pub(crate) fn python_reference_directory() -> PathBuf {
    let reference_directory = TestRunContext::current().required_reference_artifact_path();
    assert!(
        reference_directory.is_dir(),
        "the Python reference artifact must be a directory"
    );
    reference_directory
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
