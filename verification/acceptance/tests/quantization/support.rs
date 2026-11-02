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

/// Progress lines on standard error, so a long journey shows its ETA.
pub(crate) struct StandardErrorProgress;

impl astronomical_quantization::progress::ProgressReporter for StandardErrorProgress {
    fn report(&mut self, progress: &astronomical_quantization::progress::StageProgress) {
        eprintln!(
            "[{}] {}/{} ({:.0}%) elapsed {}s {:?} {}",
            progress.stage_name,
            progress.completed_work_units,
            progress.total_work_units,
            progress.percent_complete(),
            progress.elapsed.as_secs(),
            progress.estimate,
            progress.detail
        );
    }
}

/// A bf16 copy of `source` under `destination`: the source itself when it
/// is already unquantized, otherwise every module dequantized through the
/// requantizer with each decision forced to pass-through, the quantization
/// keys dropped from `config.json`, and the tokenizer copied. The installed
/// small Qwen3.5 checkpoints are 4-bit, and the exact sweep's bf16 reference
/// path needs a bf16 model.
pub(crate) fn bfloat16_copy_of(
    runtime: &MlxRuntime,
    source: &std::path::Path,
    destination: &std::path::Path,
) -> PathBuf {
    let config_text =
        std::fs::read_to_string(source.join("config.json")).expect("the source config is readable");
    let mut config: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&config_text).expect("the source config parses");
    if config.remove("quantization").is_none() {
        return source.to_path_buf();
    }
    config.remove("quantization_config");
    let inventory =
        astronomical_quantization::module_inventory::ModuleInventory::from_source_directory(
            source,
            astronomical_quantization::bit_width::GroupSize::new(64).expect("valid"),
        )
        .expect("the source inventories");
    let mut plan = astronomical_quantization::requantize::RequantizeTensorPlan::from_inventory(
        &inventory,
        &astronomical_quantization::requantize::BitPolicy::Uniform {
            bit_width: astronomical_quantization::bit_width::BitWidth::new(4).expect("valid"),
            group_size: astronomical_quantization::bit_width::GroupSize::new(64).expect("valid"),
        },
    );
    for entry in &mut plan.entries {
        entry.decision =
            astronomical_quantization::requantize::bit_policy::TensorEncodingDecision::PassThrough;
    }
    astronomical_quantization::requantize::driver::run_requantize(
        runtime,
        source,
        destination,
        &plan,
        astronomical_quantization::encoder::RangeSearchPolicy::default_policy(),
        astronomical_quantization::requantize::DEFAULT_SHARD_BYTES,
        &mut StandardErrorProgress,
    )
    .expect("the bf16 copy should be written");
    std::fs::write(
        destination.join("config.json"),
        serde_json::to_string_pretty(&config).expect("the config serializes"),
    )
    .expect("the bf16 config is writable");
    astronomical_quantization::pipeline::artifact_files::copy_source_files(
        source,
        destination,
        astronomical_quantization::requantize::DEFAULT_COPY_PATTERNS,
    )
    .expect("the tokenizer copies");
    destination.to_path_buf()
}

/// The journey's memory limits, for a probe loaded beside the journey's own
/// runtime (the runtime is process-global; both must agree).
pub(crate) fn journey_memory_limits() -> MlxMemoryLimits {
    MlxMemoryLimits::new(
        JOURNEY_ACTIVE_MEMORY_LIMIT_BYTES,
        JOURNEY_ACTIVE_MEMORY_LIMIT_BYTES,
    )
    .expect("the journey memory limits are valid")
}
