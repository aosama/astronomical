//! The installed large sparse MoE streaming artifact and the MLX runtime the
//! expert-paging probes share.
//!
//! The MLX runtime is process-global, so every probe in this binary must
//! agree on one memory-limit shape; keeping the fixture here rather than in
//! one probe lets the probes depend on it instead of on each other.

use std::collections::HashMap;
use std::path::PathBuf;

use astronomical_model_serving::Qwen3_5Config;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

/// Headroom for the complete-layer probe: one full 8-bit expert layer is
/// roughly 0.8 GB, and evaluation may transiently hold copies.
const PROBE_ACTIVE_MEMORY_LIMIT_BYTES: usize = 32 * 1_000_000_000;
const PROBE_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES: usize = 512 * 1_000_000;
const MODEL_INDEX_FILE_NAME: &str = "model.safetensors.index.json";

pub(super) struct StreamingProbeFixture {
    pub(super) model_directory: PathBuf,
    pub(super) config: Qwen3_5Config,
    pub(super) weight_map: HashMap<String, String>,
}

/// Resolves the installed artifact from the `model_directory` a
/// `cargo xtask test-run --configuration` file names, else through the
/// development configuration, so no developer path is embedded here.
pub(super) fn installed_large_sparse_moe_streaming_fixture() -> StreamingProbeFixture {
    let model_directory = astronomical_test_harness::test_run::TestRunContext::current()
        .configuration
        .model_directory
        .unwrap_or_else(crate::common::configured_large_sparse_moe_streaming_model_directory);
    let config_bytes = std::fs::read(model_directory.join("config.json"))
        .expect("the installed large sparse MoE streaming config.json should be readable");
    let config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the installed large sparse MoE streaming config should validate");
    // A test-run configuration is shared by every case in a run, so a
    // `model_directory` chosen for another test (a dense or quantization
    // model) must be refused here, not deep inside a layer plan.
    assert!(
        config.expert_count() > 0,
        "{} is not a sparse mixture-of-experts model; the expert-paging probes need one",
        model_directory.display()
    );
    let index_bytes = std::fs::read(model_directory.join(MODEL_INDEX_FILE_NAME))
        .expect("the installed large sparse MoE streaming shard index should be readable");
    let index_document: serde_json::Value =
        serde_json::from_slice(&index_bytes).expect("the shard index should be valid JSON");
    let mut weight_map = HashMap::new();
    for (tensor_name, shard_file_name) in index_document
        .get("weight_map")
        .and_then(|weight_map| weight_map.as_object())
        .into_iter()
        .flatten()
    {
        if tensor_name.starts_with("language_model.") {
            weight_map.insert(
                tensor_name.clone(),
                shard_file_name
                    .as_str()
                    .expect("shard index entries map tensor names to file names")
                    .to_owned(),
            );
        }
    }
    StreamingProbeFixture {
        model_directory,
        config,
        weight_map,
    }
}

/// Runtime shared by every streaming probe in this binary: the MLX runtime is
/// process-global, so all probes must agree on one memory-limit shape. The
/// limit leaves room for the pressured probe's held layers plus the streamed
/// page, under the repository's 36 GB real-model ceiling.
pub(super) fn probe_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            PROBE_ACTIVE_MEMORY_LIMIT_BYTES,
            PROBE_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the streaming probe memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}
