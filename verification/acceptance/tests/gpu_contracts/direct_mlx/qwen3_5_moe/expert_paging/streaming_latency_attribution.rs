//! Bottom-of-stack latency attribution for SSD expert streaming.
//!
//! These probes drive the storage-to-weights boundary directly — safetensors
//! header plan, bounded page manifest, positional MLX read, forced
//! materialization — against the installed large sparse MoE artifact. No
//! engine, no IPC (Inter-Process Communication), no REST boundary. The goal is
//! to attribute where paged-prefill time goes: disk read, MLX graph
//! materialization, or the higher-level scheduling that re-issues loads.
//!
//! Each probe reports wall time, effective gigabytes per second (decimal SI:
//! 1 GB = 1,000,000,000 bytes), positional read-call counts, and the overlap
//! ratio between cumulative read time and wall time (how parallel the reads
//! actually were). Every probe enforces the 60-second ceiling by asserting its
//! own elapsed time.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use astronomical_model_serving::{
    Qwen3_5Config, build_quantized_expert_layer_plan,
    build_quantized_expert_page_manifest_from_plan, load_quantized_expert_page,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime, PositionalFileReadMetrics};
use serial_test::serial;

/// All probes must finish within the standard test ceiling.
const PROBE_CEILING: Duration = Duration::from_secs(60);
/// Headroom for the complete-layer probe: one full 8-bit expert layer is
/// roughly 0.8 GB, and evaluation may transiently hold copies.
const PROBE_ACTIVE_MEMORY_LIMIT_BYTES: usize = 32 * 1_000_000_000;
const PROBE_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES: usize = 512 * 1_000_000;
const MODEL_INDEX_FILE_NAME: &str = "model.safetensors.index.json";

struct StreamingProbeFixture {
    model_directory: PathBuf,
    config: Qwen3_5Config,
    weight_map: HashMap<String, String>,
}

/// Resolves the installed artifact through the development configuration so no
/// developer path is embedded here.
fn installed_large_sparse_moe_fixture() -> StreamingProbeFixture {
    let model_directory = crate::common::configured_large_sparse_moe_model_directory();
    let config_bytes = std::fs::read(model_directory.join("config.json"))
        .expect("the installed large sparse MoE config.json should be readable");
    let config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the installed large sparse MoE config should validate");
    let index_bytes = std::fs::read(model_directory.join(MODEL_INDEX_FILE_NAME))
        .expect("the installed large sparse MoE shard index should be readable");
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
fn probe_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            PROBE_ACTIVE_MEMORY_LIMIT_BYTES,
            PROBE_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the streaming probe memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

struct PageLoadMeasurement {
    elapsed: Duration,
    payload_bytes: u64,
    read_call_count: u64,
    read_byte_count: u64,
    total_read_elapsed: Duration,
    maximum_read_elapsed: Duration,
    maximum_concurrent_read_count: u64,
}

impl PageLoadMeasurement {
    fn report(&self, label: &str) {
        let elapsed_seconds = self.elapsed.as_secs_f64();
        let effective_gigabytes_per_second =
            (self.payload_bytes as f64 / 1e9) / elapsed_seconds.max(f64::EPSILON);
        let average_read_milliseconds = if self.read_call_count == 0 {
            0.0
        } else {
            self.total_read_elapsed.as_secs_f64() * 1_000.0 / self.read_call_count as f64
        };
        let overlap_ratio = if elapsed_seconds > f64::EPSILON {
            self.total_read_elapsed.as_secs_f64() / elapsed_seconds
        } else {
            0.0
        };
        eprintln!(
            "[ssd-streaming-probe {label}] elapsed_ms={} payload_gb={:.3} rate_gb_per_s={:.3} read_calls={} read_gb={:.3} avg_read_ms={:.3} max_read_ms={:.3} max_concurrent_reads={} read_to_wall_overlap={:.2}",
            self.elapsed.as_millis(),
            self.payload_bytes as f64 / 1e9,
            effective_gigabytes_per_second,
            self.read_call_count,
            self.read_byte_count as f64 / 1e9,
            average_read_milliseconds,
            self.maximum_read_elapsed.as_secs_f64() * 1_000.0,
            self.maximum_concurrent_read_count,
            overlap_ratio,
        );
    }

    fn assert_within_ceiling(&self, label: &str) {
        assert!(
            self.elapsed <= PROBE_CEILING,
            "{label} exceeded the 60-second probe ceiling: {} ms",
            self.elapsed.as_millis()
        );
    }
}

/// Loads one page of experts for one layer and forces materialization so every
/// lazy positional read is charged to this measurement.
fn measure_page_load(
    runtime: &MlxRuntime,
    fixture: &StreamingProbeFixture,
    layer_index: usize,
    expert_ids: &[usize],
) -> PageLoadMeasurement {
    let layer_prefix = format!("language_model.model.layers.{layer_index}.mlp");
    let layer_plan = build_quantized_expert_layer_plan(
        &fixture.model_directory,
        &fixture.weight_map,
        &layer_prefix,
        &fixture.config,
    )
    .expect("the installed artifact layer plan should build");
    let page_manifest = build_quantized_expert_page_manifest_from_plan(&layer_plan, expert_ids)
        .expect("the routed page manifest should build");
    let read_metrics = Arc::new(PositionalFileReadMetrics::default());
    let probe_started_at = Instant::now();
    let loaded_tensors =
        load_quantized_expert_page(runtime, &page_manifest, Some(Arc::clone(&read_metrics)))
            .expect("the bounded page load should succeed");
    let array_references: Vec<_> = loaded_tensors.values().collect();
    runtime
        .evaluate_arrays(&array_references)
        .expect("the loaded page should materialize");
    let elapsed = probe_started_at.elapsed();
    let snapshot = read_metrics.snapshot();
    PageLoadMeasurement {
        elapsed,
        payload_bytes: page_manifest.payload_byte_count,
        read_call_count: snapshot.read_call_count,
        read_byte_count: snapshot.read_byte_count,
        total_read_elapsed: Duration::from_nanos(snapshot.total_read_elapsed_nanoseconds),
        maximum_read_elapsed: Duration::from_nanos(snapshot.maximum_read_elapsed_nanoseconds),
        maximum_concurrent_read_count: snapshot.maximum_concurrent_read_count,
    }
}

#[test]
#[ignore = "streams one routed page from the installed large sparse MoE and attributes read latency"]
#[serial]
fn should_attribute_one_routed_page_load_cold_then_page_cache_warm() {
    let fixture = installed_large_sparse_moe_fixture();
    let runtime = probe_runtime();
    let experts_per_token = fixture.config.experts_per_token() as usize;
    let routed_expert_ids: Vec<usize> = (0..experts_per_token).collect();

    let cold = measure_page_load(&runtime, &fixture, 0, &routed_expert_ids);
    cold.report("routed-cold");
    cold.assert_within_ceiling("routed-cold");
    // The immediately repeated load answers whether the macOS page cache, not
    // the SSD, serves subsequent requests for the same ranges — separating
    // storage bandwidth from code overhead.
    let warm = measure_page_load(&runtime, &fixture, 0, &routed_expert_ids);
    warm.report("routed-warm");
    warm.assert_within_ceiling("routed-warm");

    assert!(
        cold.payload_bytes > 0,
        "a routed page must carry payload bytes"
    );
    assert!(
        cold.read_byte_count >= cold.payload_bytes,
        "materialization must read at least the page payload"
    );
}

/// Loads and materializes one page, returning its tensors so a caller can hold
/// them resident as memory pressure.
fn load_page_tensors_for_hold(
    runtime: &MlxRuntime,
    fixture: &StreamingProbeFixture,
    layer_index: usize,
    expert_ids: &[usize],
) -> HashMap<String, astronomical_mlx_c_rust::MlxArray> {
    let layer_prefix = format!("language_model.model.layers.{layer_index}.mlp");
    let layer_plan = build_quantized_expert_layer_plan(
        &fixture.model_directory,
        &fixture.weight_map,
        &layer_prefix,
        &fixture.config,
    )
    .expect("the installed artifact layer plan should build");
    let page_manifest = build_quantized_expert_page_manifest_from_plan(&layer_plan, expert_ids)
        .expect("the held page manifest should build");
    let loaded_tensors = load_quantized_expert_page(runtime, &page_manifest, None)
        .expect("the held page load should succeed");
    let array_references: Vec<_> = loaded_tensors.values().collect();
    runtime
        .evaluate_arrays(&array_references)
        .expect("the held page should materialize");
    for array in array_references {
        array
            .detach()
            .expect("the held page should detach from its lazy graph");
    }
    loaded_tensors
}

/// How many bytes of complete expert layers to hold while re-measuring reads.
const PRESSURE_HOLD_TARGET_BYTES: u64 = 27 * 1_000_000_000;

#[test]
#[ignore = "streams one complete expert layer from the installed large sparse MoE and reports the effective rate, unpressured then under a 27 GB hold"]
#[serial]
fn should_attribute_complete_layer_streaming_rate() {
    let fixture = installed_large_sparse_moe_fixture();
    let runtime = probe_runtime();
    let expert_capacity = fixture.config.expert_count() as usize;
    let every_expert_id: Vec<usize> = (0..expert_capacity).collect();

    let unpressured = measure_page_load(&runtime, &fixture, 0, &every_expert_id);
    unpressured.report("complete-layer-unpressured");
    unpressured.assert_within_ceiling("complete-layer-unpressured");

    // Reproduce serving memory pressure: paged prefill seats roughly 27 GB of
    // experts before its later chunks stream, and the attribution records show
    // those later reads running several times slower than the first chunk's.
    // Holding a comparable materialized allocation isolates whether the
    // reader's per-call latency collapses under active-memory pressure.
    let held_layer_count = (PRESSURE_HOLD_TARGET_BYTES / unpressured.payload_bytes.max(1)) as usize;
    let mut pressure_hold: Vec<HashMap<String, astronomical_mlx_c_rust::MlxArray>> = Vec::new();
    for held_layer_index in 1..=held_layer_count {
        pressure_hold.push(load_page_tensors_for_hold(
            &runtime,
            &fixture,
            held_layer_index,
            &every_expert_id,
        ));
    }
    eprintln!(
        "[ssd-streaming-probe complete-layer] holding_gb={:.1} held_layers={held_layer_count}",
        held_layer_count as f64 * unpressured.payload_bytes as f64 / 1e9
    );
    let pressured = measure_page_load(&runtime, &fixture, 0, &every_expert_id);
    pressured.report("complete-layer-pressured");
    pressured.assert_within_ceiling("complete-layer-pressured");
    drop(pressure_hold);

    let layer_count = fixture.config.layer_count() as u64;
    let projected_whole_model_seconds = unpressured.elapsed.as_secs_f64() * layer_count as f64;
    eprintln!(
        "[ssd-streaming-probe complete-layer] layer_count={layer_count} projected_whole_model_expert_stream_seconds={:.1}",
        projected_whole_model_seconds
    );
    assert!(
        unpressured.payload_bytes > 0,
        "a complete layer page must carry payload bytes"
    );
}

#[test]
#[ignore = "reloads one routed page three times without retention to expose the re-stream multiplier"]
#[serial]
fn should_attribute_repeated_page_reloads_without_retention() {
    let fixture = installed_large_sparse_moe_fixture();
    let runtime = probe_runtime();
    let experts_per_token = fixture.config.experts_per_token() as usize;
    let routed_expert_ids: Vec<usize> = (0..experts_per_token).collect();

    for reload_index in 1..=3 {
        let reload = measure_page_load(&runtime, &fixture, 0, &routed_expert_ids);
        reload.report(&format!("reload-{reload_index}"));
        reload.assert_within_ceiling("reload");
        // A same-layer reload reads the same byte ranges: without retention at
        // this boundary the positional read count must repeat, not shrink.
        assert!(reload.read_call_count > 0, "every reload must issue reads");
    }
}
