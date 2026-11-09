//! Where a paged decode token's time goes, per residency strategy.
//!
//! Decode residency trades two per-token costs against each other: every layer
//! that is not seated complete pays a route copy to the host before its experts
//! can be chosen, and every routed expert missing from RAM pays a storage read.
//! This probe serves one Shakespeare request through the attributed in-process
//! engine (no IPC, no REST) under an MLX limit the test-run configuration
//! chooses, then prints both costs per decode token next to the decode wall
//! time, so a residency change can be judged by measured milliseconds rather
//! than by hit rate alone.
//!
//! The warm variant leaves the macOS file cache alone, so on a Mac with more
//! RAM than the limit most expert reads come from memory. The cold variant
//! keeps the model's weight files evicted for the whole run
//! (`cold_file_cache`), which is what the same reads cost on a Mac whose RAM
//! is the limit.
//!
//! Run through `cargo xtask test-run --configuration FILE`, whose
//! `model_directory` names the sparse model and whose
//! `paging_experiment.maximum_mlx_memory_gigabytes` sets the limit (the same
//! 2^30-byte gigabyte the app's memory setting uses), in a release build:
//! `cargo measure-decode-residency-attribution` (warm) and
//! `cargo measure-cold-decode-residency-attribution` (cold).

use std::path::Path;
use std::time::Duration;

use astronomical_ipc_protocol::RequestId;
use astronomical_runtime_integration::MlxMemoryLimits;
use astronomical_test_harness::test_run::TestRunContext;
use serde_json::Value;
use serial_test::serial;
use tokio::time::timeout;

use super::cold_file_cache::ColdFileCache;
use crate::serving_acceptance::support::performance_attribution::{
    counter_amount, create_attributed_engine, generation_report_for_request,
    load_engine_with_progress, read_attribution_report_documents, run_attributed_generation,
};

/// The repository's performance shape: a 500-in/50-out warmup, then 10,000
/// prompt tokens and 500 decode tokens measured. A short follow-up then
/// decodes on the residency the long request left behind; its report is
/// almost pure decode, so every operation in it can be read per decode token.
const WARMUP_PROMPT_TOKEN_COUNT: usize = 500;
const WARMUP_OUTPUT_TOKEN_COUNT: u16 = 50;
const MEASURED_PROMPT_TOKEN_COUNT: usize = 10_000;
const MEASURED_OUTPUT_TOKEN_COUNT: u16 = 500;
const WARMUP_REQUEST_ID: RequestId = RequestId::new(97_100);
const MEASURED_REQUEST_ID: RequestId = RequestId::new(97_101);
const FOLLOW_UP_PROMPT_TOKEN_COUNT: usize = 64;
const FOLLOW_UP_OUTPUT_TOKEN_COUNT: u16 = 300;
const FOLLOW_UP_REQUEST_ID: RequestId = RequestId::new(97_102);
/// Operations below this share of a decode token are left out of the
/// follow-up breakdown as noise.
const MINIMUM_REPORTED_MILLISECONDS_PER_TOKEN: f64 = 0.05;
/// Prefill operations below this total are left out as noise.
const MINIMUM_REPORTED_PREFILL_SECONDS: f64 = 0.05;
/// The repository's SSD-paging budget. On an 8-bit 35B model under a 16 GB
/// limit the warmup, a 10,000-token paged prefill and the decode finish in
/// about 90 seconds; a run that needs longer should be split into lower-level
/// probes rather than given more time.
const DECODE_ATTRIBUTION_TIMEOUT: Duration = Duration::from_secs(120);

/// Whether the model's weight files may stay in the macOS file cache.
#[derive(Clone, Copy, Debug)]
enum FileCacheCondition {
    Warm,
    Cold,
}

#[tokio::test]
#[ignore = "serves a 10,000-token Shakespeare request on an installed sparse model and attributes each decode token's time"]
#[serial]
async fn should_attribute_decode_token_time_to_route_copies_and_storage_reads() {
    timeout(
        DECODE_ATTRIBUTION_TIMEOUT,
        attribute_decode_token_time(FileCacheCondition::Warm),
    )
    .await
    .expect("the decode attribution probe must finish within 120 seconds");
}

#[tokio::test]
#[ignore = "serves a 10,000-token Shakespeare request with the model's files kept out of the file cache and attributes each decode token's time"]
#[serial]
async fn should_attribute_decode_token_time_with_a_cold_file_cache() {
    timeout(
        DECODE_ATTRIBUTION_TIMEOUT,
        attribute_decode_token_time(FileCacheCondition::Cold),
    )
    .await
    .expect("the cold decode attribution probe must finish within 120 seconds");
}

async fn attribute_decode_token_time(file_cache_condition: FileCacheCondition) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let test_run_configuration = TestRunContext::current().configuration;
    let model_directory = test_run_configuration
        .model_directory
        .expect("model_directory names the sparse model to attribute");
    let mlx_limit_bytes = usize::try_from(
        test_run_configuration
            .paging_experiment
            .maximum_mlx_memory_bytes()
            .expect("paging_experiment sets a valid MLX limit"),
    )
    .expect("the MLX limit should fit usize");
    // The configured chunk, not a fixed one: SSD-streaming prefill reads every
    // non-resident expert once per chunk, so the chunk size sets prefill's
    // storage traffic and must match what the run claims to measure.
    let prefill_chunk_tokens = test_run_configuration
        .paging_experiment
        .streaming_prompt_processing_chunk_tokens;
    eprintln!("[decode-attribution] prefill_chunk_tokens={prefill_chunk_tokens}");
    let probe_requests = probe_requests(&model_directory);
    let attribution_directory =
        tempfile::tempdir().expect("the probe should create an attribution directory");
    let attribution_log_path = attribution_directory.path().join("decode.jsonl");
    let mlx_memory_limits = MlxMemoryLimits::new(mlx_limit_bytes, mlx_limit_bytes)
        .expect("the configured limit should be a valid MLX limit");
    let (mut engine, end_of_sequence_token_ids) = create_attributed_engine(
        &model_directory,
        &attribution_log_path,
        &mlx_memory_limits,
        prefill_chunk_tokens,
    );
    load_engine_with_progress(&mut engine, "decode_attribution_model_load").await;
    // Started after loading: the resident bundle is copied into MLX memory at
    // load, so only reads made while serving should pay storage cost.
    let cold_file_cache = matches!(file_cache_condition, FileCacheCondition::Cold)
        .then(|| ColdFileCache::start(&model_directory));
    for probe_request in &probe_requests {
        let generated_token_ids = run_attributed_generation(
            &mut engine,
            probe_request.request_id,
            &probe_request.prompt_token_ids,
            probe_request.phase_name,
            probe_request.output_token_count,
            &end_of_sequence_token_ids,
        )
        .await;
        assert!(
            !generated_token_ids.is_empty(),
            "{} must generate tokens",
            probe_request.phase_name
        );
        // The residency each request leaves is what the next one starts
        // from; printing it shows whether decode's warm experts survive.
        let expert_statistics = engine
            .expert_weight_memory_cache_statistics_for_tests()
            .await
            .expect("the engine reports its expert residency");
        eprintln!(
            "[decode-attribution] after={} resident_experts={} resident_gb={:.2} warm_tables={} evictions={}",
            probe_request.phase_name,
            expert_statistics.entry_count,
            expert_statistics.resident_payload_byte_count as f64 / 1e9,
            expert_statistics.partial_layer_count,
            expert_statistics.eviction_count,
        );
    }
    let file_cache_sweep_count = cold_file_cache.map_or(0, ColdFileCache::finish);
    eprintln!(
        "[decode-attribution] file_cache={file_cache_condition:?} file_cache_sweeps={file_cache_sweep_count}"
    );
    let report_documents = read_attribution_report_documents(&attribution_log_path);
    let measured_report =
        generation_report_for_request(&report_documents, MEASURED_REQUEST_ID.value());
    print_decode_token_attribution(measured_report);
    print_prefill_operation_breakdown(measured_report);
    print_follow_up_operation_breakdown(generation_report_for_request(
        &report_documents,
        FOLLOW_UP_REQUEST_ID.value(),
    ));
}

/// One request the probe serves, in order.
struct ProbeRequest {
    request_id: RequestId,
    prompt_token_ids: Vec<u32>,
    output_token_count: u16,
    phase_name: &'static str,
}

/// The warmup, the measured long request and the decode-dominated follow-up,
/// each a Romeo and Juliet prompt of its own length.
fn probe_requests(model_directory: &Path) -> Vec<ProbeRequest> {
    let model_id = model_directory
        .file_name()
        .and_then(|name| name.to_str())
        .expect("the model directory has a UTF-8 name");
    [
        (
            WARMUP_REQUEST_ID,
            WARMUP_PROMPT_TOKEN_COUNT,
            WARMUP_OUTPUT_TOKEN_COUNT,
            "decode_attribution_warmup",
        ),
        (
            MEASURED_REQUEST_ID,
            MEASURED_PROMPT_TOKEN_COUNT,
            MEASURED_OUTPUT_TOKEN_COUNT,
            "decode_attribution_measured",
        ),
        (
            FOLLOW_UP_REQUEST_ID,
            FOLLOW_UP_PROMPT_TOKEN_COUNT,
            FOLLOW_UP_OUTPUT_TOKEN_COUNT,
            "decode_attribution_follow_up",
        ),
    ]
    .into_iter()
    .map(
        |(request_id, prompt_token_count, output_token_count, phase_name)| ProbeRequest {
            request_id,
            prompt_token_ids: crate::serving_acceptance::support::romeo_and_juliet::prepare_romeo_and_juliet_three_paragraph_summary_prompt(
                model_directory,
                model_id,
                request_id,
                prompt_token_count,
                output_token_count,
            ),
            output_token_count,
            phase_name,
        },
    )
    .collect()
}

/// The long request's prefill-side operations in total seconds, largest
/// first. Spans nest, so rows overlap; the outermost prefill span is the
/// prefill's wall time and the waits below it show where it goes.
fn print_prefill_operation_breakdown(measured_report: &Value) {
    let mut prefill_rows: Vec<(String, f64, u64)> = measured_report["operations"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|operation_report| {
            let operation_name = operation_report["operation"].as_str()?;
            let is_decode_operation = operation_name.starts_with("decode_")
                || operation_name == "generated_token_item_synchronization_wait";
            (!is_decode_operation).then(|| {
                (
                    operation_name.to_owned(),
                    operation_report["total_elapsed_nanoseconds"]
                        .as_u64()
                        .unwrap_or(0) as f64
                        / 1e9,
                    operation_report["occurrence_count"].as_u64().unwrap_or(0),
                )
            })
        })
        .filter(|(_, total_seconds, _)| *total_seconds >= MINIMUM_REPORTED_PREFILL_SECONDS)
        .collect();
    prefill_rows.sort_by(|left, right| right.1.total_cmp(&left.1));
    for (operation_name, total_seconds, occurrence_count) in prefill_rows {
        eprintln!(
            "[decode-attribution] measured_prefill operation={operation_name} seconds={total_seconds:.3} occurrences={occurrence_count}"
        );
    }
}

/// Every operation of the decode-dominated follow-up, per decode token and
/// largest first. Nested spans overlap, so the rows do not add up to the
/// wall time; the outermost span (`decode_advance_span`) is the token.
fn print_follow_up_operation_breakdown(follow_up_report: &Value) {
    let decode_token_count =
        operation_occurrences(follow_up_report, "decode_advance_span").max(1) as f64;
    let mut operation_rows: Vec<(String, f64, f64)> = follow_up_report["operations"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|operation_report| {
            (
                operation_report["operation"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
                operation_report["total_elapsed_nanoseconds"]
                    .as_u64()
                    .unwrap_or(0) as f64
                    / 1e6
                    / decode_token_count,
                operation_report["occurrence_count"].as_u64().unwrap_or(0) as f64
                    / decode_token_count,
            )
        })
        .filter(|(_, milliseconds_per_token, _)| {
            *milliseconds_per_token >= MINIMUM_REPORTED_MILLISECONDS_PER_TOKEN
        })
        .collect();
    operation_rows.sort_by(|left, right| right.1.total_cmp(&left.1));
    eprintln!(
        "[decode-attribution] follow_up decode_tokens={decode_token_count} prefill_ms={:.0}",
        operation_total_nanoseconds(follow_up_report, "prompt_prefill_advance_span") as f64 / 1e6
    );
    for (operation_name, milliseconds_per_token, occurrences_per_token) in operation_rows {
        eprintln!(
            "[decode-attribution] follow_up operation={operation_name} ms_per_token={milliseconds_per_token:.3} occurrences_per_token={occurrences_per_token:.2}"
        );
    }
    // Physical disk bytes against requested file bytes says whether the
    // decode's expert reads came from the SSD or from the OS page cache,
    // which a machine with RAM beyond the MLX limit fills with the model.
    eprintln!(
        "[decode-attribution] follow_up process_physical_disk_read_bytes_per_token={:.0}",
        follow_up_report["process_physical_disk_read_bytes"]
            .as_u64()
            .unwrap_or(0) as f64
            / decode_token_count
    );
    for counter_report in follow_up_report["counters"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let counter_name = counter_report["counter"].as_str().unwrap_or_default();
        eprintln!(
            "[decode-attribution] follow_up counter={counter_name} per_token={:.0}",
            counter_amount(follow_up_report, counter_name) as f64 / decode_token_count
        );
    }
}

/// Prints every decode operation and the residency counters per decode token.
fn print_decode_token_attribution(measured_report: &Value) {
    let decode_token_count = operation_occurrences(measured_report, "decode_advance_span").max(1);
    let per_token_milliseconds = |operation_name: &str| {
        operation_total_nanoseconds(measured_report, operation_name) as f64
            / 1e6
            / decode_token_count as f64
    };
    eprintln!(
        "[decode-attribution] decode_tokens={decode_token_count} decode_wall_ms_per_token={:.2} route_copy_ms_per_token={:.2} route_copies_per_token={:.1}",
        per_token_milliseconds("decode_advance_span"),
        per_token_milliseconds("decode_expert_route_host_materialization"),
        operation_occurrences(measured_report, "decode_expert_route_host_materialization") as f64
            / decode_token_count as f64,
    );
    eprintln!(
        "[decode-attribution] prefill_seconds={:.2} decode_seconds={:.2}",
        operation_total_nanoseconds(measured_report, "prompt_prefill_advance_span") as f64 / 1e9,
        operation_total_nanoseconds(measured_report, "decode_advance_span") as f64 / 1e9,
    );
    for operation_report in measured_report["operations"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let operation_name = operation_report["operation"].as_str().unwrap_or_default();
        if operation_name.starts_with("decode_") {
            eprintln!(
                "[decode-attribution] operation={operation_name} ms_per_token={:.3} occurrences_per_token={:.2}",
                per_token_milliseconds(operation_name),
                operation_occurrences(measured_report, operation_name) as f64
                    / decode_token_count as f64,
            );
        }
    }
    for counter_report in measured_report["counters"].as_array().into_iter().flatten() {
        let counter_name = counter_report["counter"].as_str().unwrap_or_default();
        if [
            "decode",
            "retained",
            "positional",
            "expert_source",
            "complete",
        ]
        .iter()
        .any(|marker| counter_name.contains(marker))
        {
            eprintln!(
                "[decode-attribution] counter={counter_name} amount={}",
                counter_amount(measured_report, counter_name)
            );
        }
    }
}

fn operation_report<'report>(
    report: &'report Value,
    operation_name: &str,
) -> Option<&'report Value> {
    report["operations"]
        .as_array()?
        .iter()
        .find(|operation_report| operation_report["operation"] == operation_name)
}

fn operation_total_nanoseconds(report: &Value, operation_name: &str) -> u64 {
    operation_report(report, operation_name)
        .and_then(|operation_report| operation_report["total_elapsed_nanoseconds"].as_u64())
        .unwrap_or(0)
}

fn operation_occurrences(report: &Value, operation_name: &str) -> u64 {
    operation_report(report, operation_name)
        .and_then(|operation_report| operation_report["occurrence_count"].as_u64())
        .unwrap_or(0)
}
