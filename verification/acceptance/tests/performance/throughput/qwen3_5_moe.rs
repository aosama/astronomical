//! Measures the serving throughput of the resident 35B sparse-MoE model over the
//! daemon IPC boundary: prompt-processing (prefill) tokens per second and
//! decode tokens per second, both server-attributed from the worker's generation
//! performance log (never client wall clock).
//!
//! The model is looked up by the stable `resident_sparse_moe` role from
//! registry/e2e_test_model_names.json instead of a hardcoded name, so this keeps
//! measuring the correct leaf when the retargeted artifact changes.
//!
//! The test case follows AGENTS.md "Instructions for Performance Throughput
//! Tests":
//! - Send at least 10,000 input tokens. The input is the continuation
//!   instruction plus the ~10,000-token Romeo and Juliet source fixture, which
//!   together tokenize to ~10,500 tokens.
//! - Acquire 500 output tokens (±10% is acceptable). Each measured
//!   completion is capped at 500 output tokens.
//! - The SSD (persistent prompt) cache is disabled in the worker's resolved
//!   configuration, so every completion prefills the full prompt from scratch.
//! - Both resident and streaming prefill chunk sizes are fixed at 2,048 tokens,
//!   and the configured MLX memory ceiling is capped at 36 GB.
//!
//! Method: a short warmup completion (~500 input tokens, ~50 output tokens)
//! is discarded so first-use JIT kernel compilation never inflates the measured
//! prefill, then the single measured completion over the full prompt reports the
//! server-attributed rates persisted to the durable historical record. The
//! journey verifies the measured input/output token counts and zero cache reuse.
//!
//! This is a laptop-only journey: it loads real weights into wired GPU memory,
//! so it is `#[ignore]`d and not wired into CI. Invoke it directly with
//! `cargo build --release --bin astronomical-inference-worker` followed by
//! one exact test command. The diagnostic paired run and clean paired run are
//! separate because attribution distorts measured throughput; only clean runs
//! append policy-labeled historical records:
//! - `cargo test --release -p astronomical-acceptance --features performance_throughput --test performance_throughput performance_throughput::qwen3_5_moe::should_measure_resident_sparse_moe_prompt_processing_and_decode_throughput -- --ignored --exact --test-threads=1 --nocapture`
//! - `cargo test --release -p astronomical-acceptance --features performance_throughput --test performance_throughput performance_throughput::qwen3_5_moe::should_report_streaming_sparse_moe_decode_route_materialization_evidence -- --ignored --exact --test-threads=1 --nocapture`
//! - `cargo test --release -p astronomical-acceptance --features performance_throughput --test performance_throughput performance_throughput::qwen3_5_moe::should_compare_streaming_sparse_moe_decode_cache_policies_with_attribution -- --ignored --exact --test-threads=1 --nocapture`
//! - `cargo test --release -p astronomical-acceptance --features performance_throughput --test performance_throughput performance_throughput::qwen3_5_moe::should_compare_clean_streaming_sparse_moe_decode_cache_policy_throughput -- --ignored --exact --test-threads=1 --nocapture`
//! Each paired command alternates policy order across three trials. The
//! release journey launches the release worker produced by the build step.

use crate::performance_throughput::decode_cache_experiment::{
    MeasuredShapeWindow, PAIRED_TRIAL_COUNT, run_paired_decode_cache_trials,
};
use astronomical_ipc_protocol::WorkerDecodeExpertCachePolicy;

use super::decode_cache_experiment::{
    report_attributed_policy_comparison, report_clean_policy_comparison,
    report_streaming_decode_stage_evidence,
};
use crate::performance_throughput::historical_record::ThroughputJourneyKind;
use crate::performance_throughput::support::{self as throughput_support, ThroughputJourney};
use crate::support;
use serial_test::serial;
use std::fs;

/// The short warmup: a ~500-token Romeo and Juliet opening, continued for a
/// short passage, that spins up first-use JIT kernels before the measured run.
const WARMUP_INPUT_INSTRUCTION: &str = "Continue the supplied Romeo and Juliet story above.";

/// The Romeo and Juliet source used for the bounded warmup prompt.
const WARMUP_ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_warmup_romeo_and_juliet.txt");

/// The warmup output cap.
const WARMUP_MAXIMUM_OUTPUT_TOKENS: u16 = 50;

/// The continuation instruction prepended to the full Romeo and Juliet source.
const MEASURED_INPUT_INSTRUCTION: &str =
    "Continue the supplied Romeo and Juliet story above for approximately 500 tokens.";

/// The ~10,000-token Romeo and Juliet source; with the instruction the full
/// input is ~10,500 tokens, clearing the >=10,000-token input requirement.
const MEASURED_ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_10000_tokens_romeo_and_juliet.txt");

/// The measured output cap: 500 output tokens (±10% acceptable).
const MEASURED_MAXIMUM_OUTPUT_TOKENS: u16 = 500;

/// The single-chunk lane's measured output cap: ~200 tokens keep the whole
/// journey (model load + prefill + decode) inside the 115-second budget while
/// still exercising probation, promotion, and eviction across decode steps.
const SINGLE_CHUNK_MEASURED_MAXIMUM_OUTPUT_TOKENS: u16 = 200;

/// The sampling temperature in thousandths (1_000 = 1.0, unclamped).
const TEMPERATURE_THOUSANDTHS: u16 = 1_000;

/// Measures the resident sparse-MoE model's serving prompt-processing and
/// decode throughput: after a short ~500-token warmup, a >=10,000-token Romeo
/// and Juliet input drives 500 output tokens of generation with the SSD
/// (persistent prompt) cache disabled, and the measured completion's
/// server-attributed rates are persisted to the durable historical record.
#[test]
#[ignore = "loads the resident 35B sparse-MoE model and measures serving prompt-processing and decode throughput over IPC"]
#[serial]
fn should_measure_resident_sparse_moe_prompt_processing_and_decode_throughput() {
    let journey = sparse_moe_text_journey(false);
    let measured_throughput = throughput_support::run_journey_with_timeout(
        support::resident_sparse_moe_model_id(),
        journey,
    );
    throughput_support::assert_measurement_shape(&measured_throughput);
    // The seating-regression guard: a multi-chunk prompt that stops seating
    // complete layers shows up as re-streamed expert bytes and collapsing
    // prompt-processing rate long before anything else fails. Measured
    // 2,509 tokens/second on this lane (2026-10-24); the floor leaves ~20%
    // for run-to-run noise while still failing a utilization collapse
    // (the flight ran at roughly half of this with 19.6 GB unused).
    assert!(
        measured_throughput.prefill_tokens_per_second >= 2_000.0,
        "the resident prefill rate collapsed to {:.0} tokens/second (floor 2,000): \
         expert seating or streaming regressed",
        measured_throughput.prefill_tokens_per_second
    );
}

/// The streaming engine's >=10,000-token lane with the same floor guard: the
/// streaming path seats complete layers across chunks too, and its
/// prompt-processing rate is the utilization signal for that family.
#[test]
#[ignore = "loads the streaming 35B sparse-MoE model and measures prompt-processing throughput"]
#[serial]
fn should_measure_streaming_sparse_moe_prompt_processing_throughput() {
    let mut journey = sparse_moe_text_journey(false);
    journey.decode_expert_cache_policy = WorkerDecodeExpertCachePolicy::Baseline;
    let measured_throughput = throughput_support::run_journey_with_timeout(
        support::large_sparse_moe_streaming_model_id(),
        journey,
    );
    throughput_support::assert_measurement_shape(&measured_throughput);
    // Calibrated 2026-10-24 on a cold streaming worker: 607 tokens/second.
    // The cold worker seats nothing before chunk 1, so this number carries
    // the full first-turn stream; a seating-utilization fix should raise the
    // measured rate and with it this floor. The floor's job is to fail a
    // collapse (re-stream growth, admission thrash) loudly, at 20% margin.
    assert!(
        measured_throughput.prefill_tokens_per_second >= 500.0,
        "the streaming prefill rate collapsed to {:.0} tokens/second (floor 500): \
         expert seating or streaming regressed",
        measured_throughput.prefill_tokens_per_second
    );
}

/// Captures production streaming decode-stage timings on the budget-fitting
/// single-chunk Romeo and Juliet shape. This attribution-only run does not
/// append rates to the clean historical series or serve as a throughput benchmark.
#[test]
#[ignore = "loads the streaming 35B sparse-MoE model for single-chunk decode-stage attribution"]
#[serial]
fn should_report_streaming_sparse_moe_decode_route_materialization_evidence() {
    let mut journey = streaming_single_chunk_journey(true, 0);
    journey.decode_expert_cache_policy = WorkerDecodeExpertCachePolicy::Baseline;
    let measured_throughput = throughput_support::run_journey_with_timeout(
        support::large_sparse_moe_streaming_model_id(),
        journey,
    );
    assert!(
        (400..=700).contains(&measured_throughput.total_input_tokens),
        "the measured single-chunk prompt should stay near 557 tokens, got {}",
        measured_throughput.total_input_tokens,
    );
    assert_eq!(measured_throughput.cached_tokens, 0);
    assert!(
        (180..=220).contains(&measured_throughput.total_output_tokens),
        "the measured completion should generate about 200 tokens, got {}",
        measured_throughput.total_output_tokens,
    );
    let diagnostic_evidence = measured_throughput
        .decode_cache_diagnostic_evidence
        .as_ref()
        .expect("the attributed streaming run should produce stage evidence");
    assert!(
        diagnostic_evidence
            .decode_token_forward_latency_p95_nanoseconds
            .is_some(),
        "the streaming run should record decode-forward latency"
    );
    assert!(
        diagnostic_evidence
            .decode_advance_span
            .as_ref()
            .is_some_and(|decode_advance| decode_advance.occurrence_count > 0),
        "the streaming run should record complete decode-advance spans"
    );
    assert!(
        diagnostic_evidence
            .decode_expert_route_host_materialization
            .as_ref()
            .is_some_and(|route_materialization| route_materialization.occurrence_count > 0),
        "the streaming run should record per-layer route host materialization"
    );
    assert!(
        diagnostic_evidence
            .decode_expert_route_contiguous_array_construction
            .as_ref()
            .is_some_and(|array_construction| array_construction.occurrence_count > 0),
        "the streaming run should record route array construction"
    );
    assert!(
        diagnostic_evidence
            .decode_expert_route_array_evaluation
            .as_ref()
            .is_some_and(|array_evaluation| array_evaluation.occurrence_count > 0),
        "the streaming run should record route array evaluation"
    );
    assert!(
        diagnostic_evidence
            .decode_expert_route_host_copy
            .as_ref()
            .is_some_and(|host_copy| host_copy.occurrence_count > 0),
        "the streaming run should record route host copies"
    );
    report_streaming_decode_stage_evidence(&measured_throughput, diagnostic_evidence);
}

/// Runs three diagnostic pairs with alternating order, preserving output
/// parity and attribution evidence without adding these distorted rates to the
/// clean historical throughput series.
///
/// The full-shape streaming journey (10,544-token prompt) cannot fit the
/// 115-second journey budget under the configured 36,000,000,000-byte journey
/// cap: its 2,048-token chunk is not admissible and, now that capacity-recovery halving is removed, the
/// request fails fast by design. This lane therefore pairs policies on the
/// single-chunk warmup shape (the ~557-token Romeo and Juliet opening), which
/// still exercises decode admission, probation, promotion, and eviction while
/// prefill stays identical across arms.
#[test]
#[ignore = "loads the streaming 35B sparse-MoE model for single-chunk paired decode-cache attribution"]
#[serial]
fn should_compare_streaming_sparse_moe_decode_cache_policies_with_attribution() {
    let journey = streaming_single_chunk_journey(true, 0);
    let paired_trials = run_paired_decode_cache_trials(
        support::large_sparse_moe_streaming_model_id(),
        &journey,
        PAIRED_TRIAL_COUNT,
        &MeasuredShapeWindow {
            minimum_input_tokens: 400,
            maximum_input_tokens: 700,
            minimum_output_tokens: 180,
            maximum_output_tokens: 220,
        },
    );
    report_attributed_policy_comparison(&paired_trials);
}

/// Runs clean paired trials so both policies receive policy-labeled durable
/// throughput rows without enabling request attribution.
#[test]
#[ignore = "loads the streaming 35B sparse-MoE model for single-chunk paired clean throughput"]
#[serial]
fn should_compare_clean_streaming_sparse_moe_decode_cache_policy_throughput() {
    assert!(
        !throughput_support::diagnostics_enabled(),
        "clean paired throughput must run without the diagnostics environment switch"
    );
    let journey = streaming_single_chunk_journey(false, 0);
    let paired_trials = run_paired_decode_cache_trials(
        support::large_sparse_moe_streaming_model_id(),
        &journey,
        PAIRED_TRIAL_COUNT,
        &MeasuredShapeWindow {
            minimum_input_tokens: 400,
            maximum_input_tokens: 700,
            minimum_output_tokens: 180,
            maximum_output_tokens: 220,
        },
    );
    report_clean_policy_comparison(&paired_trials);
}

fn sparse_moe_text_journey(attribution_enabled: bool) -> ThroughputJourney {
    let mut journey = ThroughputJourney::production_default(
        ThroughputJourneyKind::Text,
        throughput_support::romeo_and_juliet_warmup_prompt(
            WARMUP_INPUT_INSTRUCTION,
            WARMUP_ROMEO_AND_JULIET_SOURCE,
        ),
        Vec::new(),
        WARMUP_MAXIMUM_OUTPUT_TOKENS,
        format!("{MEASURED_INPUT_INSTRUCTION}\n\n{MEASURED_ROMEO_AND_JULIET_SOURCE}"),
        Vec::new(),
        MEASURED_MAXIMUM_OUTPUT_TOKENS,
        TEMPERATURE_THOUSANDTHS,
    );
    journey.maximum_mlx_memory_bytes = Some(throughput_support::MAXIMUM_MODEL_MEMORY_BYTES);
    journey.attribution_enabled = attribution_enabled;
    journey
}

/// The budget-fitting streaming lane: one folded single-chunk prompt (the
/// warmup Romeo and Juliet opening) driving ~200 decode tokens in a fresh
/// worker per run, so the whole journey fits the 115-second cap while decode
/// caching is still exercised. Two sequential completions in one streaming
/// worker cannot both fit under the configured 36,000,000,000-byte journey cap
/// once the first request leaves its
/// warm expert tables resident, so this lane skips the in-worker warmup.
/// The budget-fitting streaming lane's journey builder. A zero warmup output
/// marks the single-request lane (fresh worker per run); a nonzero warmup runs
/// the in-worker warmup completion before the measured one.
fn streaming_single_chunk_journey(
    attribution_enabled: bool,
    warmup_output_tokens: u16,
) -> ThroughputJourney {
    let single_chunk_prompt = throughput_support::romeo_and_juliet_warmup_prompt(
        WARMUP_INPUT_INSTRUCTION,
        WARMUP_ROMEO_AND_JULIET_SOURCE,
    );
    let mut journey = ThroughputJourney::production_default(
        ThroughputJourneyKind::Text,
        single_chunk_prompt.clone(),
        Vec::new(),
        warmup_output_tokens,
        single_chunk_prompt,
        Vec::new(),
        SINGLE_CHUNK_MEASURED_MAXIMUM_OUTPUT_TOKENS,
        TEMPERATURE_THOUSANDTHS,
    );
    journey.maximum_mlx_memory_bytes = Some(throughput_support::MAXIMUM_MODEL_MEMORY_BYTES);
    journey.attribution_enabled = attribution_enabled;
    journey
}

/// Proves one streaming worker serves back-to-back completions: the measured
/// request's prefill admission must reserve the concurrent union stream of
/// every non-resident sparse layer and evict elastic warm tables up front,
/// instead of admitting on a single-page reserve and breaching the ceiling
/// mid-forward (the terminal failure this journey reproduced before the fix).
#[test]
#[ignore = "loads the streaming 35B sparse-MoE model and serves two sequential completions in one worker"]
#[serial]
fn should_serve_two_sequential_single_chunk_streaming_completions_in_one_worker() {
    let journey = streaming_single_chunk_journey(true, WARMUP_MAXIMUM_OUTPUT_TOKENS);
    let measured_throughput = throughput_support::run_journey_with_timeout(
        support::large_sparse_moe_streaming_model_id(),
        journey,
    );
    assert!(
        (400..=700).contains(&measured_throughput.total_input_tokens),
        "the measured single-chunk prompt should stay near 557 tokens, got {}",
        measured_throughput.total_input_tokens,
    );
    assert_eq!(measured_throughput.cached_tokens, 0);
    assert!(
        (180..=220).contains(&measured_throughput.total_output_tokens),
        "the measured completion should generate about 200 tokens, got {}",
        measured_throughput.total_output_tokens,
    );
    let diagnostic_evidence = measured_throughput
        .decode_cache_diagnostic_evidence
        .expect("the attributed journey should retain decode-cache evidence");
    assert!(
        diagnostic_evidence
            .decode_token_forward_latency_p95_nanoseconds
            .is_some(),
        "the measured generation report must carry decode-forward p95"
    );
}

/// Tiles the 10,000-token fixture into a >=30,000-token conversation: the
/// multi-chunk shape of the 2026-10-24 under-utilization flight (seventeen
/// 2,048-token chunks) in a journey-sized form.
fn multi_chunk_measured_prompt() -> String {
    format!(
        "{MEASURED_INPUT_INSTRUCTION}\n\n{source}{source}{source}",
        source = MEASURED_ROMEO_AND_JULIET_SOURCE
    )
}

/// The evidence both turns of a two-turn journey contribute.
struct TwoTurnEvidence {
    first: throughput_support::ThroughputMeasurement,
    second: throughput_support::ThroughputMeasurement,
    first_positional_read_bytes: Option<u64>,
    second_positional_read_bytes: Option<u64>,
}

/// Drives one worker through the same long prompt twice. The first turn
/// streams and (per the residency plan) seats complete layers; the second
/// turn must reuse that seating instead of re-paying the SSD streams.
async fn run_two_turn_streaming_journey(
    attribution_enabled: bool,
    persistent_prompt_cache_enabled: bool,
) -> TwoTurnEvidence {
    use astronomical_daemon::{ChatGenerationExecutor, GenerationPerformanceLog, WorkerHandle};
    use astronomical_ipc_protocol::RequestId;
    use tokio::time::timeout;

    let model_id = support::large_sparse_moe_streaming_model_id();
    let model_directory = crate::support::configured_installed_model_directory_by_id(model_id);
    let long_prompt = multi_chunk_measured_prompt();
    let (
        _isolated_home,
        worker_executable_path,
        logging_directory,
        model_policy_catalog,
        mut worker_startup_configuration,
    ) = throughput_support::perf_worker_environment(
        model_id,
        &model_directory,
        attribution_enabled,
    );
    worker_startup_configuration.configured_maximum_mlx_memory_bytes =
        Some(throughput_support::MAXIMUM_MODEL_MEMORY_BYTES);
    worker_startup_configuration.persistent_prompt_cache_enabled = persistent_prompt_cache_enabled;
    let performance_log_path = logging_directory.join("performance.jsonl");
    fs::create_dir_all(&logging_directory).expect("the worker logging directory should be created");

    let worker_handle = WorkerHandle::launch_with_startup_configuration(
        worker_executable_path,
        throughput_support::MODEL_LOAD_TIMEOUT,
        GenerationPerformanceLog::open(&logging_directory)
            .expect("the two-turn performance log should open"),
        model_policy_catalog,
        worker_startup_configuration,
    )
    .await
    .expect("the two-turn worker should launch");
    throughput_support::wait_until_idle(&worker_handle, throughput_support::READY_ATTEMPT_LIMIT)
        .await;

    let _first_completion_text = timeout(
        throughput_support::JOURNEY_TIMEOUT,
        throughput_support::drive_completion(
            &worker_handle,
            RequestId::new(1),
            model_id,
            &long_prompt,
            Vec::new(),
            MEASURED_MAXIMUM_OUTPUT_TOKENS,
            TEMPERATURE_THOUSANDTHS,
        ),
    )
    .await
    .expect("the first long turn should finish inside the journey budget")
    .expect("the first long turn should complete");
    let _second_completion_text = timeout(
        throughput_support::JOURNEY_TIMEOUT,
        throughput_support::drive_completion(
            &worker_handle,
            RequestId::new(throughput_support::MEASURED_REQUEST_ID),
            model_id,
            &long_prompt,
            Vec::new(),
            MEASURED_MAXIMUM_OUTPUT_TOKENS,
            TEMPERATURE_THOUSANDTHS,
        ),
    )
    .await
    .expect("the second long turn should finish inside the journey budget")
    .expect("the second long turn should complete");

    let final_health_snapshot = worker_handle.worker_health_snapshot();
    let resident_expert_payload_bytes = final_health_snapshot
        .expert_residency
        .as_ref()
        .map(|residency| u64::from(residency.resident_expert_payload_bytes))
        .unwrap_or(0);
    eprintln!(
        "[multi-chunk-utilization] resident_expert_payload_bytes_after_second_turn={resident_expert_payload_bytes}"
    );
    worker_handle
        .shutdown()
        .await
        .expect("the two-turn worker should shut down");

    let performance_log_path_ref = performance_log_path.as_path();
    let first = throughput_support::read_completion_by_index(
        performance_log_path_ref,
        model_id,
        WorkerDecodeExpertCachePolicy::Baseline,
        throughput_support::CompletionIndex::First,
    )
    .expect("the first turn should write a performance sample");
    let second = throughput_support::read_completion_by_index(
        performance_log_path_ref,
        model_id,
        WorkerDecodeExpertCachePolicy::Baseline,
        throughput_support::CompletionIndex::Last,
    )
    .expect("the second turn should write a performance sample");

    let first_positional_read_bytes =
        crate::performance_throughput::completion::read_decode_cache_diagnostic_evidence(
            &logging_directory,
            1,
        )
        .map(|evidence| evidence.positional_file_read_byte_count);
    let second_positional_read_bytes =
        crate::performance_throughput::completion::read_decode_cache_diagnostic_evidence(
            &logging_directory,
            throughput_support::MEASURED_REQUEST_ID,
        )
        .map(|evidence| evidence.positional_file_read_byte_count);

    TwoTurnEvidence {
        first,
        second,
        first_positional_read_bytes,
        second_positional_read_bytes,
    }
}

/// The screenshot journey: a >=30,000-token streaming prompt across many
/// 2,048-token chunks. The residency plan must progressively seat complete
/// layers into the paid leftover budget, so the second identical turn
/// re-streams strictly fewer expert bytes than the first turn did.
#[test]
#[ignore = "loads the streaming 35B sparse-MoE model twice on a >=30,000-token prompt"]
#[serial]
fn should_reuse_seated_layers_on_a_repeat_multi_chunk_streaming_turn() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("the two-turn runtime should build");
    let two_turn_evidence = runtime.block_on(run_two_turn_streaming_journey(true, false));

    for turn_measurement in [&two_turn_evidence.first, &two_turn_evidence.second] {
        assert!(
            (30_000..=35_000).contains(&turn_measurement.total_input_tokens),
            "each turn should carry the >=30,000-token prompt, got {}",
            turn_measurement.total_input_tokens
        );
        assert_eq!(
            turn_measurement.cached_tokens, 0,
            "the prompt cache is off in this journey"
        );
    }
    let first_positional_read_bytes = two_turn_evidence
        .first_positional_read_bytes
        .expect("the first turn should write an attribution report");
    let second_positional_read_bytes = two_turn_evidence
        .second_positional_read_bytes
        .expect("the second turn should write an attribution report");
    eprintln!(
        "[multi-chunk-utilization] first_turn_positional_bytes={first_positional_read_bytes} second_turn_positional_bytes={second_positional_read_bytes}"
    );
    assert!(
        second_positional_read_bytes < first_positional_read_bytes,
        "the repeat turn must stream fewer expert bytes than the first turn: \
         {second_positional_read_bytes} >= {first_positional_read_bytes} means seating did \
         not carry over across chunks"
    );
}

/// The repeat turn of a persistent-cache-enabled worker must restore the
/// published prompt prefix instead of re-prefilling it: cached_tokens > 0.
#[test]
#[ignore = "loads the streaming 35B sparse-MoE model twice with the persistent prompt cache enabled"]
#[serial]
fn should_restore_the_prompt_prefix_on_a_repeat_multi_chunk_streaming_turn() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("the prefix-reuse runtime should build");
    let two_turn_evidence = runtime.block_on(run_two_turn_streaming_journey(false, true));

    assert_eq!(
        two_turn_evidence.first.cached_tokens, 0,
        "the cold turn must prefill the whole prompt"
    );
    assert!(
        two_turn_evidence.second.cached_tokens > 0,
        "the repeat turn must restore a published prefix, got zero cached tokens \
         for a {}-token repeated prompt",
        two_turn_evidence.second.total_input_tokens
    );
}
