//! User journey proving SSD-paged expert reuse across decode tokens under a constrained 23 GB ceiling.
//!
//! The model cannot keep every expert resident in this acceptance cell. The
//! desired behavior is therefore not merely "request succeeds": routed experts
//! must remain reusable across decoder layers instead of consuming nearly the
//! whole ceiling on early layers while repeatedly reading omitted routes.
//!
//! Acceptance criteria (what "good" looks like for the user):
//!
//! 1. The model produces a complete response (non-empty text, finish reason
//!    "stop" or "length").
//! 2. The model is in paging mode: expert payload is non-zero during decode,
//!    confirming the test setup exercises SSD streaming.
//! 3. Memory stays within the configured ceiling (plus a small tolerance).
//! 4. Decode keeps expert ownership in leftover RAM: either a complete-layer
//!    foundation, or retained routed pages that attribution shows being reused.
//! 5. Decode throughput is reported as positive finite evidence without imposing
//!    one laptop's hardware-specific performance threshold.
//! 6. Exactly one generation attribution report is written, proving clean request
//!    completion.

use std::path::Path;

use super::ssd_paging_decode_expert_reuse_rest_journey::support::{
    DecodeExpertReuseCounters, assert_decode_expert_reuse_attribution,
    decode_streamed_layer_indices, generation_attribution_report_count,
    generation_expert_source_read_bytes, log_status_progress, preserve_memory_utilization_evidence,
    record_expert_payload_increase,
};

mod support;

use serde_json::{Value, json};
use serial_test::serial;
use tokio::time::{Duration, Instant, sleep, timeout};

use super::thinking_budget_completion_stream::{
    CompletedStream, consume_completed_stream, thinking_budget_completion_request,
    write_worker_configuration,
};
use crate::support::openai_client::{ChatCompletionStream, LocalOpenAiClient};
use crate::support::serving_rest::{
    SSD_JOURNEY_TIMEOUT, get_json_endpoint, launch_real_model_rest_server,
    stop_real_model_rest_server,
};

fn model_id() -> &'static str {
    crate::support::large_sparse_moe_streaming_model_id()
}
// This ceiling defines a reproducible acceptance cell only. Production code must
// not hardwire it or assume this model always leaves exactly four layers cold.
const MAXIMUM_MLX_MEMORY_BYTES: u64 = 23_000_000_000;
// One percent of the ceiling covers allocator rounding and transient peaks that
// settle before the finalized snapshot.
const MAXIMUM_MLX_MEMORY_TOLERANCE_BYTES: u64 = MAXIMUM_MLX_MEMORY_BYTES / 100;
const PROMPT_TOKEN_COUNT: usize = 7_000;
const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 10_000;
const THINKING_BUDGET_TOKEN_COUNT: u32 = 1_000;
const STATUS_LOG_INTERVAL: Duration = Duration::from_secs(1);
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST server and real worker against the large sparse MoE streaming model under a 23 GB ceiling to accept routed-expert reuse from retained RAM during decode"]
#[serial]
async fn should_reuse_retained_decode_experts_while_staying_within_the_mlx_memory_ceiling() {
    timeout(
        SSD_JOURNEY_TIMEOUT,
        run_ssd_paging_decode_expert_reuse_journey(),
    )
    .await
    .expect("the progressive expert-memory SSD journey must finish within 120 seconds");
}

async fn run_ssd_paging_decode_expert_reuse_journey() {
    let model_directory = crate::support::configured_installed_model_directory_by_id(model_id());
    let isolated_worker_home =
        tempfile::tempdir().expect("the memory-management worker home should be created");
    write_acceptance_config(isolated_worker_home.path(), &model_directory);
    let user_message = build_decode_reuse_prompt(&model_directory);
    let real_model_rest_server = launch_real_model_rest_server(
        model_id(),
        model_directory,
        isolated_worker_home.path(),
        MAXIMUM_MLX_MEMORY_BYTES,
    )
    .await;
    let server_address = real_model_rest_server.server_address;
    let (completed_stream, memory_evidence) =
        stream_reply_while_observing_status(server_address, &user_message).await;

    // --- Structural assertion 1: generation completes with real output ---
    completed_stream.assert_produced_output_and_finished();

    let final_memory = assert_paging_mode_and_memory_ceiling(&memory_evidence.final_status);

    // --- Structural assertion 4: leftover RAM keeps expert ownership ---
    // Seating complete layers is the preferred reuse. Routed-page hits remain
    // valid when leftover cannot hold a complete-layer foundation.
    stop_real_model_rest_server(real_model_rest_server).await;
    let attribution_home = isolated_worker_home.path();
    let reuse_counters =
        assert_decode_expert_reuse_attribution(attribution_home, &memory_evidence.final_status);
    assert!(
        final_memory.final_expert_payload_bytes >= 1_000_000_000
            || reuse_counters.retained_route_assignment_hit_count > 0,
        "decode must keep leftover complete layers or reuse retained routed pages; expert_payload_bytes={} retained_route_assignment_hit_count={}",
        final_memory.final_expert_payload_bytes,
        reuse_counters.retained_route_assignment_hit_count,
    );

    // --- Measured assertion 5: throughput remains portable evidence ---
    let average_throughput = assert_decode_throughput_is_positive(&memory_evidence.final_status);

    // --- Measured assertion 7: the status document carries the same split the
    // attribution counters do (issue #510 observability parity) ---
    crate::support::memory_utilization_parity::assert_status_memory_ceiling_utilization_closes(
        &memory_evidence.final_status,
    );

    let reply_phase_gain_bytes = assert_reply_phase_kept_more_experts(&memory_evidence);

    // --- Structural assertion 7: exactly one generation attribution report ---
    assert_eq!(
        generation_attribution_report_count(isolated_worker_home.path()),
        1,
        "exactly one generation attribution report should be written"
    );

    print_success_evidence(
        isolated_worker_home.path(),
        &DecodeReuseSuccessReport {
            reuse_counters: &reuse_counters,
            final_memory: &final_memory,
            average_throughput: &average_throughput,
            reply_phase_gain_bytes,
            prompt_window_payload_increments: memory_evidence.retained_expert_payload_bytes.len(),
            output_character_count: completed_stream.model_text.len(),
        },
    );
    preserve_memory_utilization_evidence(
        isolated_worker_home.path(),
        &memory_evidence.final_status,
        &reuse_counters.memory_ceiling_utilization_evidence_counters(),
    );
}

async fn stream_reply_while_observing_status(
    server_address: std::net::SocketAddr,
    user_message: &str,
) -> (CompletedStream, ProgressiveExpertMemoryEvidence) {
    let openai_client = LocalOpenAiClient::new(server_address, "local-acceptance-client");
    let completion_request = thinking_budget_completion_request(
        model_id(),
        user_message,
        MAXIMUM_OUTPUT_TOKEN_COUNT,
        THINKING_BUDGET_TOKEN_COUNT,
    );
    let streamed_completion: ChatCompletionStream = openai_client
        .create_streaming_chat_completion(&completion_request)
        .await
        .expect("the public REST summary request should start");
    tokio::join!(
        consume_completed_stream(streamed_completion),
        observe_ssd_paging_decode_expert_reuse(server_address),
    )
}

fn build_decode_reuse_prompt(model_directory: &Path) -> String {
    let repeated_source = ROMEO_AND_JULIET_SOURCE.repeat(3);
    crate::support::exact_model_prompt::build_exact_model_prompt_content(
        model_directory,
        &repeated_source,
        "Summarize Romeo and Juliet in one concise paragraph. Include the central conflict, major decisions, and tragic outcome.",
        PROMPT_TOKEN_COUNT,
    )
}

/// What the finalized status reported about memory once the reply finished.
struct FinalMemoryObservation {
    final_expert_payload_bytes: u64,
    final_active_memory_bytes: u64,
    peak_memory_bytes: u64,
}

fn assert_paging_mode_and_memory_ceiling(final_status: &Value) -> FinalMemoryObservation {
    // --- Structural assertion 2: the model is in paging mode (some experts
    // retained, not all-zero, confirming the test exercises SSD streaming) ---
    let final_expert_payload_bytes = final_status["mlx_memory_snapshot"]["expert_payload_bytes"]
        .as_u64()
        .unwrap_or(0);
    assert!(
        final_expert_payload_bytes > 0,
        "the paged model should retain some expert payload in memory"
    );

    // --- Structural assertion 3: memory stays within the configured ceiling ---
    let final_active_memory_bytes = final_status["mlx_memory_snapshot"]["active_memory_bytes"]
        .as_u64()
        .expect("the completed status should report final active MLX memory");
    let peak_memory_bytes = final_status["mlx_memory_snapshot"]["peak_memory_bytes"]
        .as_u64()
        .expect("the completed status should report peak MLX memory");
    assert!(
        final_active_memory_bytes <= MAXIMUM_MLX_MEMORY_BYTES,
        "final active memory {final_active_memory_bytes} must stay within ceiling {MAXIMUM_MLX_MEMORY_BYTES}"
    );
    assert!(
        peak_memory_bytes
            <= MAXIMUM_MLX_MEMORY_BYTES.saturating_add(MAXIMUM_MLX_MEMORY_TOLERANCE_BYTES),
        "peak memory {peak_memory_bytes} must stay within ceiling plus tolerance {}",
        MAXIMUM_MLX_MEMORY_BYTES.saturating_add(MAXIMUM_MLX_MEMORY_TOLERANCE_BYTES)
    );
    FinalMemoryObservation {
        final_expert_payload_bytes,
        final_active_memory_bytes,
        peak_memory_bytes,
    }
}

struct AverageThroughput {
    generation_tokens_per_second: f64,
    prefill_tokens_per_second: f64,
}

fn assert_decode_throughput_is_positive(final_status: &Value) -> AverageThroughput {
    let average_generation_tokens_per_second =
        final_status["serving_session"]["average_generation_tok_per_second"]
            .as_f64()
            .expect("the completed status should report average generation throughput");
    let average_prefill_tokens_per_second =
        final_status["serving_session"]["average_prefill_tok_per_second"]
            .as_f64()
            .expect("the completed status should report average prefill throughput");
    assert!(
        average_generation_tokens_per_second.is_finite()
            && average_generation_tokens_per_second > 0.0,
        "decode throughput must be a positive finite measurement"
    );
    AverageThroughput {
        generation_tokens_per_second: average_generation_tokens_per_second,
        prefill_tokens_per_second: average_prefill_tokens_per_second,
    }
}

/// Returns how many bytes of experts the reply phase kept beyond the prompt phase.
fn assert_reply_phase_kept_more_experts(memory_evidence: &ProgressiveExpertMemoryEvidence) -> u64 {
    // --- Measured assertion 8: the reply phase keeps more experts than the
    // prompt phase could (issue #512). Reading a prompt needs one layer's
    // MLX working buffers for a whole chunk and writing a reply needs them for one
    // token, so the handoff has room to seat more. The handoff used to seat
    // against a conservative no-evidence fallback and grow as decode
    // forwards were measured, and this assertion watched that growth. Since
    // every prompt layer finishes before the next begins (issue #2221) the
    // fallback is small, the handoff seats its whole budget at once, and
    // nothing is left to grow during generation. The step up from the
    // prompt phase to the reply phase is what shows the reply budget is
    // used. The sampler records increases only, so each window's last
    // sample is its peak. ---
    let peak_prompt_expert_payload_bytes = memory_evidence
        .retained_expert_payload_bytes
        .last()
        .copied()
        .unwrap_or(0);
    let peak_generation_expert_payload_bytes = memory_evidence
        .generation_expert_payload_bytes
        .last()
        .copied()
        .expect("generation must observe the expert payload at least once");
    let reply_phase_gain_bytes =
        peak_generation_expert_payload_bytes.saturating_sub(peak_prompt_expert_payload_bytes);
    assert!(
        reply_phase_gain_bytes > 500_000_000,
        "the reply phase must keep more experts than the prompt phase; prompt_peak={peak_prompt_expert_payload_bytes} generation_peak={peak_generation_expert_payload_bytes} gain={reply_phase_gain_bytes}"
    );
    reply_phase_gain_bytes
}

/// Everything the success line reports that the assertions above produced.
struct DecodeReuseSuccessReport<'report> {
    reuse_counters: &'report DecodeExpertReuseCounters,
    final_memory: &'report FinalMemoryObservation,
    average_throughput: &'report AverageThroughput,
    reply_phase_gain_bytes: u64,
    prompt_window_payload_increments: usize,
    output_character_count: usize,
}

fn print_success_evidence(
    isolated_worker_home: &Path,
    success_report: &DecodeReuseSuccessReport<'_>,
) {
    let reuse_counters = success_report.reuse_counters;
    let final_memory = success_report.final_memory;
    // Diagnostic output: useful for debugging but not asserted. The prompt
    // window records each increase of the experts kept while the prompt is
    // read.
    let expert_source_read_bytes = generation_expert_source_read_bytes(isolated_worker_home);
    let decode_streamed_layer_indices = decode_streamed_layer_indices(isolated_worker_home);
    eprintln!(
        "[ssd-paging-decode-expert-reuse] status=success prompt_tokens={PROMPT_TOKEN_COUNT} \
         maximum_mlx_memory_gb={} final_expert_payload_gb={:.2} final_active_memory_gb={:.2} \
         peak_memory_gb={:.2} expert_source_read_gb={:.2} \
         retained_route_hits={} hot_expert_partial_route_hits={} \
         hot_expert_warm_insert_count={} \
         hot_expert_route_fully_covered={} hot_expert_route_partially_covered={} \
         hot_expert_route_fully_missed={} hot_expert_route_retained_assignments={} \
         hot_expert_route_missing_assignments={} hot_expert_mixed_routes={} \
         route_observation_stored={} route_observation_captured_layers={} \
         route_observation_evicted={} previous_token_prefetch_issues={} \
         previous_token_prefetch_hits={} previous_token_prefetch_misses={} \
         previous_token_prefetch_bytes={} previous_token_prefetch_drops={} \
         mem_ceiling_gb={:.2} mem_active_gb={:.2} mem_unused_gb={:.2} \
         mem_reserved_core_slack_gb={:.2} mem_reserved_context_gb={:.2} \
         mem_reserved_workspace_gb={:.2} mem_unseated_entitlement_gb={:.2} \
         mem_unexplained_gb={:.2} mem_owner_overrun_gb={:.2} reply_phase_gain_gb={:.2} \
         decode_streamed_layer_count={} prompt_window_payload_increments={} \
         average_prefill_tok_per_second={:.2} average_generation_tok_per_second={:.2} \
         output_characters={}",
        MAXIMUM_MLX_MEMORY_BYTES / 1_000_000_000,
        final_memory.final_expert_payload_bytes as f64 / 1_073_741_824.0,
        final_memory.final_active_memory_bytes as f64 / 1_073_741_824.0,
        final_memory.peak_memory_bytes as f64 / 1_073_741_824.0,
        expert_source_read_bytes as f64 / 1_073_741_824.0,
        reuse_counters.retained_route_assignment_hit_count,
        reuse_counters.hot_expert_partial_route_hit_count,
        reuse_counters.hot_expert_warm_insert_count,
        reuse_counters.hot_expert_route_fully_covered_count,
        reuse_counters.hot_expert_route_partially_covered_count,
        reuse_counters.hot_expert_route_fully_missed_count,
        reuse_counters.hot_expert_route_retained_assignment_count,
        reuse_counters.hot_expert_route_missing_assignment_count,
        reuse_counters.hot_expert_mixed_route_count,
        reuse_counters.route_observation_stored_record_count,
        reuse_counters.route_observation_captured_layer_count,
        reuse_counters.route_observation_evicted_record_count,
        reuse_counters.previous_token_prefetch_issue_count,
        reuse_counters.previous_token_prefetch_hit_count,
        reuse_counters.previous_token_prefetch_miss_count,
        reuse_counters.previous_token_prefetch_byte_count,
        reuse_counters.previous_token_prefetch_capacity_drop_count,
        reuse_counters.memory_ceiling_utilization_ceiling_bytes as f64 / 1_073_741_824.0,
        reuse_counters.memory_ceiling_utilization_active_bytes as f64 / 1_073_741_824.0,
        reuse_counters.memory_ceiling_utilization_unused_headroom_bytes as f64 / 1_073_741_824.0,
        reuse_counters.memory_ceiling_utilization_reserved_model_core_slack_bytes as f64
            / 1_073_741_824.0,
        reuse_counters.memory_ceiling_utilization_reserved_context_growth_bytes as f64
            / 1_073_741_824.0,
        reuse_counters.memory_ceiling_utilization_reserved_activation_and_workspace_bytes as f64
            / 1_073_741_824.0,
        reuse_counters.memory_ceiling_utilization_unseated_expert_entitlement_bytes as f64
            / 1_073_741_824.0,
        reuse_counters.memory_ceiling_utilization_unexplained_headroom_bytes as f64
            / 1_073_741_824.0,
        reuse_counters.memory_ceiling_utilization_owner_overrun_bytes as f64 / 1_073_741_824.0,
        success_report.reply_phase_gain_bytes as f64 / 1_073_741_824.0,
        decode_streamed_layer_indices.len(),
        success_report.prompt_window_payload_increments,
        success_report.average_throughput.prefill_tokens_per_second,
        success_report
            .average_throughput
            .generation_tokens_per_second,
        success_report.output_character_count,
    );
}

/// What the status observer collected (issue #510): each increase of the expert
/// payload while the prompt was read and while the reply was written, and the
/// status read once the worker went idle after the request.
struct ProgressiveExpertMemoryEvidence {
    retained_expert_payload_bytes: Vec<u64>,
    generation_expert_payload_bytes: Vec<u64>,
    final_status: Value,
}

async fn observe_ssd_paging_decode_expert_reuse(
    server_address: std::net::SocketAddr,
) -> ProgressiveExpertMemoryEvidence {
    let deadline = Instant::now() + SSD_JOURNEY_TIMEOUT;
    let mut observed_prompt_processing = false;
    let mut retained_expert_payload_bytes = Vec::new();
    let mut generation_expert_payload_bytes = Vec::new();
    let mut last_status_log_at = Instant::now() - STATUS_LOG_INTERVAL;
    loop {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        if last_status_log_at.elapsed() >= STATUS_LOG_INTERVAL {
            log_status_progress(&status_document);
            last_status_log_at = Instant::now();
        }
        if status_document["activity"] == "prompt_processing" {
            observed_prompt_processing = true;
            record_expert_payload_increase(&status_document, &mut retained_expert_payload_bytes);
        }
        if observed_prompt_processing && status_document["activity"] == "generating" {
            record_expert_payload_increase(&status_document, &mut generation_expert_payload_bytes);
        }
        let snapshot_source = status_document["mlx_memory_snapshot"]["source"].as_str();
        if observed_prompt_processing
            && status_document["activity"] == "idle"
            && matches!(snapshot_source, Some("finalized" | "idle_poll"))
        {
            return ProgressiveExpertMemoryEvidence {
                retained_expert_payload_bytes,
                generation_expert_payload_bytes,
                final_status: status_document,
            };
        }
        assert!(Instant::now() < deadline);
        sleep(Duration::from_millis(100)).await;
    }
}

fn write_acceptance_config(isolated_worker_home: &Path, model_directory: &Path) {
    let configuration_document = json!({
        "model_directories": [model_directory],
        "maximum_mlx_memory_gb": 23,
        "max_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "persistent_prompt_cache_enabled": false,
        "performance_attribution_enabled": true,
        "logging": {
            "level": "debug",
            "retained_files": 2,
        },
        // This cell streams experts. Submit each completed decoder layer so
        // operation-local pages can detach instead of remaining live in a
        // multi-layer lazy tape until the terminal eval.
        "chunking": {
            "prompt_processing_chunk_size_tokens": 2_048,
            "fixed_ssd_streaming_prompt_processing_chunk_size_tokens": 2_048,
            "prefill_graph_submission_layer_interval": 0,
            "experimental_ssd_paging_generation_graph_submission_layer_interval": 1,
        },
    });
    write_worker_configuration(
        isolated_worker_home,
        "memory-management",
        &configuration_document,
    );
}
