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

use std::{fs, path::Path};

use super::ssd_paging_decode_expert_reuse_journey::support::{
    decode_streamed_layer_indices, generation_attribution_counter,
    generation_attribution_report_count, generation_expert_source_read_bytes, log_status_progress,
    preserve_memory_utilization_evidence, record_expert_payload_increase,
};

mod support;

use async_openai::{Client, config::OpenAIConfig, types::stream::StreamResponse};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::time::{Duration, Instant, sleep, timeout};

use crate::support::serving_rest::{
    JOURNEY_TIMEOUT, get_json_endpoint, launch_real_model_rest_server, stop_real_model_rest_server,
};

fn model_id() -> &'static str {
    crate::support::large_sparse_moe_model_id()
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
#[ignore = "launches the production REST server and real worker to accept expert-memory management behavior"]
async fn should_reuse_retained_decode_experts_while_staying_within_the_mlx_memory_ceiling() {
    timeout(
        JOURNEY_TIMEOUT,
        run_ssd_paging_decode_expert_reuse_journey(),
    )
    .await
    .expect("the progressive expert-memory REST journey must finish within 115 seconds");
}

async fn run_ssd_paging_decode_expert_reuse_journey() {
    let model_directory = crate::support::configured_installed_model_directory_by_id(model_id());
    let isolated_worker_home =
        tempfile::tempdir().expect("the memory-management worker home should be created");
    write_acceptance_config(isolated_worker_home.path(), &model_directory);
    let repeated_source = ROMEO_AND_JULIET_SOURCE.repeat(3);
    let user_message = crate::support::exact_model_prompt::build_exact_model_prompt_content(
        &model_directory,
        &repeated_source,
        "Summarize Romeo and Juliet in one concise paragraph. Include the central conflict, major decisions, and tragic outcome.",
        PROMPT_TOKEN_COUNT,
    );
    let real_model_rest_server = launch_real_model_rest_server(
        model_id(),
        model_directory,
        isolated_worker_home.path(),
        MAXIMUM_MLX_MEMORY_BYTES,
    )
    .await;
    let server_address = real_model_rest_server.server_address;
    let openai_client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(format!("http://{server_address}/v1"))
            .with_api_key("local-acceptance-client"),
    );
    let completion_request = json!({
        "model": model_id(),
        "messages": [{"role": "user", "content": user_message}],
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "thinking_budget": THINKING_BUDGET_TOKEN_COUNT,
    });
    let streamed_completion: StreamResponse<Value> = openai_client
        .chat()
        .create_stream_byot(completion_request)
        .await
        .expect("the public REST summary request should start");
    let (completed_stream, memory_evidence) = tokio::join!(
        consume_completed_stream(streamed_completion),
        observe_ssd_paging_decode_expert_reuse(server_address),
    );

    // --- Structural assertion 1: generation completes with real output ---
    assert!(!completed_stream.model_text.is_empty());
    assert!(matches!(
        completed_stream.finish_reason.as_deref(),
        Some("stop" | "length")
    ));

    // --- Structural assertion 2: the model is in paging mode (some experts
    // retained, not all-zero, confirming the test exercises SSD streaming) ---
    let final_expert_payload_bytes =
        memory_evidence.final_status["mlx_memory_snapshot"]["expert_payload_bytes"]
            .as_u64()
            .unwrap_or(0);
    assert!(
        final_expert_payload_bytes > 0,
        "the paged model should retain some expert payload in memory"
    );

    // --- Structural assertion 3: memory stays within the configured ceiling ---
    let final_active_memory_bytes =
        memory_evidence.final_status["mlx_memory_snapshot"]["active_memory_bytes"]
            .as_u64()
            .expect("the completed status should report final active MLX memory");
    let peak_memory_bytes =
        memory_evidence.final_status["mlx_memory_snapshot"]["peak_memory_bytes"]
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

    // --- Structural assertion 4: leftover RAM keeps expert ownership ---
    // Seating complete layers is the preferred reuse. Routed-page hits remain
    // valid when leftover cannot hold a complete-layer foundation.
    stop_real_model_rest_server(real_model_rest_server).await;
    let retained_route_assignment_hit_count = generation_attribution_counter(
        isolated_worker_home.path(),
        "retained_route_assignment_hit_count",
    );
    assert!(
        final_expert_payload_bytes >= 1_000_000_000 || retained_route_assignment_hit_count > 0,
        "decode must keep leftover complete layers or reuse retained routed pages; expert_payload_bytes={final_expert_payload_bytes} retained_route_assignment_hit_count={retained_route_assignment_hit_count}"
    );

    // --- Measured assertion 4b: the routed-reuse branch is live (issue #372) ---
    // Unseated layers must warm routed experts during decode and serve later
    // tokens from the hot set instead of re-reading storage every token.
    let hot_expert_partial_route_hit_count = generation_attribution_counter(
        isolated_worker_home.path(),
        "hot_expert_partial_route_hit_count",
    );
    let hot_expert_warm_insert_count =
        generation_attribution_counter(isolated_worker_home.path(), "hot_expert_warm_insert_count");
    assert!(
        hot_expert_partial_route_hit_count > 0 && hot_expert_warm_insert_count > 0,
        "decode must warm routed experts and reuse them; hot_expert_partial_route_hit_count={hot_expert_partial_route_hit_count} hot_expert_warm_insert_count={hot_expert_warm_insert_count}"
    );

    // --- Measured assertion 4c: route-coverage classification (issue #373) ---
    // The all-or-nothing warm-table rule serves a token from RAM only when every
    // routed expert is warm. These counters classify every decode token so the
    // mixed-serving opportunity is measured rather than assumed.
    let hot_expert_route_fully_covered_count = generation_attribution_counter(
        isolated_worker_home.path(),
        "hot_expert_route_fully_covered_count",
    );
    let hot_expert_route_partially_covered_count = generation_attribution_counter(
        isolated_worker_home.path(),
        "hot_expert_route_partially_covered_count",
    );
    let hot_expert_route_fully_missed_count = generation_attribution_counter(
        isolated_worker_home.path(),
        "hot_expert_route_fully_missed_count",
    );
    let hot_expert_route_retained_assignment_count = generation_attribution_counter(
        isolated_worker_home.path(),
        "hot_expert_route_retained_assignment_count",
    );
    let hot_expert_route_missing_assignment_count = generation_attribution_counter(
        isolated_worker_home.path(),
        "hot_expert_route_missing_assignment_count",
    );
    let hot_expert_mixed_route_count =
        generation_attribution_counter(isolated_worker_home.path(), "hot_expert_mixed_route_count");
    let classified_token_count = hot_expert_route_fully_covered_count
        + hot_expert_route_partially_covered_count
        + hot_expert_route_fully_missed_count;
    assert!(
        classified_token_count > 0,
        "decode must classify route coverage per token; classified_token_count={classified_token_count}"
    );
    assert!(
        hot_expert_mixed_route_count > 0,
        "partially covered tokens must be served by the mixed route instead of reading every routed expert from storage; hot_expert_mixed_route_count={hot_expert_mixed_route_count} partially_covered={hot_expert_route_partially_covered_count}"
    );
    assert!(
        hot_expert_route_retained_assignment_count > hot_expert_route_missing_assignment_count,
        "mixed serving must serve the majority of routed assignments from retained RAM; retained={hot_expert_route_retained_assignment_count} missing={hot_expert_route_missing_assignment_count}"
    );

    // --- Measured assertion 5: throughput remains portable evidence ---
    let average_generation_tokens_per_second =
        memory_evidence.final_status["serving_session"]["average_generation_tok_per_second"]
            .as_f64()
            .expect("the completed status should report average generation throughput");
    let average_prefill_tokens_per_second =
        memory_evidence.final_status["serving_session"]["average_prefill_tok_per_second"]
            .as_f64()
            .expect("the completed status should report average prefill throughput");
    assert!(
        average_generation_tokens_per_second.is_finite()
            && average_generation_tokens_per_second > 0.0,
        "decode throughput must be a positive finite measurement"
    );

    // --- Measured assertion 6: the unused ceiling has a named owner (issue #507) ---
    let memory_ceiling_bytes = generation_attribution_counter(
        isolated_worker_home.path(),
        "memory_ceiling_utilization_ceiling_bytes",
    );
    let memory_active_bytes = generation_attribution_counter(
        isolated_worker_home.path(),
        "memory_ceiling_utilization_active_bytes",
    );
    let memory_unused_headroom_bytes = generation_attribution_counter(
        isolated_worker_home.path(),
        "memory_ceiling_utilization_unused_headroom_bytes",
    );
    let memory_reserved_model_core_slack_bytes = generation_attribution_counter(
        isolated_worker_home.path(),
        "memory_ceiling_utilization_reserved_model_core_slack_bytes",
    );
    let memory_reserved_context_growth_bytes = generation_attribution_counter(
        isolated_worker_home.path(),
        "memory_ceiling_utilization_reserved_context_growth_bytes",
    );
    let memory_reserved_activation_and_workspace_bytes = generation_attribution_counter(
        isolated_worker_home.path(),
        "memory_ceiling_utilization_reserved_activation_and_workspace_bytes",
    );
    let memory_unseated_expert_entitlement_bytes = generation_attribution_counter(
        isolated_worker_home.path(),
        "memory_ceiling_utilization_unseated_expert_entitlement_bytes",
    );
    let memory_unexplained_headroom_bytes = generation_attribution_counter(
        isolated_worker_home.path(),
        "memory_ceiling_utilization_unexplained_headroom_bytes",
    );
    let memory_owner_overrun_bytes = generation_attribution_counter(
        isolated_worker_home.path(),
        "memory_ceiling_utilization_owner_overrun_bytes",
    );
    assert!(
        memory_unused_headroom_bytes > 0,
        "decode must observe unused headroom to explain; memory_unused_headroom_bytes={memory_unused_headroom_bytes}"
    );
    let named_headroom_bytes = memory_reserved_model_core_slack_bytes
        + memory_reserved_context_growth_bytes
        + memory_reserved_activation_and_workspace_bytes
        + memory_unseated_expert_entitlement_bytes;
    assert!(
        memory_unexplained_headroom_bytes * 100 <= memory_unused_headroom_bytes,
        "every unused byte must have a named owner or surface as a small residual; unused={memory_unused_headroom_bytes} named={named_headroom_bytes} unexplained={memory_unexplained_headroom_bytes}"
    );
    assert_eq!(
        memory_unexplained_headroom_bytes, 0,
        "the utilization identity must close with no residual; unused={memory_unused_headroom_bytes} named={named_headroom_bytes} unexplained={memory_unexplained_headroom_bytes}"
    );

    // --- Measured assertion 7: the status document carries the same split the
    // attribution counters do (issue #510 observability parity) ---
    let status_utilization =
        &memory_evidence.final_status["mlx_memory_snapshot"]["memory_ceiling_utilization"];
    let status_unused_headroom_bytes = status_utilization["unused_headroom_bytes"]
        .as_u64()
        .expect("the finalized status must publish the utilization decomposition (issue #510)");
    let status_named_bytes = [
        "reserved_model_core_slack_bytes",
        "reserved_context_growth_bytes",
        "reserved_activation_and_workspace_bytes",
        "unseated_expert_entitlement_bytes",
    ]
    .into_iter()
    .map(|field| status_utilization[field].as_u64().unwrap_or(0))
    .sum::<u64>();
    let status_unexplained_bytes = status_utilization["unexplained_headroom_bytes"]
        .as_u64()
        .unwrap_or(u64::MAX);
    let status_owner_overrun_bytes = status_utilization["owner_overrun_bytes"]
        .as_u64()
        .unwrap_or(u64::MAX);
    assert_eq!(
        status_unexplained_bytes, 0,
        "the status-published decomposition must close with no residual: {status_utilization}"
    );
    assert_eq!(
        status_owner_overrun_bytes, 0,
        "the status-published decomposition must report no owner overrun: {status_utilization}"
    );
    assert!(
        status_named_bytes <= status_unused_headroom_bytes,
        "status-published named owners must not overrun the headroom: unused={status_unused_headroom_bytes} named={status_named_bytes}"
    );

    // --- Structural assertion 7: exactly one generation attribution report ---
    assert_eq!(
        generation_attribution_report_count(isolated_worker_home.path()),
        1,
        "exactly one generation attribution report should be written"
    );

    // Diagnostic output: useful for debugging but not asserted.
    let expert_source_read_bytes = generation_expert_source_read_bytes(isolated_worker_home.path());
    let decode_streamed_layer_indices = decode_streamed_layer_indices(isolated_worker_home.path());
    let retained_expert_payload_increments = memory_evidence.retained_expert_payload_bytes.len();
    eprintln!(
        "[ssd-paging-decode-expert-reuse] status=success \
         prompt_tokens={PROMPT_TOKEN_COUNT} \
         maximum_mlx_memory_gb={} \
         final_expert_payload_gb={:.2} \
         final_active_memory_gb={:.2} \
         peak_memory_gb={:.2} \
         expert_source_read_gb={:.2} \
         retained_route_hits={retained_route_assignment_hit_count} \
         hot_expert_partial_route_hits={hot_expert_partial_route_hit_count} \
         hot_expert_warm_insert_count={hot_expert_warm_insert_count} \
         hot_expert_route_fully_covered={hot_expert_route_fully_covered_count} \
         hot_expert_route_partially_covered={hot_expert_route_partially_covered_count} \
         hot_expert_route_fully_missed={hot_expert_route_fully_missed_count} \
         hot_expert_route_retained_assignments={hot_expert_route_retained_assignment_count} \
         hot_expert_route_missing_assignments={hot_expert_route_missing_assignment_count} \
         hot_expert_mixed_routes={hot_expert_mixed_route_count} \
         mem_ceiling_gb={:.2} \
         mem_active_gb={:.2} \
         mem_unused_gb={:.2} \
         mem_reserved_core_slack_gb={:.2} \
         mem_reserved_context_gb={:.2} \
         mem_reserved_workspace_gb={:.2} \
         mem_unseated_entitlement_gb={:.2} \
         mem_unexplained_gb={:.2} \
         mem_owner_overrun_gb={:.2} \
         decode_streamed_layer_count={} \
         retained_payload_increments={} \
         average_prefill_tok_per_second={average_prefill_tokens_per_second:.2} \
         average_generation_tok_per_second={average_generation_tokens_per_second:.2} \
         output_characters={}",
        MAXIMUM_MLX_MEMORY_BYTES / 1_000_000_000,
        final_expert_payload_bytes as f64 / 1e9,
        final_active_memory_bytes as f64 / 1e9,
        peak_memory_bytes as f64 / 1e9,
        expert_source_read_bytes as f64 / 1e9,
        memory_ceiling_bytes as f64 / 1e9,
        memory_active_bytes as f64 / 1e9,
        memory_unused_headroom_bytes as f64 / 1e9,
        memory_reserved_model_core_slack_bytes as f64 / 1e9,
        memory_reserved_context_growth_bytes as f64 / 1e9,
        memory_reserved_activation_and_workspace_bytes as f64 / 1e9,
        memory_unseated_expert_entitlement_bytes as f64 / 1e9,
        memory_unexplained_headroom_bytes as f64 / 1e9,
        memory_owner_overrun_bytes as f64 / 1e9,
        decode_streamed_layer_indices.len(),
        retained_expert_payload_increments,
        completed_stream.model_text.len(),
    );
    preserve_memory_utilization_evidence(
        isolated_worker_home.path(),
        &memory_evidence.final_status,
        &[
            (
                "memory_ceiling_utilization_ceiling_bytes",
                memory_ceiling_bytes,
            ),
            (
                "memory_ceiling_utilization_active_bytes",
                memory_active_bytes,
            ),
            (
                "memory_ceiling_utilization_unused_headroom_bytes",
                memory_unused_headroom_bytes,
            ),
            (
                "memory_ceiling_utilization_reserved_model_core_slack_bytes",
                memory_reserved_model_core_slack_bytes,
            ),
            (
                "memory_ceiling_utilization_reserved_context_growth_bytes",
                memory_reserved_context_growth_bytes,
            ),
            (
                "memory_ceiling_utilization_reserved_activation_and_workspace_bytes",
                memory_reserved_activation_and_workspace_bytes,
            ),
            (
                "memory_ceiling_utilization_unseated_expert_entitlement_bytes",
                memory_unseated_expert_entitlement_bytes,
            ),
            (
                "memory_ceiling_utilization_unexplained_headroom_bytes",
                memory_unexplained_headroom_bytes,
            ),
            (
                "memory_ceiling_utilization_owner_overrun_bytes",
                memory_owner_overrun_bytes,
            ),
        ],
    );
}

/// Writes the utilization decomposition and attribution counters into evidence
/// that survives the run (issue #510), so runs can be compared later instead of
/// the numbers living only in this test's stdout.
struct ProgressiveExpertMemoryEvidence {
    retained_expert_payload_bytes: Vec<u64>,
    final_status: Value,
}

async fn observe_ssd_paging_decode_expert_reuse(
    server_address: std::net::SocketAddr,
) -> ProgressiveExpertMemoryEvidence {
    let deadline = Instant::now() + JOURNEY_TIMEOUT;
    let mut observed_prompt_processing = false;
    let mut retained_expert_payload_bytes = Vec::new();
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
        let snapshot_source = status_document["mlx_memory_snapshot"]["source"].as_str();
        if observed_prompt_processing
            && status_document["activity"] == "idle"
            && matches!(snapshot_source, Some("finalized" | "idle_poll"))
        {
            return ProgressiveExpertMemoryEvidence {
                retained_expert_payload_bytes,
                final_status: status_document,
            };
        }
        assert!(Instant::now() < deadline);
        sleep(Duration::from_millis(100)).await;
    }
}

struct CompletedStream {
    model_text: String,
    finish_reason: Option<String>,
}

async fn consume_completed_stream(
    mut streamed_completion: StreamResponse<Value>,
) -> CompletedStream {
    let mut streamed_model_text = String::new();
    let mut finish_reason = None;
    while let Some(stream_item) = streamed_completion.next().await {
        let stream_chunk = stream_item.expect("the public REST stream should remain healthy");
        for choice in stream_chunk["choices"].as_array().into_iter().flatten() {
            if let Some(content_fragment) = choice["delta"]["content"].as_str() {
                streamed_model_text.push_str(content_fragment);
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                finish_reason = Some(reason.to_owned());
            }
        }
    }
    CompletedStream {
        model_text: streamed_model_text.trim().to_owned(),
        finish_reason,
    }
}

fn write_acceptance_config(isolated_worker_home: &Path, model_directory: &Path) {
    let configuration_directory = isolated_worker_home.join(".astronomical-dev");
    fs::create_dir(&configuration_directory)
        .expect("the memory-management configuration directory should be created");
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
            "fixed_prompt_processing_chunk_size_tokens": 2_048,
            "fixed_ssd_streaming_prompt_processing_chunk_size_tokens": 2_048,
            "prefill_graph_submission_layer_interval": 0,
            "experimental_ssd_paging_prefill_graph_submission_layer_interval": 1,
            "experimental_ssd_paging_generation_graph_submission_layer_interval": 1,
        },
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the memory-management configuration should serialize"),
    )
    .expect("the memory-management configuration should be written");
}
