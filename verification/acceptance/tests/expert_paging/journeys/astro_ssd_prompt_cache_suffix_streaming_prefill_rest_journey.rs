//! Reproduces the interaction between append-only prefill over an `astro_ssd_prompt_cache` prefix, expert
//! streaming, retained RAM topology, and the prefill-to-decode handoff.
//!
//! This permanent journey is intentionally verbose. It is the focused command a
//! maintainer reruns while changing phase-aware residency so a stalled token
//! frontier, memory growth, topology churn, or a same-request streamed-layer
//! reread remains visible.

mod observe;
mod reports;
mod support;

use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use serial_test::serial;
use tokio::time::timeout;

use crate::support::openai_client::LocalOpenAiClient;
use crate::support::serving_rest::{
    SSD_JOURNEY_TIMEOUT, launch_real_model_rest_server, stop_real_model_rest_server,
};
use observe::{
    ObservedRequestOutcome, assert_completed_request, completion_request, execute_observed_request,
    print_request_records,
};
use reports::{
    InteractionReports, assert_admitted_complete_layers_are_reused_across_prefill_chunks,
    assert_reported_interaction, persist_prefill_throughput_summary, print_comparison_summary,
    read_interaction_reports,
};
use support::{
    artifact_directory_regular_file_bytes, interaction_instance_paths, persist_acceptance_evidence,
    write_interaction_config,
};

pub(super) const LOG_MARKER: &str = "[astro-ssd-prompt-cache-suffix-streaming-prefill]";
// This acceptance cell is deliberately below complete-resident prefill need
// for the configured artifact, while the launcher still clamps it to the host's
// machine-derived ceiling. Production policy contains no corresponding constant.
const INITIAL_PROMPT_TOKEN_COUNT: usize = 10_000;
const PREFILL_CHUNK_TOKEN_COUNT: u32 = 2_048;
/// One configured chunk plus a remainder, so leftover unseated layers stream
/// more than once unless they stay operation-local for every chunk.
const APPEND_FOLLOW_UP_TOKEN_COUNT: usize = 5_000;
/// Long enough that a 50 tok/s control-span stall cannot finish inside the
/// 60-second journey bound, while a healthy high-RAM suffix still can.
const HIGH_RAM_APPEND_FOLLOW_UP_TOKEN_COUNT: usize = 16_000;
/// Far above the 53 tok/s Pi stall, far below a fully resident warm suffix.
const MINIMUM_HIGH_RAM_APPEND_PREFILL_TOKENS_PER_SECOND: f64 = 200.0;
/// Stable Development decode interval: eval every third layer so a one-token
/// routed page can detach without a host launch after every decoder index.
const PAGING_GENERATION_GRAPH_SUBMISSION_LAYER_INTERVAL: u32 = 3;
// A short generation budget keeps the residency journey inside the bounded
// acceptance window. The long prefill source above supplies the memory pressure
// this journey runs without making success depend on one machine's speed.
pub(super) const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 200;
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");
const ROMEO_AND_JULIET_LONG_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_50000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches one real worker and reproduces prefill/decode expert residency after an astro SSD prompt cache restore"]
#[serial]
async fn should_complete_cold_and_astro_ssd_prompt_cache_restored_append_requests_with_consistent_prefill_decode_residency()
 {
    timeout(
        SSD_JOURNEY_TIMEOUT,
        run_interaction_journey(StreamingPrefillJourneyKind::HalfModelReread),
    )
    .await
    .expect("the prefill/decode residency SSD journey must finish within 120 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches one real worker and reproduces the high-RAM tool-prefixed astro SSD prompt cache suffix stall"]
#[serial]
async fn should_keep_high_ram_tool_prefixed_astro_ssd_prompt_cache_suffix_prefill_responsive() {
    timeout(
        SSD_JOURNEY_TIMEOUT,
        run_interaction_journey(StreamingPrefillJourneyKind::HighRamResponsiveSuffix),
    )
    .await
    .expect(
        "the high-RAM astro SSD prompt cache suffix SSD journey must finish within 120 seconds",
    );
}
#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches one real worker against the large sparse MoE streaming e2e fixture under a tight ceiling that forces the prefill OOM-retry-reread stall"]
#[serial]
async fn should_recover_from_prefill_oom_without_stalling() {
    timeout(
        SSD_JOURNEY_TIMEOUT,
        run_interaction_journey(StreamingPrefillJourneyKind::TightCeiling),
    )
    .await
    .expect("the tight-ceiling large sparse MoE streaming prefill must finish within 120 seconds");
}

fn journey_kind_label(kind: StreamingPrefillJourneyKind) -> &'static str {
    match kind {
        StreamingPrefillJourneyKind::HalfModelReread => "half-model-reread",
        StreamingPrefillJourneyKind::HighRamResponsiveSuffix => "high-ram-responsive-suffix",
        StreamingPrefillJourneyKind::TightCeiling => "tight-ceiling",
    }
}

#[derive(Clone, Copy)]
enum StreamingPrefillJourneyKind {
    HalfModelReread,
    HighRamResponsiveSuffix,
    /// A limit so tight that the prompt once stalled: a span hit the limit,
    /// was retried, re-read its layers from the SSD and hit the limit
    /// again. The request must be served without that stall.
    TightCeiling,
}

async fn run_interaction_journey(journey_kind: StreamingPrefillJourneyKind) {
    let model_id = crate::support::large_sparse_moe_streaming_model_id();
    let model_directory = crate::support::configured_installed_model_directory_by_id(model_id);
    let isolated_worker_home_path = prepare_persistent_worker_home(journey_kind);
    let artifact_payload_bytes = artifact_directory_regular_file_bytes(&model_directory);
    let journey_plan = plan_journey(journey_kind, artifact_payload_bytes);
    let allocated_mlx_memory_bytes = journey_plan.allocated_mlx_memory_bytes;
    write_interaction_config(
        &isolated_worker_home_path,
        &model_directory,
        allocated_mlx_memory_bytes,
    );
    let (initial_user_message, follow_up_user_message) =
        build_journey_user_messages(&model_directory, &journey_plan);
    log_journey_start(&journey_plan, artifact_payload_bytes);

    let real_model_rest_server = launch_real_model_rest_server(
        model_id,
        model_directory,
        &isolated_worker_home_path,
        allocated_mlx_memory_bytes,
    )
    .await;
    let server_address = real_model_rest_server.server_address;
    let openai_client =
        LocalOpenAiClient::new(server_address, "local-residency-interaction-client");

    let cold_request = completion_request(
        model_id,
        json!([{"role": "user", "content": initial_user_message}]),
    );
    let logging_directory =
        interaction_instance_paths(&isolated_worker_home_path).logging_directory();
    let (cold_outcome, _) = execute_request_and_persist_evidence(
        &openai_client,
        server_address,
        &logging_directory,
        &isolated_worker_home_path,
        "cold",
        cold_request,
    )
    .await;

    let append_request = completion_request(
        model_id,
        json!([
            {"role": "user", "content": initial_user_message},
            {"role": "assistant", "content": cold_outcome.model_text},
            {"role": "user", "content": follow_up_user_message},
        ]),
    );
    let (append_outcome, reports) = execute_request_and_persist_evidence(
        &openai_client,
        server_address,
        &logging_directory,
        &isolated_worker_home_path,
        "append",
        append_request,
    )
    .await;
    stop_real_model_rest_server(real_model_rest_server).await;
    assert_journey_expectations(
        journey_kind,
        &reports,
        &cold_outcome,
        &append_outcome,
        artifact_payload_bytes,
        allocated_mlx_memory_bytes,
    );
}

/// The memory ceiling and the two prompt sizes one journey kind runs with.
struct JourneyPlan {
    allocated_mlx_memory_bytes: u64,
    initial_prompt_token_count: usize,
    append_follow_up_token_count: usize,
}

fn plan_journey(
    journey_kind: StreamingPrefillJourneyKind,
    artifact_payload_bytes: u64,
) -> JourneyPlan {
    let allocated_mlx_memory_bytes = match journey_kind {
        StreamingPrefillJourneyKind::HalfModelReread => artifact_payload_bytes / 2,
        // On-disk bytes plus one SI gigabyte of first-forward workspace. That
        // is the 38.6 GB model / 40 GB RAM shape without a machine-specific cap.
        StreamingPrefillJourneyKind::HighRamResponsiveSuffix => {
            artifact_payload_bytes.saturating_add(1_000_000_000)
        }
        StreamingPrefillJourneyKind::TightCeiling => 20_000_000_000,
    };
    let initial_prompt_token_count = match journey_kind {
        StreamingPrefillJourneyKind::TightCeiling => 8_000,
        _ => INITIAL_PROMPT_TOKEN_COUNT,
    };
    let append_follow_up_token_count = match journey_kind {
        StreamingPrefillJourneyKind::HalfModelReread => APPEND_FOLLOW_UP_TOKEN_COUNT,
        StreamingPrefillJourneyKind::HighRamResponsiveSuffix => {
            HIGH_RAM_APPEND_FOLLOW_UP_TOKEN_COUNT
        }
        StreamingPrefillJourneyKind::TightCeiling => 8_000,
    };
    assert!(
        allocated_mlx_memory_bytes > 0,
        "the discovered artifact must have a positive on-disk payload"
    );
    JourneyPlan {
        allocated_mlx_memory_bytes,
        initial_prompt_token_count,
        append_follow_up_token_count,
    }
}

fn log_journey_start(journey_plan: &JourneyPlan, artifact_payload_bytes: u64) {
    let allocated_mlx_memory_bytes = journey_plan.allocated_mlx_memory_bytes;
    let initial_prompt_token_count = journey_plan.initial_prompt_token_count;
    eprintln!(
        "{LOG_MARKER} request=journey status=start timeout_seconds={} artifact_payload_bytes={artifact_payload_bytes} artifact_payload_gb={:.3} allocated_mlx_memory_bytes={allocated_mlx_memory_bytes} allocated_mlx_memory_gb={:.3} initial_prompt_tokens={initial_prompt_token_count} fixed_prefill_tokens={PREFILL_CHUNK_TOKEN_COUNT} astro_ssd_prompt_cache_enabled=true",
        SSD_JOURNEY_TIMEOUT.as_secs(),
        artifact_payload_bytes as f64 / 1_073_741_824.0,
        allocated_mlx_memory_bytes as f64 / 1_073_741_824.0,
    );
}

fn prepare_persistent_worker_home(journey_kind: StreamingPrefillJourneyKind) -> PathBuf {
    // Persistent (not tempfile) so worker logs survive a timeout or panic.
    // tempfile::tempdir deletes everything when dropped, including on abort.
    let isolated_worker_home_path = support::acceptance_evidence_root()
        .join(format!("worker-home-{}", journey_kind_label(journey_kind)));
    fs::remove_dir_all(&isolated_worker_home_path).ok();
    fs::create_dir_all(&isolated_worker_home_path)
        .expect("the persistent worker home should be created");
    // Canonicalize so the `astro_ssd_prompt_cache` validator sees no ../.. components.
    let isolated_worker_home_path = isolated_worker_home_path
        .canonicalize()
        .unwrap_or(isolated_worker_home_path);
    eprintln!(
        "{LOG_MARKER} request=journey status=worker_home path={}",
        isolated_worker_home_path.display()
    );
    isolated_worker_home_path
}

/// Returns the cold request's user message and the follow-up message the
/// append request adds to the same conversation (empty when the plan appends
/// no tokens).
fn build_journey_user_messages(
    model_directory: &Path,
    journey_plan: &JourneyPlan,
) -> (String, String) {
    let short_source_repeated = ROMEO_AND_JULIET_SOURCE.repeat(4);
    let cold_prompt_source = &short_source_repeated[..];
    let initial_user_message = crate::support::exact_model_prompt::build_exact_model_prompt_content(
        model_directory,
        cold_prompt_source,
        "Summarize the play's central conflict and the decisions that produce its tragic ending.",
        journey_plan.initial_prompt_token_count,
    );
    let follow_up_user_message = if journey_plan.append_follow_up_token_count == 0 {
        String::new()
    } else {
        crate::support::exact_model_prompt::build_exact_model_prompt_content(
            model_directory,
            ROMEO_AND_JULIET_LONG_SOURCE,
            "Relate that summary to the consequences of haste. Use this additional Romeo and Juliet context without quoting it.",
            journey_plan.append_follow_up_token_count,
        )
    };
    (initial_user_message, follow_up_user_message)
}

async fn execute_request_and_persist_evidence(
    openai_client: &LocalOpenAiClient,
    server_address: SocketAddr,
    logging_directory: &Path,
    isolated_worker_home_path: &Path,
    request_label: &'static str,
    completion_request: Value,
) -> (ObservedRequestOutcome, InteractionReports) {
    let request_outcome = execute_observed_request(
        openai_client,
        server_address,
        logging_directory,
        request_label,
        completion_request,
    )
    .await;
    persist_acceptance_evidence(isolated_worker_home_path);
    let reports = read_interaction_reports(isolated_worker_home_path);
    print_request_records(&reports);
    assert_completed_request(&request_outcome, request_label);
    (request_outcome, reports)
}

fn assert_journey_expectations(
    journey_kind: StreamingPrefillJourneyKind,
    reports: &InteractionReports,
    cold_outcome: &ObservedRequestOutcome,
    append_outcome: &ObservedRequestOutcome,
    artifact_payload_bytes: u64,
    allocated_mlx_memory_bytes: u64,
) {
    persist_prefill_throughput_summary(reports, artifact_payload_bytes, allocated_mlx_memory_bytes);
    print_comparison_summary(reports, cold_outcome, append_outcome);
    assert_reported_interaction(
        reports,
        cold_outcome,
        append_outcome,
        allocated_mlx_memory_bytes,
    );
    if matches!(journey_kind, StreamingPrefillJourneyKind::HalfModelReread) {
        assert_admitted_complete_layers_are_reused_across_prefill_chunks(reports);
    }
    if matches!(
        journey_kind,
        StreamingPrefillJourneyKind::HighRamResponsiveSuffix
    ) {
        assert_high_ram_suffix_stayed_responsive(
            reports,
            cold_outcome,
            append_outcome,
            artifact_payload_bytes,
        );
    }
}

fn assert_high_ram_suffix_stayed_responsive(
    reports: &InteractionReports,
    cold_outcome: &ObservedRequestOutcome,
    append_outcome: &ObservedRequestOutcome,
    artifact_payload_bytes: u64,
) {
    let append_prefill_tokens_per_second = reports.performance_records[1]["prefill_tok_per_second"]
        .as_f64()
        .unwrap_or(0.0);
    assert!(
        append_prefill_tokens_per_second >= MINIMUM_HIGH_RAM_APPEND_PREFILL_TOKENS_PER_SECOND,
        "high-RAM tool-prefixed astro SSD prompt cache suffix must stay responsive: append_prefill_tokens_per_second={append_prefill_tokens_per_second:.2} minimum={MINIMUM_HIGH_RAM_APPEND_PREFILL_TOKENS_PER_SECOND:.2}"
    );
    assert_eq!(
        append_outcome.live_evidence.final_status["status"], "ready",
        "a slow or over-budget suffix must not kill the worker"
    );
    assert!(
        cold_outcome.live_evidence.maximum_expert_payload_bytes >= artifact_payload_bytes / 2,
        "a near-disk-size ceiling must load a complete-model-scale expert payload on the first request: expert_bytes={} artifact_bytes={artifact_payload_bytes}",
        cold_outcome.live_evidence.maximum_expert_payload_bytes
    );
    assert!(
        append_outcome
            .live_evidence
            .longest_unmoving_prefill_seconds
            < 20.0,
        "a seated astro SSD prompt cache suffix must keep publishing prefill progress: longest_unmoving_prefill_seconds={:.1}",
        append_outcome
            .live_evidence
            .longest_unmoving_prefill_seconds
    );
}
