//! Reproduces the interaction between cached append-only prefill, expert
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

use async_openai::{Client, config::OpenAIConfig};
use serde_json::{Value, json};
use tokio::time::{Duration, timeout};

use crate::common::real_model_rest_server::{
    JOURNEY_TIMEOUT, launch_real_model_rest_server, stop_real_model_rest_server,
};
use observe::{
    assert_completed_request, completion_request, execute_observed_request, print_request_records,
};
use reports::{
    assert_reported_interaction, persist_prefill_throughput_summary, print_comparison_summary,
    read_interaction_reports,
};
use support::{
    artifact_directory_regular_file_bytes, interaction_instance_paths,
    persist_qualification_evidence, write_interaction_config,
};

pub(super) const LOG_MARKER: &str = "[cached-suffix-streaming-prefill]";
const OQ6E_MODEL_ID: &str = crate::common::ORNITH_SSD_STREAMING_MODEL_ID;
// This qualification cell is deliberately below complete-resident prefill need
// for the configured artifact, while the launcher still clamps it to the host's
// machine-derived ceiling. Production policy contains no corresponding constant.
const INITIAL_PROMPT_TOKEN_COUNT: usize = 10_000;
const PREFILL_CHUNK_TOKEN_COUNT: u32 = 2_048;
/// One configured chunk plus a remainder, so leftover unseated layers stream
/// more than once unless they stay operation-local for every chunk.
const APPEND_FOLLOW_UP_TOKEN_COUNT: usize = 5_000;
/// Long enough that a 50 tok/s control-span stall cannot finish inside the
/// 115-second journey bound, while a healthy high-RAM suffix still can.
const HIGH_RAM_APPEND_FOLLOW_UP_TOKEN_COUNT: usize = 16_000;
/// Far above the 53 tok/s Pi stall, far below a fully resident warm suffix.
const MINIMUM_HIGH_RAM_APPEND_PREFILL_TOKENS_PER_SECOND: f64 = 200.0;
const PAGING_GRAPH_SUBMISSION_LAYER_INTERVAL: u32 = 1;
/// Stable Development decode interval: eval every third layer so a one-token
/// routed page can detach without a host launch after every decoder index.
const PAGING_GENERATION_GRAPH_SUBMISSION_LAYER_INTERVAL: u32 = 3;
// A short generation budget keeps the residency journey inside the bounded
// acceptance window. The long prefill source above supplies the memory pressure
// this journey qualifies without making success depend on one machine's speed.
pub(super) const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 200;
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");
const ROMEO_AND_JULIET_LONG_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_50000_romeo_and_juliet_words.txt");
/// Long enough to reproduce the 150k-token Pi conversation OOM: the context
/// reserve and attention workspace for 100k tokens consume enough of the
/// 40 GB ceiling that promoting one missing layer during the first chunk
/// exceeds it. This is an endurance / OOM-reproduction test, not a 120s gate.
const LONG_CONVERSATION_COLD_PROMPT_TOKEN_COUNT: usize = 2_000;
const LONG_CONVERSATION_TIMEOUT: Duration = Duration::from_secs(1200);

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches one real worker and reproduces cached prefill/decode expert residency"]
async fn should_complete_cold_and_cached_append_requests_with_consistent_prefill_decode_residency()
{
    timeout(
        JOURNEY_TIMEOUT,
        run_interaction_journey(StreamingPrefillJourneyKind::HalfModelReread),
    )
    .await
    .expect("the prefill/decode residency interaction must finish within 115 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches one real worker and reproduces the high-RAM tool-prefixed cached-suffix stall"]
async fn should_keep_high_ram_tool_prefixed_cached_suffix_prefill_responsive() {
    timeout(
        JOURNEY_TIMEOUT,
        run_interaction_journey(StreamingPrefillJourneyKind::HighRamResponsiveSuffix),
    )
    .await
    .expect("the high-RAM cached suffix must finish within 115 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches one real worker against the near-disk oQ6e artifact and a long cached suffix"]
async fn should_keep_near_disk_oq6e_cached_suffix_prefill_responsive() {
    timeout(
        JOURNEY_TIMEOUT,
        run_interaction_journey(StreamingPrefillJourneyKind::NearDiskOq6eLongSuffix),
    )
    .await
    .expect("the near-disk oQ6e cached suffix must finish within 115 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches one real worker against the oQ6e artifact under a tight ceiling that forces the prefill OOM-retry-reread stall"]
async fn should_recover_from_prefill_oom_without_stalling_on_oq6e() {
    timeout(
        JOURNEY_TIMEOUT,
        run_interaction_journey(StreamingPrefillJourneyKind::NearDiskOq6eTightCeiling),
    )
    .await
    .expect("the tight-ceiling oQ6e prefill must recover from OOM and finish within 115 seconds");
}

fn journey_kind_label(kind: StreamingPrefillJourneyKind) -> &'static str {
    match kind {
        StreamingPrefillJourneyKind::HalfModelReread => "half-model-reread",
        StreamingPrefillJourneyKind::HighRamResponsiveSuffix => "high-ram-responsive-suffix",
        StreamingPrefillJourneyKind::NearDiskOq6eLongSuffix => "near-disk-oq6e-long-suffix",
        StreamingPrefillJourneyKind::NearDiskOq6eTightCeiling => "near-disk-oq6e-tight-ceiling",
        StreamingPrefillJourneyKind::HalfModelOq6eLongConversation => "oq6e-long-conversation",
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "endurance test: oQ6e long conversation at 50 percent of on-disk SI size as MLX RAM"]
async fn should_measure_oq6e_long_conversation_at_half_disk_ram() {
    timeout(
        LONG_CONVERSATION_TIMEOUT,
        run_interaction_journey(StreamingPrefillJourneyKind::HalfModelOq6eLongConversation),
    )
    .await
    .expect("the oQ6e half-disk-RAM conversation must produce output within 1200 seconds");
}

#[derive(Clone, Copy)]
enum StreamingPrefillJourneyKind {
    HalfModelReread,
    HighRamResponsiveSuffix,
    NearDiskOq6eLongSuffix,
    /// Tight ceiling that forces the OOM-retry-reread stall on the first
    /// prefill chunk. Promoting two missing layers exceeds the budget and
    /// the retry re-reads them from SSD before OOMing again.
    NearDiskOq6eTightCeiling,
    /// Same long-conversation shape on oQ6e, with MLX RAM set to 20 GB SI.
    HalfModelOq6eLongConversation,
}

async fn run_interaction_journey(journey_kind: StreamingPrefillJourneyKind) {
    let model_id = OQ6E_MODEL_ID;
    let model_directory = crate::common::configured_model_artifact_directory_by_id(model_id);
    // Persistent (not tempfile) so worker logs survive a timeout or panic.
    // tempfile::tempdir deletes everything when dropped, including on abort.
    let isolated_worker_home_path = support::qualification_evidence_root()
        .join(format!("worker-home-{}", journey_kind_label(journey_kind)));
    fs::remove_dir_all(&isolated_worker_home_path).ok();
    fs::create_dir_all(&isolated_worker_home_path)
        .expect("the persistent worker home should be created");
    // Canonicalize so the prompt-cache validator sees no ../.. components.
    let isolated_worker_home_path = isolated_worker_home_path
        .canonicalize()
        .unwrap_or(isolated_worker_home_path);
    eprintln!(
        "{LOG_MARKER} request=journey status=worker_home path={}",
        isolated_worker_home_path.display()
    );
    let artifact_payload_bytes = artifact_directory_regular_file_bytes(&model_directory);
    let allocated_mlx_memory_bytes = match journey_kind {
        StreamingPrefillJourneyKind::HalfModelReread => artifact_payload_bytes / 2,
        // Same 20 GB SI ceiling as the Stable oQ6e streaming screenshot.
        StreamingPrefillJourneyKind::HalfModelOq6eLongConversation => 20_000_000_000,
        // On-disk bytes plus one SI gigabyte of first-forward workspace. That
        // is the 38.6 GB model / 40 GB RAM shape without a machine-specific cap.
        StreamingPrefillJourneyKind::HighRamResponsiveSuffix => {
            artifact_payload_bytes.saturating_add(1_000_000_000)
        }
        // Near-complete oQ6e ceiling: on-disk size plus a small workspace.
        StreamingPrefillJourneyKind::NearDiskOq6eLongSuffix => 32_000_000_000,
        StreamingPrefillJourneyKind::NearDiskOq6eTightCeiling => 20_000_000_000,
    };
    let initial_prompt_token_count = match journey_kind {
        StreamingPrefillJourneyKind::NearDiskOq6eLongSuffix
        | StreamingPrefillJourneyKind::NearDiskOq6eTightCeiling => 8_000,
        StreamingPrefillJourneyKind::HalfModelOq6eLongConversation => {
            LONG_CONVERSATION_COLD_PROMPT_TOKEN_COUNT
        }
        _ => INITIAL_PROMPT_TOKEN_COUNT,
    };
    let append_follow_up_token_count = match journey_kind {
        StreamingPrefillJourneyKind::HalfModelReread => APPEND_FOLLOW_UP_TOKEN_COUNT,
        StreamingPrefillJourneyKind::HighRamResponsiveSuffix => {
            HIGH_RAM_APPEND_FOLLOW_UP_TOKEN_COUNT
        }
        StreamingPrefillJourneyKind::NearDiskOq6eLongSuffix
        | StreamingPrefillJourneyKind::NearDiskOq6eTightCeiling => 8_000,
        // The long conversation test sends one very large cold prompt; no append.
        StreamingPrefillJourneyKind::HalfModelOq6eLongConversation => 0,
    };
    assert!(
        allocated_mlx_memory_bytes > 0,
        "the discovered artifact must have a positive on-disk payload"
    );
    write_interaction_config(
        &isolated_worker_home_path,
        &model_directory,
        allocated_mlx_memory_bytes,
    );
    let short_source_repeated = ROMEO_AND_JULIET_SOURCE.repeat(4);
    let cold_prompt_source = &short_source_repeated[..];
    let initial_user_message = crate::common::exact_model_prompt::build_exact_model_prompt_content(
        &model_directory,
        cold_prompt_source,
        "Summarize the play's central conflict and the decisions that produce its tragic ending.",
        initial_prompt_token_count,
    );
    let follow_up_user_message = if append_follow_up_token_count == 0 {
        String::new()
    } else {
        crate::common::exact_model_prompt::build_exact_model_prompt_content(
            &model_directory,
            ROMEO_AND_JULIET_LONG_SOURCE,
            "Relate that summary to the consequences of haste. Use this additional Romeo and Juliet context without quoting it.",
            append_follow_up_token_count,
        )
    };
    let timeout_duration = match journey_kind {
        StreamingPrefillJourneyKind::HalfModelOq6eLongConversation => LONG_CONVERSATION_TIMEOUT,
        _ => JOURNEY_TIMEOUT,
    };
    eprintln!(
        "{LOG_MARKER} request=journey status=start timeout_seconds={} artifact_payload_bytes={artifact_payload_bytes} artifact_payload_gb={:.3} allocated_mlx_memory_bytes={allocated_mlx_memory_bytes} allocated_mlx_memory_gb={:.3} initial_prompt_tokens={initial_prompt_token_count} fixed_prefill_tokens={PREFILL_CHUNK_TOKEN_COUNT} paging_graph_submission_layer_interval={PAGING_GRAPH_SUBMISSION_LAYER_INTERVAL} persistent_prompt_cache_enabled=true",
        timeout_duration.as_secs(),
        artifact_payload_bytes as f64 / 1_000_000_000.0,
        allocated_mlx_memory_bytes as f64 / 1_000_000_000.0,
    );

    let real_model_rest_server = launch_real_model_rest_server(
        model_id,
        model_directory,
        &isolated_worker_home_path,
        allocated_mlx_memory_bytes,
    )
    .await;
    let server_address = real_model_rest_server.server_address;
    let openai_client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(format!("http://{server_address}/v1"))
            .with_api_key("local-residency-interaction-client"),
    );

    let cold_request = completion_request(
        model_id,
        json!([{"role": "user", "content": initial_user_message}]),
    );
    let logging_directory =
        interaction_instance_paths(&isolated_worker_home_path).logging_directory();
    let cold_outcome = execute_observed_request(
        &openai_client,
        server_address,
        &logging_directory,
        "cold",
        cold_request,
    )
    .await;
    persist_qualification_evidence(&isolated_worker_home_path);
    print_request_records(&read_interaction_reports(&isolated_worker_home_path));
    assert_completed_request(&cold_outcome, "cold");

    let skip_append = matches!(
        journey_kind,
        StreamingPrefillJourneyKind::HalfModelOq6eLongConversation,
    );
    if skip_append {
        // Long conversation endurance test: only the cold request matters.
        // The cold itself must not stall at 0 percent; the OOM-retry-skip
        // should halve the chunk and keep progress moving.
        stop_real_model_rest_server(real_model_rest_server).await;
        persist_qualification_evidence(&isolated_worker_home_path);
        let reports = read_interaction_reports(&isolated_worker_home_path);
        print_request_records(&reports);
    } else {
        let append_request = completion_request(
            model_id,
            json!([
                {"role": "user", "content": initial_user_message},
                {"role": "assistant", "content": cold_outcome.model_text},
                {"role": "user", "content": follow_up_user_message},
            ]),
        );
        let append_outcome = execute_observed_request(
            &openai_client,
            server_address,
            &logging_directory,
            "append",
            append_request,
        )
        .await;
        persist_qualification_evidence(&isolated_worker_home_path);
        let reports = read_interaction_reports(&isolated_worker_home_path);
        print_request_records(&reports);
        assert_completed_request(&append_outcome, "append");
        stop_real_model_rest_server(real_model_rest_server).await;
        persist_prefill_throughput_summary(
            &reports,
            artifact_payload_bytes,
            allocated_mlx_memory_bytes,
        );
        print_comparison_summary(&reports, &cold_outcome, &append_outcome);
        assert_reported_interaction(
            &reports,
            &cold_outcome,
            &append_outcome,
            allocated_mlx_memory_bytes,
        );
        if matches!(
            journey_kind,
            StreamingPrefillJourneyKind::HighRamResponsiveSuffix
                | StreamingPrefillJourneyKind::NearDiskOq6eLongSuffix
        ) {
            let append_prefill_tokens_per_second =
                reports.performance_records[1]["prefill_tok_per_second"]
                    .as_f64()
                    .unwrap_or(0.0);
            assert!(
                append_prefill_tokens_per_second
                    >= MINIMUM_HIGH_RAM_APPEND_PREFILL_TOKENS_PER_SECOND,
                "high-RAM tool-prefixed cached suffix must stay responsive: append_prefill_tokens_per_second={append_prefill_tokens_per_second:.2} minimum={MINIMUM_HIGH_RAM_APPEND_PREFILL_TOKENS_PER_SECOND:.2}"
            );
            assert_eq!(
                append_outcome.live_evidence.final_status["status"], "ready",
                "a slow or over-budget suffix must not kill the worker"
            );
            assert!(
                cold_outcome.live_evidence.maximum_expert_payload_bytes
                    >= artifact_payload_bytes / 2,
                "a near-disk-size ceiling must load a complete-model-scale expert payload on the first request: expert_bytes={} artifact_bytes={artifact_payload_bytes}",
                cold_outcome.live_evidence.maximum_expert_payload_bytes
            );
            assert!(
                append_outcome
                    .live_evidence
                    .longest_unmoving_prefill_seconds
                    < 20.0,
                "a seated cached suffix must keep publishing prefill progress: longest_unmoving_prefill_seconds={:.1}",
                append_outcome
                    .live_evidence
                    .longest_unmoving_prefill_seconds
            );
        }
    }
}
