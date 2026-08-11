use std::time::Duration;

use astronomical_ipc_protocol::RequestId;
use astronomical_runtime_integration::MlxMemoryLimits;
use tokio::time::timeout;

use super::expert_paging_prefill_performance::prepare_reproduced_long_prompt_token_ids_for_model;
use super::performance_attribution::{
    counter_amount, create_attributed_engine, generation_report_for_request,
    load_engine_with_progress, read_attribution_report_documents, run_attributed_generation,
};

const QWOPUS_MODEL_ID: &str = "Qwopus3.5-122B-A10B-Kimi-K2.6-destill-healed-abliterated-MLX-4bit";
const QWOPUS_INPUT_TOKEN_COUNT: usize = 1_181;
const QWOPUS_OUTPUT_TOKEN_COUNT: u16 = 300;
const OQ4E_MODEL_DIRECTORY_NAME: &str = "Qwen3.6-35B-A3B-oQ4e-mtp";
const OQ4E_MODEL_ID: &str = "Jundot/Qwen3.6-35B-A3B-oQ4e-mtp";
const OQ4E_INPUT_TOKEN_COUNT: usize = 1_024;
const OQ4E_OUTPUT_TOKEN_COUNT: u16 = 512;
const OQ4E_MLX_MEMORY_LIMIT_BYTES: usize = 10_000_000_000;
const FIXED_PREFILL_CHUNCK_TOKENS: u32 = 2_048;
const TEST_TIMEOUT: Duration = Duration::from_secs(120);
const QWOPUS_REQUEST_ID: u64 = 9_100;
const OQ4E_REQUEST_ID: u64 = 9_101;

#[tokio::test]
#[ignore = "loads Qwopus with 1181 input tokens and measures 300-token route reuse"]
async fn should_measure_qwopus_previous_token_expert_route_reuse() {
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    timeout(
        TEST_TIMEOUT,
        run_expert_route_reuse_probe(
            QWOPUS_MODEL_ID,
            QWOPUS_MODEL_ID,
            QWOPUS_INPUT_TOKEN_COUNT,
            QWOPUS_OUTPUT_TOKEN_COUNT,
            QWOPUS_REQUEST_ID,
            "qwopus-route-reuse",
            &mlx_memory_limits,
        ),
    )
    .await
    .expect("the Qwopus expert-route reuse probe must finish within 120 seconds");
}

#[tokio::test]
#[ignore = "loads oQ4e with a 10 GB MLX limit and measures 512-token route reuse"]
async fn should_measure_oq4e_previous_token_expert_route_reuse_with_ten_gb_limit() {
    let mlx_memory_limits =
        MlxMemoryLimits::new(OQ4E_MLX_MEMORY_LIMIT_BYTES, OQ4E_MLX_MEMORY_LIMIT_BYTES)
            .expect("the oQ4e route-reuse memory limits should be valid");
    timeout(
        TEST_TIMEOUT,
        run_expert_route_reuse_probe(
            OQ4E_MODEL_DIRECTORY_NAME,
            OQ4E_MODEL_ID,
            OQ4E_INPUT_TOKEN_COUNT,
            OQ4E_OUTPUT_TOKEN_COUNT,
            OQ4E_REQUEST_ID,
            "oq4e-route-reuse",
            &mlx_memory_limits,
        ),
    )
    .await
    .expect("the oQ4e expert-route reuse probe must finish within 120 seconds");
}

#[allow(clippy::too_many_arguments)]
async fn run_expert_route_reuse_probe(
    model_directory_name: &str,
    model_id: &str,
    input_token_count: usize,
    output_token_count: u16,
    request_id: u64,
    progress_log_label: &str,
    mlx_memory_limits: &MlxMemoryLimits,
) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let Some(model_directory) =
        crate::common::configured_model_directory_by_id(model_directory_name)
    else {
        eprintln!("[{progress_log_label}] status=skipped reason=checkpoint_not_found");
        return;
    };
    let temporary_log_directory =
        tempfile::tempdir().expect("the route-reuse probe should create a temporary log directory");
    let performance_attribution_log_path = temporary_log_directory
        .path()
        .join("performance-attribution.jsonl");
    eprintln!(
        "[{progress_log_label}] status=progress phase=model_load input_tokens={input_token_count} output_tokens={output_token_count} ETA_seconds=120"
    );
    let (mut qwen3_5_engine, end_of_sequence_token_ids) = create_attributed_engine(
        &model_directory,
        &performance_attribution_log_path,
        mlx_memory_limits,
        FIXED_PREFILL_CHUNCK_TOKENS,
    );
    load_engine_with_progress(&mut qwen3_5_engine, progress_log_label).await;
    let prompt_token_ids = prepare_reproduced_long_prompt_token_ids_for_model(
        &model_directory,
        model_id,
        input_token_count,
        output_token_count,
    )
    .expect("the route-reuse prompt should prepare at the exact requested length");
    let generated_token_ids = run_attributed_generation(
        &mut qwen3_5_engine,
        RequestId::new(request_id),
        &prompt_token_ids,
        progress_log_label,
        output_token_count,
        &end_of_sequence_token_ids,
    )
    .await;
    drop(qwen3_5_engine);

    assert!(
        generated_token_ids.len() > 1,
        "the route-reuse probe needs at least two decode tokens"
    );
    let attribution_report_documents =
        read_attribution_report_documents(&performance_attribution_log_path);
    let generation_report =
        generation_report_for_request(&attribution_report_documents, request_id);
    let predicted_expert_count =
        counter_amount(generation_report, "expert_route_predicted_expert_count");
    let matched_expert_count =
        counter_amount(generation_report, "expert_route_matched_expert_count");
    let completely_matched_layer_count = counter_amount(
        generation_report,
        "expert_route_completely_matched_layer_count",
    );
    let examined_layer_count =
        counter_amount(generation_report, "expert_route_examined_layer_count");
    let attributed_generated_token_count =
        counter_amount(generation_report, "generated_token_count");
    let native_expert_cache_source_read_bytes = counter_amount(
        generation_report,
        "native_expert_cache_successful_source_read_byte_count",
    );
    let native_expert_cache_snapshot_publication_count = counter_amount(
        generation_report,
        "native_expert_cache_snapshot_publication_count",
    );
    let native_expert_cache_payload_copy_bytes = counter_amount(
        generation_report,
        "native_expert_cache_payload_copy_byte_count",
    );
    let native_paged_expert_projection_graph_count = counter_amount(
        generation_report,
        "native_paged_expert_projection_graph_count",
    );
    let one_expert_page_hit_count =
        counter_amount(generation_report, "native_expert_cache_hit_count");
    let expert_miss_count = counter_amount(generation_report, "native_expert_cache_miss_count");
    let expert_eviction_count =
        counter_amount(generation_report, "native_expert_cache_eviction_count");
    assert_eq!(
        attributed_generated_token_count,
        generated_token_ids.len() as u64,
        "the engine report and token-in/token-out benchmark must count the same generated tokens"
    );
    assert_eq!(
        generation_report["configured_maximum_output_tokens"],
        output_token_count,
    );
    if generated_token_ids.len() < usize::from(output_token_count) {
        let final_generated_token_id = generated_token_ids
            .last()
            .copied()
            .expect("an early terminal request should have generated one token");
        assert!(
            end_of_sequence_token_ids.contains(&final_generated_token_id),
            "an early terminal request must end with a configured end-of-sequence token: generated_tokens={} final_generated_token_id={final_generated_token_id} end_of_sequence_token_ids={end_of_sequence_token_ids:?}",
            generated_token_ids.len(),
        );
    }
    assert!(predicted_expert_count > 0);
    assert!(matched_expert_count <= predicted_expert_count);
    assert!(examined_layer_count > 0);
    assert!(native_expert_cache_source_read_bytes > 0);
    assert!(native_expert_cache_snapshot_publication_count > 0);
    assert_eq!(native_expert_cache_payload_copy_bytes, 0);
    assert!(native_paged_expert_projection_graph_count > 0);
    let route_reuse_ratio = matched_expert_count as f64 / predicted_expert_count as f64;
    let per_layer_route_reuse = generation_report["previous_token_expert_route_reuse_by_layer"]
        .as_array()
        .expect("the route-reuse report should include bounded per-layer totals");
    assert!(!per_layer_route_reuse.is_empty());
    assert!(!generation_report.to_string().contains("\"expert_ids\""));
    eprintln!(
        "[{progress_log_label}] status=success generated_tokens={} predicted_experts={predicted_expert_count} matched_experts={matched_expert_count} route_reuse_ratio={route_reuse_ratio:.4} examined_layers={examined_layer_count} completely_matched_layers={completely_matched_layer_count} reported_layer_count={} native_cache_hits={one_expert_page_hit_count} native_cache_misses={expert_miss_count} native_cache_evictions={expert_eviction_count} native_cache_source_read_bytes={native_expert_cache_source_read_bytes} native_cache_snapshot_publications={native_expert_cache_snapshot_publication_count} native_cache_payload_copy_bytes={native_expert_cache_payload_copy_bytes} native_paged_expert_projection_graphs={native_paged_expert_projection_graph_count}",
        generated_token_ids.len(),
        per_layer_route_reuse.len(),
    );
}
