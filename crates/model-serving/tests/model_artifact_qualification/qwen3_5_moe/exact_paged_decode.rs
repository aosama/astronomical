use std::time::Duration;

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::Qwen3_5ArtifactValidator;
use serde_json::Value;
use tokio::time::timeout;

use super::performance_attribution::{
    counter_amount, create_attributed_engine, generation_report_for_request,
    load_engine_with_progress, read_attribution_report_documents, run_attributed_generation,
};
use super::speculative_prefill_qualification_support::prepare_romeo_and_juliet_three_paragraph_summary_prompt;

const PROMPT_TOKEN_COUNT: usize = 48;
const OUTPUT_TOKEN_COUNT: u16 = 32;
const MODEL_ID: &str = "Ornith-1.0-397B-mlx-4bit";
const COLD_REQUEST_ID: RequestId = RequestId::new(96_100);
const WARM_REQUEST_ID: RequestId = RequestId::new(96_101);

#[tokio::test]
#[ignore = "loads Ornith and qualifies exact cold plus warm demand-paged decode"]
async fn should_preserve_exact_cold_and_warm_generation_without_whole_token_replay() {
    timeout(Duration::from_secs(115), async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let model_directory =
            crate::common::configured_model_artifact_directory_by_id(MODEL_ID);
        let validated_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&model_directory, u32::from(OUTPUT_TOKEN_COUNT))
            .expect("the configured Ornith artifact should validate");
        assert_eq!(validated_artifact.model_id(), MODEL_ID);
        let model_id = validated_artifact.model_id().to_owned();
        let sparse_layer_count = u64::from(validated_artifact.config().layer_count());
        let romeo_and_juliet_prompt = prepare_romeo_and_juliet_three_paragraph_summary_prompt(
            &model_directory,
            &model_id,
            COLD_REQUEST_ID,
            PROMPT_TOKEN_COUNT,
            OUTPUT_TOKEN_COUNT,
        );
        let temporary_attribution_directory = tempfile::tempdir()
            .expect("the exact paged decode journey should create an attribution directory");
        let attribution_log_path = temporary_attribution_directory
            .path()
            .join("exact-paged-decode.jsonl");
        let mlx_memory_limits =
            crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
        let configured_mlx_memory_ceiling_bytes = u64::try_from(
            mlx_memory_limits.active_memory_limit_bytes(),
        )
        .expect("the configured MLX memory ceiling should fit u64");
        let (mut qwen3_5_engine, end_of_sequence_token_ids) = create_attributed_engine(
            &model_directory,
            &attribution_log_path,
            &mlx_memory_limits,
            PROMPT_TOKEN_COUNT as u32,
        );

        eprintln!("[exact-paged-decode 0/4] status=progress phase=model_load ETA_seconds=90");
        load_engine_with_progress(&mut qwen3_5_engine, "exact_paged_decode_model_load").await;
        eprintln!("[exact-paged-decode 1/4] status=progress phase=cold_generation");
        let cold_generated_token_ids = run_attributed_generation(
            &mut qwen3_5_engine,
            COLD_REQUEST_ID,
            &romeo_and_juliet_prompt.prompt_token_ids,
            "exact_paged_decode_cold",
            OUTPUT_TOKEN_COUNT,
            &end_of_sequence_token_ids,
        )
        .await;
        eprintln!("[exact-paged-decode 2/4] status=progress phase=warm_generation");
        let warm_generated_token_ids = run_attributed_generation(
            &mut qwen3_5_engine,
            WARM_REQUEST_ID,
            &romeo_and_juliet_prompt.prompt_token_ids,
            "exact_paged_decode_warm",
            OUTPUT_TOKEN_COUNT,
            &end_of_sequence_token_ids,
        )
        .await;
        assert_eq!(
            warm_generated_token_ids, cold_generated_token_ids,
            "cold and warm demand paging must preserve the exact greedy continuation"
        );

        let attribution_report_documents =
            read_attribution_report_documents(&attribution_log_path);
        let cold_generation_report = generation_report_for_request(
            &attribution_report_documents,
            COLD_REQUEST_ID.value(),
        );
        let warm_generation_report = generation_report_for_request(
            &attribution_report_documents,
            WARM_REQUEST_ID.value(),
        );
        assert_exact_layer_preparation_count(
            cold_generation_report,
            sparse_layer_count,
            cold_generated_token_ids.len(),
        );
        assert_exact_layer_preparation_count(
            warm_generation_report,
            sparse_layer_count,
            warm_generated_token_ids.len(),
        );
        let cold_disk_page_load_count = counter_amount(
            cold_generation_report,
            "positional_file_read_call_count",
        );
        let warm_disk_page_load_count = counter_amount(
            warm_generation_report,
            "positional_file_read_call_count",
        );
        let cold_memory_evidence = assert_memory_within_policy(
            cold_generation_report,
            configured_mlx_memory_ceiling_bytes,
        );
        let warm_memory_evidence = assert_memory_within_policy(
            warm_generation_report,
            configured_mlx_memory_ceiling_bytes,
        );
        eprintln!(
            "[exact-paged-decode 3/4] status=progress cold_disk_page_load_count={cold_disk_page_load_count} warm_disk_page_load_count={warm_disk_page_load_count} cold_active_plus_allocator_bytes={} warm_active_plus_allocator_bytes={} observed_peak_bytes={} allowed_peak_bytes={}",
            cold_memory_evidence.active_plus_allocator_bytes,
            warm_memory_evidence.active_plus_allocator_bytes,
            cold_memory_evidence
                .peak_bytes
                .max(warm_memory_evidence.peak_bytes),
            cold_memory_evidence.allowed_peak_bytes,
        );
        assert!(
            warm_disk_page_load_count <= cold_disk_page_load_count,
            "retained per-layer decode pages must not make the repeated Romeo and Juliet journey read more pages"
        );
        eprintln!("[exact-paged-decode 4/4] status=success");
    })
    .await
    .expect("the exact paged decode qualification must finish within 115 seconds");
}

struct MemoryPolicyEvidence {
    active_plus_allocator_bytes: u64,
    peak_bytes: u64,
    allowed_peak_bytes: u64,
}

fn assert_memory_within_policy(
    generation_report: &Value,
    configured_mlx_memory_ceiling_bytes: u64,
) -> MemoryPolicyEvidence {
    let active_memory_bytes = generation_report["mlx_active_memory_bytes"]
        .as_u64()
        .expect("successful generation should report MLX active memory");
    let allocator_cache_memory_bytes = generation_report["mlx_allocator_cache_memory_bytes"]
        .as_u64()
        .expect("successful generation should report MLX allocator-cache memory");
    let active_plus_allocator_bytes = active_memory_bytes
        .checked_add(allocator_cache_memory_bytes)
        .expect("active and allocator-cache memory should fit u64");
    assert!(
        active_plus_allocator_bytes <= configured_mlx_memory_ceiling_bytes,
        "stable MLX residency {active_plus_allocator_bytes} exceeded the configured ceiling {configured_mlx_memory_ceiling_bytes}"
    );
    let peak_bytes = generation_report["mlx_peak_memory_bytes"]
        .as_u64()
        .expect("successful generation should report MLX peak memory");
    let allowed_peak_bytes = configured_mlx_memory_ceiling_bytes
        .saturating_add(configured_mlx_memory_ceiling_bytes / 100);
    assert!(
        peak_bytes <= allowed_peak_bytes,
        "MLX peak memory {peak_bytes} exceeded the one-percent transient policy limit {allowed_peak_bytes}"
    );
    MemoryPolicyEvidence {
        active_plus_allocator_bytes,
        peak_bytes,
        allowed_peak_bytes,
    }
}

fn assert_exact_layer_preparation_count(
    generation_report: &Value,
    sparse_layer_count: u64,
    generated_token_count: usize,
) {
    let generated_token_count =
        u64::try_from(generated_token_count).expect("the generated token count should fit u64");
    let prefill_chunck_count = counter_amount(generation_report, "prefill_chunck_count");
    assert_eq!(
        operation_occurrence_count(generation_report, "rust_expert_streaming_layer_preparation",),
        sparse_layer_count * (prefill_chunck_count + generated_token_count),
        "every prompt prefill chunck and generated target token must follow one exact route preparation per sparse layer"
    );
}

fn operation_occurrence_count(report: &Value, operation_identifier: &str) -> u64 {
    report["operations"]
        .as_array()
        .and_then(|operation_reports| {
            operation_reports.iter().find_map(|operation_report| {
                (operation_report["operation"] == operation_identifier)
                    .then(|| operation_report["occurrence_count"].as_u64())
                    .flatten()
            })
        })
        .unwrap_or(0)
}
