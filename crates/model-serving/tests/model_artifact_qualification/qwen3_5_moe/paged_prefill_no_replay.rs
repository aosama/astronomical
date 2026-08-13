//! Real Fable acceptance journey for fixed-chunk whole-layer paged prefill.

use std::time::{Duration, Instant};

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::Qwen3_5ArtifactValidator;
use astronomical_runtime_integration::MlxMemoryLimits;
use tokio::time::timeout;

use super::performance_attribution::{
    counter_amount, create_attributed_engine, generation_report_for_request,
    load_engine_with_progress, read_attribution_report_documents, run_attributed_generation,
};
use super::speculative_prefill_qualification_support::prepare_romeo_and_juliet_three_paragraph_summary_prompt;

const MODEL_ID: &str = "Qwen3.6-35B-A3B-Fable-Holo3.1-Qwopus-KAT-Coder-C-qx86-hi-mlx";
const MEMORY_CEILING_BYTES: usize = 39_000_000_000;
const PROMPT_TOKEN_COUNT: usize = 4_096;
const FIXED_PREFILL_CHUNCK_TOKENS: u32 = 2_048;
const OUTPUT_TOKEN_COUNT: u16 = 1;
const REQUEST_ID: RequestId = RequestId::new(96_204);
const PRODUCT_PERFORMANCE_BUDGET: Duration = Duration::from_secs(60);

#[tokio::test]
#[ignore = "loads Fable under 39 GB and proves two fixed 2048-token prefill chunks finish under 60 seconds without whole-forward replay"]
async fn should_process_each_fixed_paged_prefill_chunk_without_whole_forward_replay() {
    timeout(Duration::from_secs(120), async {
        super::automatic_residency_support::initialize_automatic_residency_tracing();
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let model_directory =
            crate::common::configured_model_artifact_directory_by_id(MODEL_ID);
        let validated_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&model_directory, u32::from(OUTPUT_TOKEN_COUNT))
            .expect("the configured Fable artifact should validate");
        let sparse_layer_count = u64::from(validated_artifact.config().layer_count());
        let prompt = prepare_romeo_and_juliet_three_paragraph_summary_prompt(
            &model_directory,
            MODEL_ID,
            REQUEST_ID,
            PROMPT_TOKEN_COUNT,
            OUTPUT_TOKEN_COUNT,
        );
        let temporary_attribution_directory = tempfile::tempdir()
            .expect("the paged-prefill journey should create an attribution directory");
        let attribution_log_path = temporary_attribution_directory
            .path()
            .join("paged-prefill-no-replay.jsonl");
        let mlx_memory_limits =
            MlxMemoryLimits::new(MEMORY_CEILING_BYTES, MEMORY_CEILING_BYTES)
                .expect("the fixed qualification ceiling should be valid");
        let (mut engine, end_of_sequence_token_ids) = create_attributed_engine(
            &model_directory,
            &attribution_log_path,
            &mlx_memory_limits,
            FIXED_PREFILL_CHUNCK_TOKENS,
        );

        eprintln!(
            "[paged-prefill-no-replay 0/3] status=progress phase=model_load ETA_seconds=90"
        );
        load_engine_with_progress(&mut engine, "paged_prefill_no_replay_model_load").await;
        eprintln!(
            "[paged-prefill-no-replay 1/3] status=progress phase=generation prompt_tokens={PROMPT_TOKEN_COUNT} fixed_prefill_tokens={FIXED_PREFILL_CHUNCK_TOKENS}"
        );
        let generation_started_at = Instant::now();
        let generated_token_ids = run_attributed_generation(
            &mut engine,
            REQUEST_ID,
            &prompt.prompt_token_ids,
            "paged_prefill_no_replay_generation",
            OUTPUT_TOKEN_COUNT,
            &end_of_sequence_token_ids,
        )
        .await;
        let generation_elapsed = generation_started_at.elapsed();

        let reports = read_attribution_report_documents(&attribution_log_path);
        let generation_report = generation_report_for_request(&reports, REQUEST_ID.value());
        let prefill_chunck_count = counter_amount(generation_report, "prefill_chunck_count");
        assert_eq!(
            prefill_chunck_count, 2,
            "4,096 prompt tokens with fixed 2,048-token chunks must use two prefill chunks"
        );
        let route_preparation_count = operation_occurrence_count(
            generation_report,
            "rust_expert_streaming_layer_preparation",
        );
        let ordinary_forward_count = prefill_chunck_count
            .saturating_add(u64::try_from(generated_token_ids.len()).unwrap_or(u64::MAX));
        // Complete-layer streaming still prepares one snapshot per nonresident layer
        // per ordinary forward. Bound preparations so whole-forward replay cannot hide.
        let baseline_route_preparation_count = sparse_layer_count
            .saturating_mul(ordinary_forward_count.saturating_add(1));
        let maximum_route_preparation_count = baseline_route_preparation_count
            .saturating_add(sparse_layer_count.saturating_sub(1));
        eprintln!(
            "[paged-prefill-no-replay 2/3] status=progress prefill_chuncks={prefill_chunck_count} route_preparations={route_preparation_count} baseline_route_preparations={baseline_route_preparation_count} generation_elapsed_seconds={:.3} disk_page_loads={} source_read_bytes={}",
            generation_elapsed.as_secs_f64(),
            counter_amount(generation_report, "positional_file_read_call_count"),
            counter_amount(generation_report, "positional_file_read_byte_count"),
        );
        assert!(
            route_preparation_count <= maximum_route_preparation_count,
            "paged prefill replayed whole forwards: route_preparations={route_preparation_count}, allowed={maximum_route_preparation_count}"
        );
        assert!(
            generation_elapsed <= PRODUCT_PERFORMANCE_BUDGET,
            "Fable whole-layer prefill must finish under {} seconds, observed {:.3} seconds",
            PRODUCT_PERFORMANCE_BUDGET.as_secs(),
            generation_elapsed.as_secs_f64()
        );
        eprintln!("[paged-prefill-no-replay 3/3] status=success");
    })
    .await
    .expect("the paged-prefill no-replay qualification must finish within 120 seconds");
}

fn operation_occurrence_count(report: &serde_json::Value, operation_identifier: &str) -> u64 {
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
