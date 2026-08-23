//! Attributes model loading plus equivalent cold and warm SSD-streamed generations.

use std::time::Duration;

use astronomical_ipc_protocol::{ExpertMemoryMode, RequestId};
use tokio::time::timeout;

use super::super::performance_attribution::{
    INPUT_TOKEN_COUNT, OUTPUT_TOKEN_COUNT, assert_attributed_memory_within_machine_cap,
    counter_amount, create_attributed_engine, generation_report_for_request,
    load_engine_with_progress, model_loading_report, operation_total_elapsed_nanoseconds,
    print_attribution_metadata, print_attribution_operation_table,
    print_expert_streaming_source_summary_table, read_attribution_report_documents,
    run_attributed_generation,
};

const TEST_TIMEOUT: Duration = Duration::from_secs(120);
const MODEL_ID: &str = "Ornith-1.5-35B-A3B-oQ8e-mtp";

#[tokio::test]
#[ignore = "loads the configured Ornith model for automatic cold and warm attribution"]
async fn should_measure_model_ssd_streaming_attribution_across_automatic_cold_and_warm_runs() {
    timeout(
        TEST_TIMEOUT,
        run_model_ssd_streaming_attribution_benchmark(),
    )
    .await
    .expect("the attribution benchmark must finish within its 120-second timeout");
}

async fn run_model_ssd_streaming_attribution_benchmark() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let Some(model_directory) = crate::common::configured_model_directory_by_id(MODEL_ID) else {
        eprintln!(
            "[performance-attribution] status=skipped reason={MODEL_ID}_checkpoint_not_found"
        );
        return;
    };
    let representative_prompt = super::super::speculative_prefill_qualification_support::prepare_romeo_and_juliet_three_paragraph_summary_prompt(
        &model_directory,
        MODEL_ID,
        RequestId::new(8_999),
        INPUT_TOKEN_COUNT,
        OUTPUT_TOKEN_COUNT,
    );
    let prompt_token_ids = representative_prompt.prompt_token_ids;
    let temporary_log_directory =
        tempfile::tempdir().expect("the benchmark should create a temporary log directory");
    let performance_attribution_log_path = temporary_log_directory
        .path()
        .join("performance-attribution.jsonl");
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;

    eprintln!(
        "[performance-attribution 0/3] status=progress phase=automatic_model_load input_tokens={INPUT_TOKEN_COUNT} output_tokens={OUTPUT_TOKEN_COUNT} ETA_seconds=120"
    );
    let (mut automatic_engine, end_of_sequence_token_ids) = create_attributed_engine(
        &model_directory,
        &performance_attribution_log_path,
        &mlx_memory_limits,
        1_024,
    );
    load_engine_with_progress(&mut automatic_engine, "automatic_model_load").await;
    let expert_memory_mode = automatic_engine
        .expert_memory_mode_for_tests()
        .await
        .expect("the loaded model should expose its expert mode");
    let expert_statistics = automatic_engine
        .expert_weight_memory_cache_statistics_for_tests()
        .await
        .expect("the loaded model should expose expert statistics");
    match expert_memory_mode {
        Some(ExpertMemoryMode::Resident) => {}
        Some(ExpertMemoryMode::Hybrid) => {
            assert!(
                expert_statistics.complete_layer_count > 0,
                "hybrid readiness must install complete layers before the first generate"
            );
        }
        other_mode => {
            panic!("SSD-streamed Ornith must be resident or hybrid after load, got {other_mode:?}")
        }
    }
    let cold_generated_token_ids = run_attributed_generation(
        &mut automatic_engine,
        RequestId::new(9_000),
        &prompt_token_ids,
        "automatic_cold",
        OUTPUT_TOKEN_COUNT,
        &end_of_sequence_token_ids,
    )
    .await;

    eprintln!(
        "[performance-attribution 1/3] status=progress phase=automatic_warm output_tokens={OUTPUT_TOKEN_COUNT} ETA_seconds=80"
    );
    let warm_generated_token_ids = run_attributed_generation(
        &mut automatic_engine,
        RequestId::new(9_001),
        &prompt_token_ids,
        "automatic_warm",
        OUTPUT_TOKEN_COUNT,
        &end_of_sequence_token_ids,
    )
    .await;
    drop(automatic_engine);

    assert_eq!(cold_generated_token_ids, warm_generated_token_ids);

    let attribution_report_documents =
        read_attribution_report_documents(&performance_attribution_log_path);
    let automatic_model_loading_report = model_loading_report(&attribution_report_documents);
    let cold_generation_report =
        generation_report_for_request(&attribution_report_documents, 9_000);
    let warm_generation_report =
        generation_report_for_request(&attribution_report_documents, 9_001);
    for (phase_name, performance_attribution_report) in [
        ("automatic_model_load", automatic_model_loading_report),
        ("automatic_cold", cold_generation_report),
        ("automatic_warm", warm_generation_report),
    ] {
        assert_attributed_memory_within_machine_cap(
            phase_name,
            performance_attribution_report,
            mlx_memory_limits.active_memory_limit_bytes(),
        );
        print_attribution_metadata(phase_name, performance_attribution_report);
        print_attribution_operation_table(phase_name, performance_attribution_report);
        print_expert_streaming_source_summary_table(phase_name, performance_attribution_report);
    }

    assert!(
        operation_total_elapsed_nanoseconds(
            automatic_model_loading_report,
            "model_safetensors_mapping",
        ) > 0
    );
    assert!(
        operation_total_elapsed_nanoseconds(automatic_model_loading_report, "model_tensor_binding")
            > 0
    );
    assert!(
        operation_total_elapsed_nanoseconds(
            automatic_model_loading_report,
            "expert_pager_plan_construction",
        ) > 0
    );
    assert!(
        operation_total_elapsed_nanoseconds(cold_generation_report, "paged_moe_graph_construction",)
            > 0
    );
    if expert_memory_mode == Some(ExpertMemoryMode::Hybrid) {
        assert!(
            counter_amount(
                automatic_model_loading_report,
                "mandatory_prefill_complete_layer_promoted_payload_byte_count",
            ) > 0,
            "hybrid load must pack complete layers before the worker is ready"
        );
        assert!(
            counter_amount(
                cold_generation_report,
                "avoided_complete_layer_expert_source_payload_bytes",
            ) > 0,
            "the first generate must run packed complete layers from RAM"
        );
    }
    assert!(
        operation_occurrence_count(
            cold_generation_report,
            "rust_expert_streaming_layer_preparation",
        ) > 0
    );
    assert!(counter_amount(cold_generation_report, "positional_file_read_byte_count",) > 0);
    assert_source_plan_payload_matches_streaming_counters(cold_generation_report);
    assert_source_plan_payload_matches_streaming_counters(warm_generation_report);
    assert!(
        counter_amount(
            cold_generation_report,
            "rust_streamed_expert_projection_graph_count",
        ) > 0
    );
    assert!(
        operation_total_elapsed_nanoseconds(
            cold_generation_report,
            "generated_token_item_synchronization_wait",
        ) > 0
    );
    eprintln!("[performance-attribution 3/3] status=success");
}

fn assert_source_plan_payload_matches_streaming_counters(report: &serde_json::Value) {
    let source_summaries = report["expert_streaming_source_summaries"]
        .as_array()
        .expect("SSD-streamed generation attribution should include source summaries");
    assert!(
        source_summaries
            .iter()
            .any(|source_summary| source_summary["phase"] == "prefill"),
        "the representative prompt should expose per-layer prefill source summaries"
    );
    let attributed_source_payload_bytes =
        source_summaries
            .iter()
            .fold(0_u64, |total, source_summary| {
                total.saturating_add(source_summary["payload_byte_count"].as_u64().unwrap_or(0))
            });
    let mandatory_source_payload_bytes =
        counter_amount(report, "mandatory_prefill_expert_source_payload_bytes").saturating_add(
            counter_amount(report, "mandatory_decode_expert_source_payload_bytes"),
        );
    assert_eq!(
        attributed_source_payload_bytes,
        counter_amount(report, "rust_expert_streaming_payload_byte_count"),
        "per-layer source summaries must reconcile with the total Rust-streamed payload"
    );
    assert_eq!(
        attributed_source_payload_bytes, mandatory_source_payload_bytes,
        "per-layer source summaries must reconcile with phase-specific mandatory source payload"
    );
}

fn operation_occurrence_count(report: &serde_json::Value, operation_identifier: &str) -> u64 {
    report["operations"]
        .as_array()
        .and_then(|operations| {
            operations.iter().find_map(|operation| {
                (operation["operation"] == operation_identifier)
                    .then(|| operation["occurrence_count"].as_u64())
                    .flatten()
            })
        })
        .unwrap_or(0)
}
