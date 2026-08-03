use std::time::Duration;

use astronomical_ipc_protocol::RequestId;
use tokio::time::timeout;

const TEST_TIMEOUT: Duration = Duration::from_secs(120);

use super::{
    DETERMINISTIC_PROMPT_TOKEN_ID, INPUT_TOKEN_COUNT, MODEL_ID, OUTPUT_TOKEN_COUNT,
    assert_attributed_memory_within_machine_cap, create_attributed_engine,
    generation_report_for_request, load_engine_with_progress, model_loading_report,
    operation_total_elapsed_nanoseconds, print_attribution_metadata,
    print_attribution_operation_table, qwen3_6_35b_a3b_optiq_4bit_model_directory,
    read_attribution_report_documents, run_attributed_generation,
};

#[tokio::test]
#[ignore = "loads Qwen3.6-35B-A3B-OptiQ-4bit for automatic cold and warm attribution"]
async fn should_measure_qwen3_6_35b_a3b_optiq_4bit_attribution_across_automatic_cold_and_warm_runs()
{
    timeout(
        TEST_TIMEOUT,
        run_qwen3_6_35b_a3b_optiq_4bit_attribution_benchmark(),
    )
    .await
    .expect("the attribution benchmark must finish within its 120-second timeout");
}

async fn run_qwen3_6_35b_a3b_optiq_4bit_attribution_benchmark() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let Some(model_directory) = qwen3_6_35b_a3b_optiq_4bit_model_directory() else {
        eprintln!(
            "[performance-attribution] status=skipped reason={MODEL_ID}_checkpoint_not_found"
        );
        return;
    };
    let prompt_token_ids = vec![DETERMINISTIC_PROMPT_TOKEN_ID; INPUT_TOKEN_COUNT];
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
    assert!(
        operation_total_elapsed_nanoseconds(
            cold_generation_report,
            "generated_token_item_synchronization_wait",
        ) > 0
    );
    eprintln!("[performance-attribution 3/3] status=success");
}
