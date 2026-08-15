use astronomical_ipc_protocol::{
    WorkerPromptProcessingChunkCandidateMeasurementSummary,
    WorkerPromptProcessingChunkMeasurementSource, WorkerPromptProcessingChunkOptimizationContext,
    WorkerPromptProcessingChunkOptimizationOutcome, WorkerPromptProcessingChunkSelectionReason,
};
use astronomical_supervisor::build_application;
use axum::{
    body::{Body, to_bytes},
    http::Request,
};
use tower::ServiceExt;

use super::observatory_contracts::{ContractScriptedExecutor, ready_health_snapshot_with_model};

#[tokio::test]
async fn should_expose_structured_prompt_processing_chunk_outcome() {
    let mut health_snapshot = ready_health_snapshot_with_model();
    health_snapshot
        .recent_prompt_processing_chunk_optimization_outcomes
        .push(WorkerPromptProcessingChunkOptimizationOutcome {
            selected_candidate_chunk_size_tokens: 4_096,
            processed_prompt_token_count: 4_096,
            forward_elapsed_millis: 720,
            was_reduced_by_memory_capacity: false,
            selection_reason:
                WorkerPromptProcessingChunkSelectionReason::MinimizeProjectedRemainingPromptLatency,
            measurement_context: WorkerPromptProcessingChunkOptimizationContext {
                chunk_start_token_position: 32_768,
                position_range_start_token_position: 32_768,
                position_range_end_token_position_exclusive: 65_536,
                has_restored_prefix: true,
                is_first_chunk_after_restore: false,
                has_visual_embeddings: false,
                is_mtp_active: false,
                are_sparse_experts_paged: false,
                is_prompt_cache_capture_eligible: true,
                has_prior_capacity_reduction: false,
            },
            all_candidates_have_measurements: true,
            candidate_measurement_summaries: vec![
                WorkerPromptProcessingChunkCandidateMeasurementSummary {
                    candidate_chunk_size_tokens: 4_096,
                    measurement_source:
                        WorkerPromptProcessingChunkMeasurementSource::CurrentPositionRange,
                    measurement_count: 5,
                    average_processed_prompt_token_count: 4_096,
                    average_forward_elapsed_millis: 710,
                    selections_since_last_measurement: Some(0),
                },
            ],
        });
    let application = build_application(ContractScriptedExecutor::ready(health_snapshot));

    let response = application
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .expect("the status request should be valid"),
        )
        .await
        .expect("the application should return a status response");
    let response_body = to_bytes(response.into_body(), 32 * 1024)
        .await
        .expect("the status body should be readable");
    let status_document: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the status body should contain JSON");

    let optimizer_document = &status_document["prompt_processing_chunk_size_optimizer"];
    assert_eq!(optimizer_document["mode"], "unavailable");
    assert_eq!(
        optimizer_document["latest_chunk_outcome"]["selection"]["selected_candidate_chunk_size_tokens"],
        4_096
    );
    assert_eq!(
        optimizer_document["latest_chunk_outcome"]["processed_prompt_token_count"],
        4_096
    );
    assert_eq!(
        optimizer_document["latest_chunk_outcome"]["measurement_context"]["position_range_end_token_position_exclusive"],
        65_536
    );
    assert_eq!(
        optimizer_document["latest_chunk_outcome"]["candidate_measurement_summaries"][0]["measurement_source"],
        "current_position_range"
    );
    assert_eq!(
        optimizer_document["recent_chunk_outcomes"]
            .as_array()
            .map(Vec::len),
        Some(1)
    );
    assert!(status_document.get("prefill_optimizer").is_none());
}
