use astronomical_model_serving::{
    ExpertWeightPage, PerformanceAttribution, PerformanceOperation, forward_paged_routed_swiglu,
    load_laguna_expert_page, sorted_expert_weighted_sum_kernel,
};

use super::page_artifact::{filled, paging_plan, test_runtime, write_sparse_artifact};

#[tokio::test]
async fn should_stream_complete_and_routed_pages_with_matching_gathered_swiglu() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = tempfile::tempdir().expect("page test directory");
    write_sparse_artifact(model_directory.path(), false);
    let (_artifact, plan) = paging_plan(model_directory.path());
    let runtime = test_runtime();
    let mut performance_attribution = PerformanceAttribution::enabled();
    let sparse_layer = &plan.sparse_layers()[0];
    let complete_page = load_laguna_expert_page(
        &runtime,
        sparse_layer,
        &[0, 1],
        &mut performance_attribution,
    )
    .expect("a complete-layer page should stream");
    let routed_page =
        load_laguna_expert_page(&runtime, sparse_layer, &[0], &mut performance_attribution)
            .expect("a routed page should stream");
    assert_eq!(
        complete_page.resident_payload_byte_count(),
        sparse_layer
            .complete_layer_payload_byte_count()
            .expect("complete bytes")
    );
    assert_eq!(
        routed_page.resident_payload_byte_count(),
        sparse_layer
            .routed_page_payload_byte_count()
            .expect("routed bytes")
    );
    assert!(
        performance_attribution
            .operation_measurement(PerformanceOperation::RustExpertStreamingLayerPreparation)
            .is_some()
    );

    let hidden_states = filled(&runtime, &[1, 2, 4], 0.2);
    let complete_indices = runtime
        .array_from_u32(&[0, 0], &[1, 2])
        .expect("complete indices");
    let routed_indices = runtime
        .array_from_u32(&[0, 0], &[1, 2])
        .expect("routed indices");
    let scores = filled(&runtime, &[1, 2, 1], 1.0);
    let kernel = sorted_expert_weighted_sum_kernel().expect("reduction kernel");
    let complete_output = forward_paged_routed_swiglu(
        &runtime,
        &hidden_states,
        &complete_page,
        &complete_indices,
        &scores,
        false,
        Some(&kernel),
        &mut performance_attribution,
    )
    .expect("complete-page SwiGLU should execute");
    let routed_output = forward_paged_routed_swiglu(
        &runtime,
        &hidden_states,
        &routed_page,
        &routed_indices,
        &scores,
        false,
        Some(&kernel),
        &mut performance_attribution,
    )
    .expect("routed-page SwiGLU should execute");
    let complete_values = complete_output.to_vec_f32().expect("complete host values");
    let routed_values = routed_output.to_vec_f32().expect("routed host values");
    assert_eq!(complete_values.len(), routed_values.len());
    for (complete_value, routed_value) in complete_values.iter().zip(routed_values.iter()) {
        assert!(
            (complete_value - routed_value).abs() < 1e-5,
            "complete and routed pages must agree for expert 0"
        );
    }
}

#[tokio::test]
async fn should_concatenate_per_expert_shards_into_one_compact_page() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = tempfile::tempdir().expect("per-expert page test directory");
    write_sparse_artifact(model_directory.path(), true);
    let (_artifact, plan) = paging_plan(model_directory.path());
    let runtime = test_runtime();
    let mut performance_attribution = PerformanceAttribution::enabled();
    let complete_page = load_laguna_expert_page(
        &runtime,
        &plan.sparse_layers()[0],
        &[0, 1],
        &mut performance_attribution,
    )
    .expect("per-expert complete page should concatenate shards");
    assert_eq!(
        complete_page.resident_payload_byte_count(),
        plan.sparse_layers()[0]
            .complete_layer_payload_byte_count()
            .expect("complete bytes")
    );
}

#[tokio::test]
async fn should_fail_closed_when_a_page_source_shard_disappears() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = tempfile::tempdir().expect("missing-shard page test directory");
    write_sparse_artifact(model_directory.path(), false);
    let (_artifact, plan) = paging_plan(model_directory.path());
    std::fs::remove_file(
        model_directory
            .path()
            .join("model-00001-of-00002.safetensors"),
    )
    .expect("the first shard should be removable");
    let runtime = test_runtime();
    let mut performance_attribution = PerformanceAttribution::disabled();
    let rejection = load_laguna_expert_page(
        &runtime,
        &plan.sparse_layers()[0],
        &[0, 1],
        &mut performance_attribution,
    );
    assert!(rejection.is_err());
}
