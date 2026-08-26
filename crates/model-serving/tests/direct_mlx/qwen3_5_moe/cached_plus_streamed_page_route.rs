use std::time::Duration;

use astronomical_model_serving::{
    ExpertPageRoutePartition, QuantizedExpertPageManifest, Qwen3_5MoECachedPlusStreamedPageRoute,
};
use astronomical_runtime_integration::{MlxArray, MlxMemoryLimits, MlxRuntime};
use tokio::time::timeout;

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

const CACHED_PLUS_STREAMED_PAGE_ROUTE_TEST_TIMEOUT: Duration = Duration::from_secs(30);

#[tokio::test]
async fn should_preserve_weighted_expert_output_across_compact_retained_and_missing_routes() {
    timeout(
        CACHED_PLUS_STREAMED_PAGE_ROUTE_TEST_TIMEOUT,
        verify_compact_cached_plus_streamed_page_route_parity(),
    )
    .await
    .expect("the cached-plus-streamed direct MLX contract must finish within 30 seconds");
}

#[tokio::test]
async fn should_reject_a_cached_plus_streamed_route_with_an_empty_assignment_side() {
    timeout(CACHED_PLUS_STREAMED_PAGE_ROUTE_TEST_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let runtime = direct_mlx_runtime();
        let retained_page_manifest = page_manifest(&[1, 3], &[u32::MAX, 0, u32::MAX, 1]);
        let selected_indices = runtime
            .array_from_u32(&[3, 1], &[1, 1, 2])
            .expect("selected expert indices should be valid");
        let selected_scores = runtime
            .array_from_f32(&[0.4, 0.6], &[1, 1, 2])
            .expect("selected expert scores should be valid");
        let route_partition = retained_page_manifest.partition_route_assignments(&[3, 1]);

        let route_error = Qwen3_5MoECachedPlusStreamedPageRoute::build(
            &runtime,
            &selected_indices,
            &selected_scores,
            &route_partition,
            &retained_page_manifest,
            &retained_page_manifest,
        )
        .err()
        .expect("an empty missing side should be rejected before MLX indexing");

        assert!(
            route_error.to_string().contains("non-empty route sides"),
            "the typed boundary should explain the invalid empty side: {route_error}"
        );
    })
    .await
    .expect("the empty-side contract must finish within 30 seconds");
}

#[tokio::test]
async fn should_reject_an_expert_absent_from_its_declared_cached_plus_streamed_page() {
    timeout(CACHED_PLUS_STREAMED_PAGE_ROUTE_TEST_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let runtime = direct_mlx_runtime();
        let retained_page_manifest = page_manifest(&[1], &[u32::MAX, 0, u32::MAX, u32::MAX]);
        let missing_page_manifest = page_manifest(&[0], &[0, u32::MAX, u32::MAX, u32::MAX]);
        let selected_indices = runtime
            .array_from_u32(&[1, 3], &[1, 1, 2])
            .expect("selected expert indices should be valid");
        let selected_scores = runtime
            .array_from_f32(&[0.4, 0.6], &[1, 1, 2])
            .expect("selected expert scores should be valid");
        let malformed_partition = ExpertPageRoutePartition {
            retained_assignment_positions: vec![0],
            retained_expert_ids: vec![1],
            missing_assignment_positions: vec![1],
            missing_expert_ids: vec![3],
        };

        let route_error = Qwen3_5MoECachedPlusStreamedPageRoute::build(
            &runtime,
            &selected_indices,
            &selected_scores,
            &malformed_partition,
            &retained_page_manifest,
            &missing_page_manifest,
        )
        .err()
        .expect("an expert absent from its declared page should be rejected");

        assert!(
            route_error.to_string().contains("routed expert is absent"),
            "the typed boundary should explain the absent expert: {route_error}"
        );
    })
    .await
    .expect("the absent-expert contract must finish within 30 seconds");
}

async fn verify_compact_cached_plus_streamed_page_route_parity() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = direct_mlx_runtime();

    let retained_page_manifest = page_manifest(&[1, 3], &[u32::MAX, 0, u32::MAX, 1, u32::MAX]);
    let missing_page_manifest = page_manifest(&[0, 4], &[0, u32::MAX, u32::MAX, u32::MAX, 1]);
    let selected_expert_ids = [3, 0, 1, 4];
    let route_partition = retained_page_manifest.partition_route_assignments(&selected_expert_ids);
    let selected_indices = runtime
        .array_from_u32(&[3, 0, 1, 4], &[1, 1, 4])
        .expect("the selected expert indices should be valid");
    let selected_scores = runtime
        .array_from_f32(&[0.1, 0.2, 0.3, 0.4], &[1, 1, 4])
        .expect("the selected expert scores should be valid");

    eprintln!("[cached-plus-streamed-page-route] status=progress phase=build_compact_route_arrays");
    let cached_plus_streamed_page_route = Qwen3_5MoECachedPlusStreamedPageRoute::build(
        &runtime,
        &selected_indices,
        &selected_scores,
        &route_partition,
        &retained_page_manifest,
        &missing_page_manifest,
    )
    .expect(
        "the retained and missing assignments should build one compact cached-plus-streamed route",
    );

    assert_eq!(
        cached_plus_streamed_page_route
            .retained_page_slot_indices
            .shape(),
        vec![2]
    );
    assert_eq!(
        cached_plus_streamed_page_route.retained_scores.shape(),
        vec![2]
    );
    assert_eq!(
        cached_plus_streamed_page_route
            .missing_page_slot_indices
            .shape(),
        vec![2]
    );
    assert_eq!(
        cached_plus_streamed_page_route.missing_scores.shape(),
        vec![2]
    );
    assert_eq!(
        cached_plus_streamed_page_route
            .retained_page_slot_indices
            .to_vec_u32()
            .expect("retained page slots should evaluate"),
        vec![1, 0]
    );
    assert_eq!(
        cached_plus_streamed_page_route
            .missing_page_slot_indices
            .to_vec_u32()
            .expect("missing page slots should evaluate"),
        vec![0, 1]
    );
    assert_f32_close(
        &cached_plus_streamed_page_route
            .retained_scores
            .to_vec_f32()
            .expect("retained scores should evaluate"),
        &[0.1, 0.3],
    );
    assert_f32_close(
        &cached_plus_streamed_page_route
            .missing_scores
            .to_vec_f32()
            .expect("missing scores should evaluate"),
        &[0.2, 0.4],
    );

    let retained_page_expert_outputs = runtime
        .array_from_f32(&[10.0, 30.0], &[2])
        .expect("retained expert outputs should be valid");
    let missing_page_expert_outputs = runtime
        .array_from_f32(&[0.0, 40.0], &[2])
        .expect("missing expert outputs should be valid");
    let retained_weighted_output = weighted_page_output(
        &runtime,
        &retained_page_expert_outputs,
        &cached_plus_streamed_page_route.retained_page_slot_indices,
        &cached_plus_streamed_page_route.retained_scores,
    );
    let missing_weighted_output = weighted_page_output(
        &runtime,
        &missing_page_expert_outputs,
        &cached_plus_streamed_page_route.missing_page_slot_indices,
        &cached_plus_streamed_page_route.missing_scores,
    );
    let cached_plus_streamed_weighted_output = runtime
        .add(&retained_weighted_output, &missing_weighted_output)
        .expect("retained and missing weighted outputs should combine");

    let global_expert_outputs = runtime
        .array_from_f32(&[0.0, 10.0, 20.0, 30.0, 40.0], &[5])
        .expect("global expert outputs should be valid");
    let flattened_selected_indices = runtime
        .reshape(&selected_indices, &[4])
        .expect("selected indices should flatten");
    let flattened_selected_scores = runtime
        .reshape(&selected_scores, &[4])
        .expect("selected scores should flatten");
    let selected_global_outputs = runtime
        .take_axis(&global_expert_outputs, &flattened_selected_indices, 0)
        .expect("the combined route should gather global expert outputs");
    let combined_weighted_assignments = runtime
        .multiply(&selected_global_outputs, &flattened_selected_scores)
        .expect("the combined route should weight every assignment once");
    let combined_weighted_output = runtime
        .sum_axis(&combined_weighted_assignments, 0, false)
        .expect("the combined route should sum weighted assignments");

    let actual_cached_plus_streamed_output = cached_plus_streamed_weighted_output
        .to_vec_f32()
        .expect("the cached-plus-streamed weighted output should evaluate");
    let expected_combined_output = combined_weighted_output
        .to_vec_f32()
        .expect("the combined weighted output should evaluate");
    eprintln!(
        "[cached-plus-streamed-page-route] status=success cached_plus_streamed_output={actual_cached_plus_streamed_output:?} combined_output={expected_combined_output:?}"
    );
    assert_f32_close(
        &actual_cached_plus_streamed_output,
        &expected_combined_output,
    );
}

fn direct_mlx_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the cached-plus-streamed route memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize for a cached-plus-streamed route contract")
}

fn page_manifest(
    expert_ids: &[usize],
    page_slot_by_global_expert_id: &[u32],
) -> QuantizedExpertPageManifest {
    QuantizedExpertPageManifest {
        expert_ids: expert_ids.to_vec(),
        page_slot_by_global_expert_id: page_slot_by_global_expert_id.to_vec(),
        source_manifests: Vec::new(),
        payload_byte_count: 1,
    }
}

fn weighted_page_output(
    runtime: &MlxRuntime,
    page_expert_outputs: &MlxArray,
    page_slot_indices: &MlxArray,
    assignment_scores: &MlxArray,
) -> MlxArray {
    // This mirrors the production route's observable weighted sum while keeping
    // the fixture independent of model projection geometry and model artifacts.
    let selected_page_outputs = runtime
        .take_axis(page_expert_outputs, page_slot_indices, 0)
        .expect("compact page slots should gather expert outputs");
    let weighted_assignments = runtime
        .multiply(&selected_page_outputs, assignment_scores)
        .expect("compact expert outputs should multiply by matching scores");
    runtime
        .sum_axis(&weighted_assignments, 0, false)
        .expect("compact weighted assignments should sum")
}

fn assert_f32_close(actual_values: &[f32], expected_values: &[f32]) {
    assert_eq!(actual_values.len(), expected_values.len());
    for (actual_value, expected_value) in actual_values.iter().zip(expected_values) {
        let comparison_scale = expected_value.abs().max(1.0);
        assert!(
            (*actual_value - *expected_value).abs() <= 1e-6 * comparison_scale,
            "expected {actual_value} to be close to {expected_value}"
        );
    }
}
