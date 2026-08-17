//! Complete Laguna model wiring compared with a decomposed MLX operations oracle.
//!
//! Primitive attention, rotary, and gathered-projection tests already prove the
//! individual MLX operations against host references. These tests deliberately
//! compose those operations again without Laguna's model helpers, so a mistake
//! in descriptor order, residual placement, cache ownership, or output binding
//! cannot make the production path and its oracle fail in the same way.

mod binding;
mod fixture;
mod moe_operations;
mod operations;
mod rows;
mod tensor_fixture;
mod tensor_identity;

use astronomical_model_serving::{
    LagunaDecoderState, PerformanceAttribution, PerformanceOperation,
};
use astronomical_runtime_integration::{MlxDtype, MlxMemoryLimits, MlxRuntime};

use self::fixture::build_fixture;
use self::operations::{ReferenceDecoderState, reference_forward};
use self::rows::{ReferenceRow, generic_moe_rows, generic_rows, named_rows};
use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

#[tokio::test]
async fn should_match_complete_model_reference_for_generalized_descriptor_rows() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    for row in generic_rows() {
        assert_row_matches_reference(&runtime, &row);
    }
}

#[tokio::test]
async fn should_match_complete_model_reference_for_named_xs_and_s_rows() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    for row in named_rows() {
        assert_row_matches_reference(&runtime, &row);
    }
}

#[tokio::test]
async fn should_match_complete_model_reference_for_resident_moe_rows() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    for row in generic_moe_rows() {
        assert_row_matches_reference(&runtime, &row);
    }
}

#[tokio::test]
async fn should_avoid_optional_model_diagnostics_when_attribution_is_disabled() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let row = generic_moe_rows()
        .into_iter()
        .find(|row| row.row_name == "native_mixed_shared")
        .expect("the resident-MoE attribution row should exist");
    let fixture = build_fixture(&runtime, &row);
    let mut decoder_state = LagunaDecoderState::empty(fixture.model.contract())
        .expect("the disabled-attribution cache should construct");
    let token_ids = runtime
        .array_from_u32(
            &row.prefill_token_ids,
            &[row.prefill_token_ids.len() as i32],
        )
        .expect("disabled-attribution tokens should construct");
    let mut disabled_attribution = PerformanceAttribution::disabled();
    fixture
        .model
        .forward(
            &runtime,
            &token_ids,
            &mut decoder_state,
            &mut disabled_attribution,
        )
        .expect("disabled attribution must not alter model execution");
    for operation in attributed_operations() {
        assert!(
            disabled_attribution
                .operation_measurement(operation)
                .is_none(),
            "disabled attribution unexpectedly retained {operation:?}"
        );
    }
}

fn assert_row_matches_reference(runtime: &MlxRuntime, row: &ReferenceRow) {
    let fixture = build_fixture(runtime, row);
    assert_eq!(
        fixture
            .observed_affine_profiles
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        row.expected_affine_profiles,
        "{} executed affine bit/group profiles",
        row.row_name
    );
    let mut production_state = LagunaDecoderState::empty(fixture.model.contract())
        .unwrap_or_else(|error| panic!("{} production cache failed: {error:?}", row.row_name));
    let mut reference_state = ReferenceDecoderState::new(fixture.model.contract());
    let mut attribution = PerformanceAttribution::enabled();

    for (boundary_name, token_ids) in [
        ("prefill", row.prefill_token_ids.as_slice()),
        ("decode", row.decode_token_ids.as_slice()),
    ] {
        let token_array = runtime
            .array_from_u32(token_ids, &[token_ids.len() as i32])
            .unwrap_or_else(|error| {
                panic!("{} {boundary_name} tokens failed: {error}", row.row_name)
            });
        let production_logits = fixture
            .model
            .forward(
                runtime,
                &token_array,
                &mut production_state,
                &mut attribution,
            )
            .unwrap_or_else(|error| panic!("{} {boundary_name} failed: {error:?}", row.row_name));
        let reference_logits = reference_forward(
            runtime,
            fixture.model.contract(),
            &fixture.reference_tensors,
            &token_array,
            &mut reference_state,
        )
        .unwrap_or_else(|error| {
            panic!("{} {boundary_name} reference failed: {error}", row.row_name)
        });
        assert_eq!(
            production_logits.dtype(),
            row.activation_dtype,
            "{}",
            row.row_name
        );
        assert_arrays_close(
            runtime,
            row.row_name,
            boundary_name,
            &production_logits,
            &reference_logits,
            row.tolerance,
        );
    }

    for layer_descriptor in fixture.model.contract().layers() {
        let layer_index = layer_descriptor.layer_index();
        assert_eq!(
            production_state.absolute_position(layer_index),
            Some(reference_state.absolute_position(layer_index)),
            "{} layer {layer_index} absolute position",
            row.row_name
        );
        assert_eq!(
            production_state.committed_token_count(layer_index),
            Some(reference_state.committed_token_count(layer_index)),
            "{} layer {layer_index} committed tokens",
            row.row_name
        );
    }
    for operation in attributed_operations() {
        if operation == PerformanceOperation::SoftplusAttentionGateApplication
            && !row.has_attention_gate
        {
            continue;
        }
        if operation == PerformanceOperation::RotatingKeyValueStateUpdate
            && !row.has_sliding_attention
        {
            continue;
        }
        if is_moe_operation(operation) && !row.has_sparse_feed_forward {
            continue;
        }
        if operation == PerformanceOperation::SharedExpertExecution && !row.has_shared_expert {
            continue;
        }
        if operation == PerformanceOperation::ExpertAssignmentPreparation
            && !row.expects_assignment_sort
        {
            assert!(
                attribution.operation_measurement(operation).is_none(),
                "{} unexpectedly attributed assignment sorting",
                row.row_name
            );
            continue;
        }
        assert!(
            attribution.operation_measurement(operation).is_some(),
            "{} did not attribute {operation:?}",
            row.row_name
        );
    }
    if row.has_sparse_feed_forward {
        assert_eq!(
            fixture.model.expert_memory_mode(),
            astronomical_model_serving::ExpertMemoryMode::Resident,
            "{} resident mode",
            row.row_name
        );
        assert_eq!(
            fixture
                .model
                .expert_weight_memory_cache_statistics()
                .disk_page_load_count,
            0,
            "{} resident execution must not read expert pages",
            row.row_name
        );
    }
}

fn attributed_operations() -> [PerformanceOperation; 12] {
    [
        PerformanceOperation::AttentionForwardSpan,
        PerformanceOperation::RotaryEmbeddingApplication,
        PerformanceOperation::RotatingKeyValueStateUpdate,
        PerformanceOperation::SoftplusAttentionGateApplication,
        PerformanceOperation::MlpForwardSpan,
        PerformanceOperation::FinalLogitsGraphConstruction,
        PerformanceOperation::RouterScoreSelection,
        PerformanceOperation::ExpertAssignmentPreparation,
        PerformanceOperation::GatheredExpertExecution,
        PerformanceOperation::ExpertWeightedReduction,
        PerformanceOperation::SharedExpertExecution,
        PerformanceOperation::ResidentMoeGraphConstruction,
    ]
}

fn is_moe_operation(operation: PerformanceOperation) -> bool {
    matches!(
        operation,
        PerformanceOperation::RouterScoreSelection
            | PerformanceOperation::ExpertAssignmentPreparation
            | PerformanceOperation::GatheredExpertExecution
            | PerformanceOperation::ExpertWeightedReduction
            | PerformanceOperation::SharedExpertExecution
            | PerformanceOperation::ResidentMoeGraphConstruction
    )
}

fn assert_arrays_close(
    runtime: &MlxRuntime,
    row_name: &str,
    boundary_name: &str,
    actual: &astronomical_runtime_integration::MlxArray,
    expected: &astronomical_runtime_integration::MlxArray,
    tolerance: f32,
) {
    // Comparison happens only after both lazy graphs are complete. Casting here
    // keeps production execution at its declared dtype while giving one stable
    // host representation for diagnostics.
    let evaluated_float32_values = |array: &astronomical_runtime_integration::MlxArray| {
        runtime
            .astype(array, MlxDtype::Float32)
            .and_then(|array| runtime.build_contiguous_row_major_copy(&array))
            .and_then(|array| array.to_vec_f32())
            .expect("reference comparison should evaluate")
    };
    let actual_values = evaluated_float32_values(actual);
    let expected_values = evaluated_float32_values(expected);
    assert_eq!(
        actual_values.len(),
        expected_values.len(),
        "{row_name} {boundary_name}"
    );
    for (element_index, (actual_value, expected_value)) in
        actual_values.iter().zip(&expected_values).enumerate()
    {
        let comparison_scale = expected_value.abs().max(1.0);
        assert!(
            (*actual_value - *expected_value).abs() <= tolerance * comparison_scale,
            "{row_name} {boundary_name} element {element_index}: expected {expected_value}, got {actual_value}"
        );
    }
}

pub(super) fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("Laguna model-reference memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}
