use std::collections::HashMap;

use astronomical_model_serving::{
    LagunaDecoderState, LagunaModel, LagunaNativeWeights, PerformanceAttribution,
    PerformanceOperation,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::common::laguna::{bind_tiny_weights, tiny_mixed_contract};
use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("Laguna execution test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

#[tokio::test]
async fn should_grow_full_state_and_bound_sliding_state_through_prefill_and_decode() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let contract = tiny_mixed_contract();
    let weights = bind_tiny_weights(&runtime, &contract);
    let model = LagunaModel::new(
        contract,
        weights,
        crate::common::test_worker_kernel_capabilities(&runtime),
    )
    .expect("the mixed model should construct");
    let mut decoder_state =
        LagunaDecoderState::empty(model.contract()).expect("decoder state should allocate");
    assert_eq!(decoder_state.payload_byte_count(), 0);
    let prefill_memory_projection = decoder_state
        .projected_forward_memory(model.contract(), 6)
        .expect("mixed prefill memory geometry should be exact");
    assert_eq!(prefill_memory_projection.persistent_growth_bytes(), 8_320);
    assert_eq!(
        prefill_memory_projection.sliding_temporary_workspace_bytes(),
        288
    );
    let chunk_bounded_admission_projection = decoder_state
        .projected_context_admission_memory(model.contract(), 6, 2)
        .expect("context admission should separate persistent growth from one executable chunk");
    let executable_chunk_projection = decoder_state
        .projected_forward_memory(model.contract(), 2)
        .expect("the executable chunk workspace should be exact");
    assert_eq!(
        chunk_bounded_admission_projection.persistent_growth_bytes(),
        prefill_memory_projection.persistent_growth_bytes()
    );
    assert_eq!(
        chunk_bounded_admission_projection.sliding_temporary_workspace_bytes(),
        executable_chunk_projection.sliding_temporary_workspace_bytes()
    );
    assert!(
        chunk_bounded_admission_projection.sliding_temporary_workspace_bytes()
            < prefill_memory_projection.sliding_temporary_workspace_bytes(),
        "a long prompt must not reserve a full-prompt rotating workspace when execution is chunked"
    );
    let mut performance_attribution = PerformanceAttribution::enabled();
    let prompt_tokens = runtime
        .array_from_u32(&[1, 2, 3, 4, 5, 6], &[6])
        .expect("prompt token ids should be valid");
    let prompt_logits = model
        .forward(
            &runtime,
            &prompt_tokens,
            &mut decoder_state,
            &mut performance_attribution,
        )
        .expect("mixed prefill should execute");
    assert_eq!(prompt_logits.shape(), vec![1, 1, 8]);
    assert_eq!(decoder_state.absolute_position(0), Some(6));
    assert_eq!(decoder_state.committed_token_count(1), Some(4));
    assert!(
        decoder_state.payload_byte_count() > 0,
        "a written Laguna decoder cache must report live context payload"
    );
    let decode_memory_projection = decoder_state
        .projected_forward_memory(model.contract(), 1)
        .expect("mixed decode memory geometry should be exact");
    assert_eq!(decode_memory_projection.persistent_growth_bytes(), 0);
    assert_eq!(
        decode_memory_projection.sliding_temporary_workspace_bytes(),
        0
    );

    let decode_tokens = runtime
        .array_from_u32(&[7], &[1])
        .expect("decode token ids should be valid");
    let decode_logits = model
        .forward(
            &runtime,
            &decode_tokens,
            &mut decoder_state,
            &mut performance_attribution,
        )
        .expect("mixed decode should execute");
    assert_eq!(decode_logits.shape(), vec![1, 1, 8]);
    assert_eq!(decoder_state.absolute_position(0), Some(7));
    assert_eq!(decoder_state.committed_token_count(1), Some(4));
    assert!(
        performance_attribution
            .operation_measurement(PerformanceOperation::AttentionForwardSpan)
            .is_some()
    );
}

#[tokio::test]
async fn should_submit_intermediate_prefill_layers_and_keep_resident_decode_as_one_graph() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let contract = tiny_mixed_contract();
    let weights = bind_tiny_weights(&runtime, &contract);
    // The documented production default keeps a resident multi-token prefill on
    // one lazy tape (zero interval), so the submission test configures the
    // configurable resident interval explicitly.
    let model = LagunaModel::new(
        contract,
        weights,
        crate::common::test_worker_kernel_capabilities(&runtime),
    )
    .expect("the mixed model should construct")
    .with_graph_submission_layer_intervals(1, 0, 0);
    let mut decoder_state =
        LagunaDecoderState::empty(model.contract()).expect("decoder state should allocate");
    let mut performance_attribution = PerformanceAttribution::enabled();
    let prompt_tokens = runtime
        .array_from_u32(&[1, 2, 3, 4, 5, 6], &[6])
        .expect("prompt token ids should be valid");

    model
        .forward(
            &runtime,
            &prompt_tokens,
            &mut decoder_state,
            &mut performance_attribution,
        )
        .expect("mixed prefill should execute");

    // Two layers at interval one commit after the first layer only. The caller
    // owns the terminal evaluation after final norm and logits.
    assert_eq!(
        performance_attribution
            .operation_measurement(PerformanceOperation::PrefillStateAsyncEvaluationSubmission)
            .map(|measurement| measurement.occurrence_count())
            .unwrap_or(0),
        1
    );

    let decode_tokens = runtime
        .array_from_u32(&[7], &[1])
        .expect("decode token ids should be valid");
    model
        .forward(
            &runtime,
            &decode_tokens,
            &mut decoder_state,
            &mut performance_attribution,
        )
        .expect("mixed decode should execute");
    assert_eq!(
        performance_attribution
            .operation_measurement(PerformanceOperation::PrefillStateAsyncEvaluationSubmission)
            .map(|measurement| measurement.occurrence_count())
            .unwrap_or(0),
        1
    );

    let lazy_tape_contract = tiny_mixed_contract();
    let lazy_tape_model = LagunaModel::new(
        lazy_tape_contract.clone(),
        bind_tiny_weights(&runtime, &lazy_tape_contract),
        crate::common::test_worker_kernel_capabilities(&runtime),
    )
    .expect("the comparison model should construct")
    .with_graph_submission_layer_intervals(0, 0, 0);
    let mut lazy_tape_decoder_state = LagunaDecoderState::empty(lazy_tape_model.contract())
        .expect("the comparison decoder state should allocate");
    let mut lazy_tape_attribution = PerformanceAttribution::enabled();
    lazy_tape_model
        .forward(
            &runtime,
            &prompt_tokens,
            &mut lazy_tape_decoder_state,
            &mut lazy_tape_attribution,
        )
        .expect("a zero-interval prefill should still execute");
    assert!(
        lazy_tape_attribution
            .operation_measurement(PerformanceOperation::PrefillStateAsyncEvaluationSubmission)
            .is_none()
    );
}

#[tokio::test]
async fn should_reject_a_missing_canonical_weight() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let contract = tiny_mixed_contract();
    let empty = HashMap::new();
    let rejection = LagunaNativeWeights::bind(&runtime, empty, &contract);
    assert!(rejection.is_err());
    let _ = runtime;
}
