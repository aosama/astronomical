use std::collections::HashMap;

use astronomical_model_serving::{
    ExpertLayerResidencyTarget, ExpertMemoryMode, LagunaAttentionProjection, LagunaDecoderState,
    LagunaExpertProjection, LagunaGlobalTensorRole, LagunaLayerTensorRole, LagunaModel,
    LagunaNativeWeights, LagunaTensorComponent, LagunaTensorId, MemoryPhase,
    PerformanceAttribution, PerformanceOperation,
};

use super::page_artifact::{filled, paging_plan, test_runtime, write_sparse_artifact};

#[path = "paged_model/demotion.rs"]
mod demotion;

fn weight_id(role: LagunaGlobalTensorRole) -> LagunaTensorId {
    LagunaTensorId::Global {
        role,
        component: LagunaTensorComponent::Weight,
    }
}

fn layer_weight_id(layer_index: usize, role: LagunaLayerTensorRole) -> LagunaTensorId {
    LagunaTensorId::Layer {
        layer_index,
        role,
        component: LagunaTensorComponent::Weight,
    }
}

fn bind_core_page_weights(
    runtime: &astronomical_runtime_integration::MlxRuntime,
    contract: &astronomical_model_serving::LagunaTargetContract,
    include_routed_experts: bool,
) -> Result<LagunaNativeWeights, astronomical_model_serving::LagunaExecutionError> {
    let mut tensors = HashMap::new();
    tensors.insert(
        weight_id(LagunaGlobalTensorRole::TokenEmbedding),
        filled(runtime, &[8, 4], 0.05),
    );
    tensors.insert(
        weight_id(LagunaGlobalTensorRole::FinalNormalization),
        filled(runtime, &[4], 1.0),
    );
    tensors.insert(
        weight_id(LagunaGlobalTensorRole::OutputHead),
        filled(runtime, &[8, 4], 0.05),
    );
    tensors.insert(
        layer_weight_id(0, LagunaLayerTensorRole::InputNormalization),
        filled(runtime, &[4], 1.0),
    );
    tensors.insert(
        layer_weight_id(0, LagunaLayerTensorRole::PostAttentionNormalization),
        filled(runtime, &[4], 1.0),
    );
    tensors.insert(
        layer_weight_id(
            0,
            LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Query),
        ),
        filled(runtime, &[4, 4], 0.05),
    );
    tensors.insert(
        layer_weight_id(
            0,
            LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Key),
        ),
        filled(runtime, &[2, 4], 0.05),
    );
    tensors.insert(
        layer_weight_id(
            0,
            LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Value),
        ),
        filled(runtime, &[2, 4], 0.05),
    );
    tensors.insert(
        layer_weight_id(
            0,
            LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Output),
        ),
        filled(runtime, &[4, 4], 0.05),
    );
    tensors.insert(
        layer_weight_id(0, LagunaLayerTensorRole::AttentionQueryNormalization),
        filled(runtime, &[2], 1.0),
    );
    tensors.insert(
        layer_weight_id(0, LagunaLayerTensorRole::AttentionKeyNormalization),
        filled(runtime, &[2], 1.0),
    );
    tensors.insert(
        layer_weight_id(0, LagunaLayerTensorRole::Router),
        filled(runtime, &[2, 4], 0.1),
    );
    tensors.insert(
        layer_weight_id(0, LagunaLayerTensorRole::RouterCorrectionBias),
        runtime
            .array_from_f32(&[0.5, 0.0], &[2])
            .expect("router correction bias should be valid"),
    );
    if include_routed_experts {
        tensors.insert(
            layer_weight_id(
                0,
                LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Gate),
            ),
            filled(runtime, &[2, 4, 4], 0.04),
        );
        tensors.insert(
            layer_weight_id(
                0,
                LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Up),
            ),
            filled(runtime, &[2, 4, 4], 0.05),
        );
        tensors.insert(
            layer_weight_id(
                0,
                LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Down),
            ),
            filled(runtime, &[2, 4, 4], 0.06),
        );
    }
    LagunaNativeWeights::bind(runtime, tensors, contract)
}

#[tokio::test]
async fn should_page_prefill_and_decode_through_the_model_and_report_status() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = tempfile::tempdir().expect("paged model directory");
    write_sparse_artifact(model_directory.path(), false);
    let (artifact, plan) = paging_plan(model_directory.path());
    let runtime = test_runtime();
    let contract = artifact.target_contract().clone();
    let weights = bind_core_page_weights(&runtime, &contract, false)
        .expect("core Laguna weights should bind without stacked experts");
    let model = LagunaModel::new(
        contract,
        weights,
        crate::common::test_worker_kernel_capabilities(&runtime),
    )
    .expect("a core-only model should construct")
    .with_paging_plan(plan)
    .expect("a paging plan should attach to a core-only model");
    assert_eq!(model.expert_memory_mode(), ExpertMemoryMode::Paged);
    let mut decoder_state =
        LagunaDecoderState::empty(model.contract()).expect("decoder state should allocate");
    let mut performance_attribution = PerformanceAttribution::enabled();

    let prompt_tokens = runtime
        .array_from_u32(&[1], &[1])
        .expect("Romeo-shaped prompt tokens should be valid");
    let prompt_logits = model
        .forward_prefill(
            &runtime,
            &prompt_tokens,
            &mut decoder_state,
            &mut performance_attribution,
        )
        .expect("one-token paged prefill should retain prefill residency semantics");
    assert_eq!(prompt_logits.shape(), vec![1, 1, 8]);
    assert_eq!(model.expert_memory_mode(), ExpertMemoryMode::Paged);
    let prefill_telemetry = model.expert_residency_telemetry();
    assert_eq!(prefill_telemetry.total_layer_count, 1);
    assert!(prefill_telemetry.resident_expert_count > 0);
    assert!(prefill_telemetry.resident_expert_payload_bytes > 0);
    let prefill_statistics = model.expert_weight_memory_cache_statistics();
    assert_eq!(prefill_statistics.disk_page_load_count, 2);
    assert_eq!(prefill_statistics.disk_batch_load_count, 1);
    assert_eq!(prefill_statistics.complete_layer_count, 1);
    assert!(
        performance_attribution
            .operation_measurement(PerformanceOperation::PagedMoeGraphConstruction)
            .is_some()
    );
    assert!(
        performance_attribution
            .operation_measurement(
                PerformanceOperation::PagedMoeOutputMaterializationSynchronizationWait,
            )
            .is_some(),
        "paged execution must sever each streamed page from the lazy MLX graph"
    );
    assert!(
        performance_attribution
            .operation_measurement(
                PerformanceOperation::MandatoryPrefillCompleteLayerMaterializationWait
            )
            .is_some()
    );
    let prefill_plan = model
        .active_expert_residency_plan()
        .expect("paged prefill should publish a phase-aware plan");
    assert_eq!(prefill_plan.phase, MemoryPhase::Prefill);
    assert_eq!(
        prefill_plan.layer_targets[0],
        ExpertLayerResidencyTarget::StreamOperationLocal
    );

    let decode_tokens = runtime
        .array_from_u32(&[3], &[1])
        .expect("Romeo-shaped decode tokens should be valid");
    let decode_logits = model
        .forward(
            &runtime,
            &decode_tokens,
            &mut decoder_state,
            &mut performance_attribution,
        )
        .expect("paged decode should execute");
    assert_eq!(decode_logits.shape(), vec![1, 1, 8]);
    let decode_telemetry = model.expert_residency_telemetry();
    assert!(decode_telemetry.resident_expert_count > 0);
    assert!(decode_telemetry.resident_expert_payload_bytes > 0);
    assert!(
        decode_telemetry.resident_expert_payload_bytes
            < prefill_telemetry.resident_expert_payload_bytes
    );
    assert_eq!(
        model
            .expert_weight_memory_cache_statistics()
            .disk_page_load_count,
        3
    );
    assert!(
        performance_attribution
            .operation_measurement(
                PerformanceOperation::MandatoryDecodeRoutePageMaterializationWait
            )
            .is_some()
    );
    let decode_plan = model
        .active_expert_residency_plan()
        .expect("paged decode should publish a phase-aware plan");
    assert_eq!(decode_plan.phase, MemoryPhase::Decode);
}

#[tokio::test]
async fn should_reject_a_core_only_model_without_a_paging_plan() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = tempfile::tempdir().expect("core-only rejection directory");
    write_sparse_artifact(model_directory.path(), false);
    let (artifact, _plan) = paging_plan(model_directory.path());
    let runtime = test_runtime();
    let contract = artifact.target_contract().clone();
    let weights = bind_core_page_weights(&runtime, &contract, false)
        .expect("core Laguna weights should bind without stacked experts");
    let model = LagunaModel::new(
        contract,
        weights,
        crate::common::test_worker_kernel_capabilities(&runtime),
    )
    .expect("a core-only model should construct before paging is required");
    let mut decoder_state =
        LagunaDecoderState::empty(model.contract()).expect("decoder state should allocate");
    let mut performance_attribution = PerformanceAttribution::disabled();
    let prompt_tokens = runtime
        .array_from_u32(&[1, 2], &[2])
        .expect("Romeo-shaped prompt tokens should be valid");
    let rejection = model.forward(
        &runtime,
        &prompt_tokens,
        &mut decoder_state,
        &mut performance_attribution,
    );
    assert!(rejection.is_err());
}

#[tokio::test]
async fn should_keep_a_fully_bound_model_resident_even_with_a_paging_plan() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = tempfile::tempdir().expect("resident page-plan directory");
    write_sparse_artifact(model_directory.path(), false);
    let (artifact, plan) = paging_plan(model_directory.path());
    let runtime = test_runtime();
    let contract = artifact.target_contract().clone();
    let weights = bind_core_page_weights(&runtime, &contract, true)
        .expect("stacked Laguna experts should bind");
    let model = LagunaModel::new(
        contract,
        weights,
        crate::common::test_worker_kernel_capabilities(&runtime),
    )
    .expect("a resident model should construct")
    .with_paging_plan(plan)
    .expect("a resident model may still own a paging plan");
    assert_eq!(model.expert_memory_mode(), ExpertMemoryMode::Resident);
    let mut decoder_state =
        LagunaDecoderState::empty(model.contract()).expect("decoder state should allocate");
    let mut performance_attribution = PerformanceAttribution::enabled();
    let prompt_tokens = runtime
        .array_from_u32(&[1, 2], &[2])
        .expect("Romeo-shaped prompt tokens should be valid");
    let prompt_logits = model
        .forward(
            &runtime,
            &prompt_tokens,
            &mut decoder_state,
            &mut performance_attribution,
        )
        .expect("resident prefill should execute");
    assert_eq!(prompt_logits.shape(), vec![1, 1, 8]);
    assert_eq!(model.expert_memory_mode(), ExpertMemoryMode::Resident);
    let telemetry = model.expert_residency_telemetry();
    assert!(telemetry.resident_expert_count > 0);
    assert!(telemetry.resident_expert_payload_bytes > 0);
    assert_eq!(
        model
            .expert_weight_memory_cache_statistics()
            .disk_page_load_count,
        0
    );
    assert!(
        performance_attribution
            .operation_measurement(PerformanceOperation::ResidentMoeGraphConstruction)
            .is_some()
    );
}

#[tokio::test]
async fn should_retain_a_complete_layer_when_the_ceiling_fits_and_reuse_it() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = tempfile::tempdir().expect("retain-on-ceiling directory");
    write_sparse_artifact(model_directory.path(), false);
    let (artifact, plan) = paging_plan(model_directory.path());
    let complete_layer_payload_bytes = plan.sparse_layers()[0]
        .complete_layer_payload_byte_count()
        .expect("complete-layer bytes");
    let runtime = test_runtime();
    let contract = artifact.target_contract().clone();
    let weights = bind_core_page_weights(&runtime, &contract, false)
        .expect("core Laguna weights should bind without stacked experts");
    let model = LagunaModel::new(
        contract,
        weights,
        crate::common::test_worker_kernel_capabilities(&runtime),
    )
    .expect("a core-only model should construct")
    .with_paging_plan(plan)
    .expect("a paging plan should attach")
    .with_retained_expert_ceiling(complete_layer_payload_bytes)
    .expect("a fitting ceiling should install");
    assert_eq!(model.expert_memory_mode(), ExpertMemoryMode::Paged);
    let mut decoder_state =
        LagunaDecoderState::empty(model.contract()).expect("decoder state should allocate");
    let mut performance_attribution = PerformanceAttribution::enabled();
    let prompt_tokens = runtime
        .array_from_u32(&[1, 2], &[2])
        .expect("Romeo-shaped prompt tokens should be valid");
    model
        .forward(
            &runtime,
            &prompt_tokens,
            &mut decoder_state,
            &mut performance_attribution,
        )
        .expect("first prefill should stream and commit");
    assert_eq!(model.expert_memory_mode(), ExpertMemoryMode::Resident);
    let first_telemetry = model.expert_residency_telemetry();
    assert!(first_telemetry.resident_expert_count > 0);
    assert!(first_telemetry.resident_expert_payload_bytes > 0);
    assert_eq!(
        model
            .expert_weight_memory_cache_statistics()
            .disk_page_load_count,
        2
    );
    assert!(
        performance_attribution
            .operation_measurement(PerformanceOperation::ExpertResidencyCommit)
            .is_some()
    );

    let second_prompt_tokens = runtime
        .array_from_u32(&[1, 2], &[2])
        .expect("second Romeo-shaped prompt should be valid");
    model
        .forward(
            &runtime,
            &second_prompt_tokens,
            &mut decoder_state,
            &mut performance_attribution,
        )
        .expect("second prefill should reuse the retained complete layer");
    assert_eq!(model.expert_memory_mode(), ExpertMemoryMode::Resident);
    assert_eq!(
        model
            .expert_weight_memory_cache_statistics()
            .disk_page_load_count,
        2
    );
    let reuse_plan = model
        .active_expert_residency_plan()
        .expect("retained reuse should still publish a plan");
    assert_eq!(
        reuse_plan.layer_targets[0],
        ExpertLayerResidencyTarget::PreserveComplete
    );
}

#[tokio::test]
async fn should_evict_a_retained_layer_when_the_ceiling_drops_to_zero() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = tempfile::tempdir().expect("live-ceiling reclaim directory");
    write_sparse_artifact(model_directory.path(), false);
    let (artifact, plan) = paging_plan(model_directory.path());
    let complete_layer_payload_bytes = plan.sparse_layers()[0]
        .complete_layer_payload_byte_count()
        .expect("complete-layer bytes");
    let runtime = test_runtime();
    let contract = artifact.target_contract().clone();
    let weights = bind_core_page_weights(&runtime, &contract, false)
        .expect("core Laguna weights should bind without stacked experts");
    let model = LagunaModel::new(
        contract,
        weights,
        crate::common::test_worker_kernel_capabilities(&runtime),
    )
    .expect("a core-only model should construct")
    .with_paging_plan(plan)
    .expect("a paging plan should attach")
    .with_retained_expert_ceiling(complete_layer_payload_bytes)
    .expect("a fitting ceiling should install");
    let mut decoder_state =
        LagunaDecoderState::empty(model.contract()).expect("decoder state should allocate");
    let mut performance_attribution = PerformanceAttribution::disabled();
    let prompt_tokens = runtime
        .array_from_u32(&[1, 2], &[2])
        .expect("Romeo-shaped prompt tokens should be valid");
    model
        .forward(
            &runtime,
            &prompt_tokens,
            &mut decoder_state,
            &mut performance_attribution,
        )
        .expect("prefill should retain the complete layer");
    assert_eq!(model.expert_memory_mode(), ExpertMemoryMode::Resident);

    model
        .set_retained_expert_ceiling(0)
        .expect("a zero ceiling should reclaim retained experts");
    assert_eq!(model.expert_memory_mode(), ExpertMemoryMode::Paged);
    assert_eq!(model.expert_residency_telemetry().resident_expert_count, 0);

    model
        .forward(
            &runtime,
            &prompt_tokens,
            &mut decoder_state,
            &mut performance_attribution,
        )
        .expect("post-reclaim prefill should stream again");
    assert_eq!(
        model
            .expert_weight_memory_cache_statistics()
            .disk_page_load_count,
        4
    );
}

#[tokio::test]
async fn should_retain_a_routed_decode_page_when_the_ceiling_fits_only_that_page() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = tempfile::tempdir().expect("routed-page retain directory");
    write_sparse_artifact(model_directory.path(), false);
    let (artifact, plan) = paging_plan(model_directory.path());
    let routed_page_payload_bytes = plan.sparse_layers()[0]
        .routed_page_payload_byte_count()
        .expect("routed-page bytes");
    let runtime = test_runtime();
    let contract = artifact.target_contract().clone();
    let weights = bind_core_page_weights(&runtime, &contract, false)
        .expect("core Laguna weights should bind without stacked experts");
    let model = LagunaModel::new(
        contract,
        weights,
        crate::common::test_worker_kernel_capabilities(&runtime),
    )
    .expect("a core-only model should construct")
    .with_paging_plan(plan)
    .expect("a paging plan should attach")
    .with_retained_expert_ceiling(routed_page_payload_bytes)
    .expect("a routed-page ceiling should install");
    let mut decoder_state =
        LagunaDecoderState::empty(model.contract()).expect("decoder state should allocate");
    let mut performance_attribution = PerformanceAttribution::enabled();
    let prompt_tokens = runtime
        .array_from_u32(&[1, 2], &[2])
        .expect("Romeo-shaped prompt tokens should be valid");
    model
        .forward(
            &runtime,
            &prompt_tokens,
            &mut decoder_state,
            &mut performance_attribution,
        )
        .expect("prefill should stream a complete layer without retaining it");
    assert_eq!(model.expert_memory_mode(), ExpertMemoryMode::Paged);
    assert_eq!(
        model
            .expert_weight_memory_cache_statistics()
            .disk_page_load_count,
        2
    );

    let decode_tokens = runtime
        .array_from_u32(&[3], &[1])
        .expect("Romeo-shaped decode tokens should be valid");
    model
        .forward(
            &runtime,
            &decode_tokens,
            &mut decoder_state,
            &mut performance_attribution,
        )
        .expect("decode should stream and retain the routed page");
    assert_eq!(model.expert_memory_mode(), ExpertMemoryMode::Hybrid);
    let decode_telemetry = model.expert_residency_telemetry();
    assert!(decode_telemetry.resident_expert_count > 0);
    assert!(decode_telemetry.resident_expert_payload_bytes > 0);
    assert_eq!(
        model
            .expert_weight_memory_cache_statistics()
            .disk_page_load_count,
        3
    );

    let second_decode_tokens = runtime
        .array_from_u32(&[3], &[1])
        .expect("second Romeo-shaped decode tokens should be valid");
    model
        .forward(
            &runtime,
            &second_decode_tokens,
            &mut decoder_state,
            &mut performance_attribution,
        )
        .expect("second decode should reuse the retained routed page");
    assert_eq!(model.expert_memory_mode(), ExpertMemoryMode::Hybrid);
    assert_eq!(
        model
            .expert_weight_memory_cache_statistics()
            .disk_page_load_count,
        3
    );
    let reuse_plan = model
        .active_expert_residency_plan()
        .expect("routed reuse should still publish a plan");
    assert_eq!(
        reuse_plan.layer_targets[0],
        ExpertLayerResidencyTarget::PreservePartial
    );
}

#[tokio::test]
async fn should_restore_decoder_allocation_ownership_after_a_failed_prefill_attempt() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = tempfile::tempdir().expect("checkpoint model directory");
    write_sparse_artifact(model_directory.path(), false);
    let (artifact, plan) = paging_plan(model_directory.path());
    let runtime = test_runtime();
    let contract = artifact.target_contract().clone();
    let weights = bind_core_page_weights(&runtime, &contract, false)
        .expect("core Laguna weights should bind without stacked experts");
    let model = LagunaModel::new(
        contract,
        weights,
        crate::common::test_worker_kernel_capabilities(&runtime),
    )
    .expect("a checkpoint model should construct")
    .with_paging_plan(plan)
    .expect("a checkpoint model needs paging");
    let mut decoder_state =
        LagunaDecoderState::empty(model.contract()).expect("decoder state should allocate");
    let allocation_checkpoint = decoder_state
        .allocation_checkpoint()
        .expect("an empty decoder checkpoint should retain safely");
    let prompt_tokens = runtime
        .array_from_u32(&[1, 2], &[2])
        .expect("Romeo-shaped prompt tokens should be valid");
    let mut performance_attribution = PerformanceAttribution::disabled();

    model
        .forward(
            &runtime,
            &prompt_tokens,
            &mut decoder_state,
            &mut performance_attribution,
        )
        .expect("the attempted prefill should mutate decoder state");
    assert_eq!(decoder_state.absolute_position(0), Some(2));

    decoder_state
        .restore_allocation_checkpoint(allocation_checkpoint)
        .expect("checkpoint restoration should return to the pre-attempt state");
    assert_eq!(decoder_state.absolute_position(0), Some(0));
    assert_eq!(decoder_state.payload_byte_count(), 0);
}
