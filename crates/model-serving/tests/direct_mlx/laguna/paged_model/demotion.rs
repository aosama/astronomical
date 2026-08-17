//! Resident-to-paged transition parity and lazy complete-layer recovery.

use astronomical_model_serving::{
    ExpertMemoryMode, LagunaDecoderState, LagunaModel, PerformanceAttribution,
};

use super::super::page_artifact::{paging_plan, test_runtime, write_sparse_artifact};
use super::bind_core_page_weights;

#[tokio::test]
async fn should_demote_native_experts_and_recover_complete_residency_through_bounded_pages() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = tempfile::tempdir().expect("resident demotion model directory");
    write_sparse_artifact(model_directory.path(), false);
    let (artifact, plan) = paging_plan(model_directory.path());
    let complete_layer_payload_bytes = plan.sparse_layers()[0]
        .complete_layer_payload_byte_count()
        .expect("complete layer bytes should be exact");
    let runtime = test_runtime();
    let contract = artifact.target_contract().clone();
    let weights = bind_core_page_weights(&runtime, &contract, true)
        .expect("resident Laguna weights should bind");
    let mut model = LagunaModel::new(contract, weights)
        .expect("a resident model should construct")
        .with_paging_plan(plan)
        .expect("resident demotion requires a paging fallback")
        .with_retained_expert_ceiling(complete_layer_payload_bytes)
        .expect("a complete retained layer should fit");
    let prompt_tokens = runtime
        .array_from_u32(&[1, 2], &[2])
        .expect("Romeo-shaped prompt tokens should be valid");
    let resident_output_values = {
        let mut resident_decoder_state =
            LagunaDecoderState::empty(model.contract()).expect("resident state should allocate");
        let mut resident_attribution = PerformanceAttribution::disabled();
        model
            .forward(
                &runtime,
                &prompt_tokens,
                &mut resident_decoder_state,
                &mut resident_attribution,
            )
            .expect("resident prefill should execute")
            .to_vec_f32()
            .expect("resident output should materialize")
    };

    let mut transition_attribution = PerformanceAttribution::enabled();
    assert!(
        model
            .demote_native_routed_experts(&runtime, &mut transition_attribution)
            .expect("native expert demotion should succeed")
    );
    assert_eq!(model.expert_memory_mode(), ExpertMemoryMode::Paged);

    let mut paged_decoder_state =
        LagunaDecoderState::empty(model.contract()).expect("paged state should allocate");
    let mut paged_attribution = PerformanceAttribution::disabled();
    let paged_output_values = model
        .forward(
            &runtime,
            &prompt_tokens,
            &mut paged_decoder_state,
            &mut paged_attribution,
        )
        .expect("paged prefill should execute after native demotion")
        .to_vec_f32()
        .expect("paged output should materialize");
    assert_eq!(resident_output_values.len(), paged_output_values.len());
    for (resident_value, paged_value) in resident_output_values.iter().zip(paged_output_values) {
        assert!((resident_value - paged_value).abs() < 1e-5);
    }
    assert_eq!(model.expert_memory_mode(), ExpertMemoryMode::Resident);
    assert_eq!(
        model
            .expert_weight_memory_cache_statistics()
            .mandatory_read_promotion_count,
        1
    );
}
