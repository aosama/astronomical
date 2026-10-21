//! User journey: the ceiling fits this sparse model's expert weights, but not
//! weights plus activation headroom. Status must not claim complete residency.

use std::time::Duration;

use astronomical_ipc_protocol::{ExpertMemoryMode, RequestId};
use astronomical_model_serving::{
    CompleteResidencyHeadroomBoundary, InferenceEngine,
    mlx_ram_budget_model_geometry_from_validated_artifact,
};
use tokio::time::timeout;

use super::support::{
    RESIDENCY_LIFECYCLE_PROMPT_TOKEN_COUNT, construct_automatic_residency_engine,
    initialize_automatic_residency_tracing, serve_romeo_and_juliet_request,
};

#[tokio::test]
#[ignore = "loads a configured sparse MoE under an artifact-derived ceiling that fits weights but not activation headroom"]
async fn should_not_claim_complete_residency_when_activation_headroom_does_not_fit() {
    timeout(Duration::from_secs(120), async {
        initialize_automatic_residency_tracing();
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let request_id = RequestId::new(1_041);
        let model_directory = crate::common::configured_large_sparse_moe_model_directory();
        let validated_artifact = astronomical_model_serving::Qwen3_5ArtifactValidator::new()
            .validate(&model_directory, 20_480)
            .expect("the configured sparse artifact should validate");
        let (model_geometry, required_headroom_bytes) =
            mlx_ram_budget_model_geometry_from_validated_artifact(
                &validated_artifact,
                &model_directory,
            )
            .expect("the sparse artifact should expose disk RAM geometry");
        let paging_ceiling_bytes = CompleteResidencyHeadroomBoundary::from_model_geometry(
            model_geometry,
            required_headroom_bytes,
        )
        .paging_ceiling_bytes()
        .expect(
            "this sparse artifact must have a ceiling that fits expert weights but not activation headroom",
        );
        let machine_mlx_memory_ceiling_bytes = u64::try_from(
            crate::common::sample_machine_serving_acceptance_mlx_memory_limits()
                .await
                .active_memory_limit_bytes(),
        )
        .expect("the machine MLX ceiling should fit u64");
        assert!(
            paging_ceiling_bytes <= machine_mlx_memory_ceiling_bytes,
            "this machine's MLX ceiling is below the artifact's expert payload, so the headroom-boundary journey cannot run"
        );
        let paging_ceiling_limit_bytes = usize::try_from(paging_ceiling_bytes)
            .expect("the paging ceiling should fit usize");
        eprintln!(
            "[complete-residency-headroom] status=progress phase=derived_ceiling core_bytes={} expert_bytes={} headroom_bytes={required_headroom_bytes} ceiling_bytes={paging_ceiling_bytes}",
            model_geometry.model_core_payload_bytes,
            model_geometry.complete_expert_payload_bytes
        );
        let (
            mut qwen3_5_engine,
            prompt_token_ids,
            image_pad_token_id,
            _context_memory_reservation_bytes,
        ) = construct_automatic_residency_engine(
            model_directory,
            paging_ceiling_limit_bytes,
            paging_ceiling_limit_bytes,
            request_id,
            RESIDENCY_LIFECYCLE_PROMPT_TOKEN_COUNT,
        );

        eprintln!("[complete-residency-headroom] status=progress phase=model_load");
        let engine_load_result = qwen3_5_engine
            .load()
            .await
            .expect("the sparse model should load under the artifact-derived paging ceiling");
        assert_ne!(
            engine_load_result.expert_memory_mode(),
            Some(ExpertMemoryMode::Resident),
            "readiness must not claim complete residency when activation headroom does not fit"
        );
        assert_ne!(
            qwen3_5_engine
                .expert_memory_mode_for_tests()
                .await
                .expect("the loaded model should expose its expert mode"),
            Some(ExpertMemoryMode::Resident),
            "startup admission must not seat complete experts when activation headroom does not fit"
        );

        let generation_finalization = serve_romeo_and_juliet_request(
            &mut qwen3_5_engine,
            request_id,
            prompt_token_ids,
            image_pad_token_id,
            "complete-residency-headroom",
        )
        .await;
        assert_ne!(
            generation_finalization.expert_memory_mode(),
            Some(ExpertMemoryMode::Resident),
            "after Romeo the model must still not claim complete residency"
        );
        eprintln!("[complete-residency-headroom] status=success");
    })
    .await
    .expect("the complete-residency headroom journey must finish within 120 seconds");
}
