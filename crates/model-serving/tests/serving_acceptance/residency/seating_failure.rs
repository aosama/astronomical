//! User journey: raising RAM cannot seat every expert. Stay truthful (not Resident)
//! and still answer Romeo.

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
#[ignore = "breaks complete-residency sources then proves raising RAM cannot seat experts and Romeo still works"]
async fn should_keep_serving_when_complete_residency_cannot_be_seated() {
    timeout(Duration::from_secs(120), async {
        initialize_automatic_residency_tracing();
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let mlx_memory_limits =
            crate::common::sample_machine_serving_acceptance_mlx_memory_limits().await;
        let machine_mlx_memory_ceiling_bytes =
            u64::try_from(mlx_memory_limits.active_memory_limit_bytes())
                .expect("the machine MLX ceiling should fit u64");
        let model_directory =
            crate::serving_acceptance::support::configured_resident_sparse_moe_model_directory();
        let validated_artifact = astronomical_model_serving::Qwen3_5ArtifactValidator::new()
            .validate(&model_directory, 20_480)
            .expect("the resident sparse artifact should validate");
        let (model_geometry, required_headroom_bytes) =
            mlx_ram_budget_model_geometry_from_validated_artifact(
                &validated_artifact,
                &model_directory,
            )
            .expect("the resident sparse artifact should expose disk RAM geometry");
        let residency_boundary = CompleteResidencyHeadroomBoundary::from_model_geometry(
            model_geometry,
            required_headroom_bytes,
        );
        let paging_ceiling_bytes = residency_boundary.paging_ceiling_bytes().expect(
            "this sparse artifact must have a ceiling that fits expert weights but not activation headroom",
        );
        let paging_ceiling_limit_bytes = usize::try_from(paging_ceiling_bytes)
            .expect("the paging ceiling should fit usize");
        eprintln!(
            "[seating-failure] status=progress phase=derived_ceiling expert_bytes={} headroom_bytes={required_headroom_bytes} paging_ceiling_bytes={paging_ceiling_bytes} machine_ceiling_bytes={machine_mlx_memory_ceiling_bytes}",
            model_geometry.complete_expert_payload_bytes
        );
        let request_id = RequestId::new(1_008);
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

        eprintln!("[seating-failure] status=progress phase=model_load");
        qwen3_5_engine
            .load()
            .await
            .expect("the sparse model should load under the paging ceiling");
        assert_ne!(
            qwen3_5_engine
                .expert_memory_mode_for_tests()
                .await
                .expect("the loaded model should expose its expert mode"),
            Some(ExpertMemoryMode::Resident),
            "the paging ceiling must not seat complete experts"
        );
        qwen3_5_engine
            .remove_resident_expert_source_files_for_tests()
            .await
            .expect("the journey should be able to break complete-residency sources");

        eprintln!("[seating-failure] status=progress phase=failed_ceiling_raise");
        qwen3_5_engine
            .update_mlx_memory_limit(machine_mlx_memory_ceiling_bytes)
            .await
            .expect_err("raising RAM must not silently seat experts when promotion sources are gone");
        assert_ne!(
            qwen3_5_engine
                .expert_memory_mode_for_tests()
                .await
                .expect("the failed seating should retain a truthful mode"),
            Some(ExpertMemoryMode::Resident),
            "failed seating must not claim complete residency"
        );

        let generation_finalization = serve_romeo_and_juliet_request(
            &mut qwen3_5_engine,
            request_id,
            prompt_token_ids,
            image_pad_token_id,
            "seating-failure",
        )
        .await;
        assert_ne!(
            generation_finalization.expert_memory_mode(),
            Some(ExpertMemoryMode::Resident),
            "Romeo must still run without claiming complete residency"
        );
        eprintln!("[seating-failure] status=success");
    })
    .await
    .expect("the seating-failure journey must finish within 120 seconds");
}
