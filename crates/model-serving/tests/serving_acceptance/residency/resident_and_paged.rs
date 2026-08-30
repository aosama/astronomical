//! Resident seating and live MLX-ceiling journeys for one sparse artifact.
//!
//! Ceilings come from that artifact's geometry, not souvenir gigabyte constants.

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
#[ignore = "loads the resident sparse MoE under the machine ceiling when that ceiling covers weights plus headroom"]
async fn should_keep_the_complete_sparse_model_resident_when_idle_memory_is_sufficient() {
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
        let bytes_required_to_seat = CompleteResidencyHeadroomBoundary::from_model_geometry(
            model_geometry,
            required_headroom_bytes,
        )
        .static_complete_residency_bytes
        .saturating_add(required_headroom_bytes);
        assert!(
            machine_mlx_memory_ceiling_bytes >= bytes_required_to_seat,
            "this machine's MLX ceiling cannot seat this artifact's experts plus activation headroom"
        );
        eprintln!(
            "[complete-residency-fit] status=progress phase=machine_ceiling core_bytes={} expert_bytes={} headroom_bytes={required_headroom_bytes} required_bytes={bytes_required_to_seat} machine_ceiling_bytes={machine_mlx_memory_ceiling_bytes}",
            model_geometry.model_core_payload_bytes,
            model_geometry.complete_expert_payload_bytes
        );
        let request_id = RequestId::new(1_003);
        let (
            mut qwen3_5_engine,
            prompt_token_ids,
            image_pad_token_id,
            _context_memory_reservation_bytes,
        ) = construct_automatic_residency_engine(
            model_directory,
            mlx_memory_limits.active_memory_limit_bytes(),
            mlx_memory_limits.allocator_cache_memory_limit_bytes(),
            request_id,
            RESIDENCY_LIFECYCLE_PROMPT_TOKEN_COUNT,
        );

        eprintln!("[complete-residency-fit] status=progress phase=model_load");
        let engine_load_result = qwen3_5_engine
            .load()
            .await
            .expect("the resident sparse model should load under the machine ceiling");
        assert_eq!(
            engine_load_result.expert_memory_mode(),
            Some(ExpertMemoryMode::Resident),
            "readiness must report complete residency when the ceiling covers weights plus headroom"
        );
        assert_eq!(
            qwen3_5_engine
                .expert_memory_mode_for_tests()
                .await
                .expect("the loaded model should expose its expert mode"),
            Some(ExpertMemoryMode::Resident),
            "startup admission must seat complete experts when the ceiling covers weights plus headroom"
        );
        let native_cache_statistics_before_request = qwen3_5_engine
            .expert_weight_memory_cache_statistics_for_tests()
            .await
            .expect("the resident model should expose expert statistics");

        let generation_finalization = serve_romeo_and_juliet_request(
            &mut qwen3_5_engine,
            request_id,
            prompt_token_ids,
            image_pad_token_id,
            "complete-residency-fit",
        )
        .await;
        let native_cache_statistics_after_request = qwen3_5_engine
            .expert_weight_memory_cache_statistics_for_tests()
            .await
            .expect("the resident model should retain expert statistics");

        assert_eq!(
            generation_finalization.expert_memory_mode(),
            Some(ExpertMemoryMode::Resident),
            "after Romeo the model must remain fully resident"
        );
        assert_eq!(
            native_cache_statistics_after_request.disk_page_load_count,
            native_cache_statistics_before_request.disk_page_load_count,
            "resident inference must not stream expert pages from SSD"
        );
        assert_eq!(
            native_cache_statistics_after_request.disk_batch_load_count,
            native_cache_statistics_before_request.disk_batch_load_count,
            "resident inference must not batch-load expert pages from SSD"
        );
        eprintln!("[complete-residency-fit] status=success");
    })
    .await
    .expect("the complete-residency fit journey must finish within 120 seconds");
}

#[tokio::test]
#[ignore = "lowers then raises the live MLX ceiling on one resident sparse MoE and checks residency follows"]
async fn should_transition_the_complete_sparse_model_across_live_memory_limits() {
    timeout(Duration::from_secs(120), async {
        initialize_automatic_residency_tracing();
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let mlx_memory_limits =
            crate::common::sample_machine_serving_acceptance_mlx_memory_limits().await;
        let raised_mlx_memory_ceiling_bytes =
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
        let bytes_required_to_seat = residency_boundary
            .static_complete_residency_bytes
            .saturating_add(required_headroom_bytes);
        let paging_ceiling_bytes = residency_boundary.paging_ceiling_bytes().expect(
            "this sparse artifact must have a ceiling that fits expert weights but not activation headroom",
        );
        assert!(
            raised_mlx_memory_ceiling_bytes >= bytes_required_to_seat,
            "this machine's MLX ceiling cannot seat this artifact, so the live RAM-knob journey cannot run"
        );
        eprintln!(
            "[live-residency-transition] status=progress phase=derived_ceilings core_bytes={} expert_bytes={} headroom_bytes={required_headroom_bytes} paging_ceiling_bytes={paging_ceiling_bytes} machine_ceiling_bytes={raised_mlx_memory_ceiling_bytes}",
            model_geometry.model_core_payload_bytes,
            model_geometry.complete_expert_payload_bytes
        );
        let squeezed_request_id = RequestId::new(1_005);
        let (
            mut qwen3_5_engine,
            prompt_token_ids,
            image_pad_token_id,
            _context_memory_reservation_bytes,
        ) = construct_automatic_residency_engine(
            model_directory,
            mlx_memory_limits.active_memory_limit_bytes(),
            mlx_memory_limits.allocator_cache_memory_limit_bytes(),
            squeezed_request_id,
            RESIDENCY_LIFECYCLE_PROMPT_TOKEN_COUNT,
        );

        eprintln!("[live-residency-transition] status=progress phase=model_load");
        qwen3_5_engine
            .load()
            .await
            .expect("the fitting model should load as resident");
        assert_eq!(
            qwen3_5_engine
                .expert_memory_mode_for_tests()
                .await
                .expect("the loaded model should expose its initial mode"),
            Some(ExpertMemoryMode::Resident),
            "enough RAM must seat complete experts before the knob moves"
        );

        eprintln!("[live-residency-transition] status=progress phase=ceiling_lower");
        qwen3_5_engine
            .update_mlx_memory_limit(paging_ceiling_bytes)
            .await
            .expect("the idle model should accept the artifact-derived paging ceiling");
        assert_ne!(
            qwen3_5_engine
                .expert_memory_mode_for_tests()
                .await
                .expect("the lowered model should expose its mode"),
            Some(ExpertMemoryMode::Resident),
            "lowering the ceiling below seating must drop complete residency"
        );
        serve_romeo_and_juliet_request(
            &mut qwen3_5_engine,
            squeezed_request_id,
            prompt_token_ids,
            image_pad_token_id,
            "live-residency-transition-squeezed",
        )
        .await;
        assert_ne!(
            qwen3_5_engine
                .expert_memory_mode_for_tests()
                .await
                .expect("the squeezed model should still expose its mode"),
            Some(ExpertMemoryMode::Resident),
            "Romeo under the lowered ceiling must not restore complete residency"
        );

        eprintln!("[live-residency-transition] status=progress phase=ceiling_raise");
        qwen3_5_engine
            .update_mlx_memory_limit(raised_mlx_memory_ceiling_bytes)
            .await
            .expect("the idle model should accept the restored machine ceiling");
        assert_eq!(
            qwen3_5_engine
                .expert_memory_mode_for_tests()
                .await
                .expect("the raised model should expose resident mode"),
            Some(ExpertMemoryMode::Resident),
            "raising the ceiling must seat complete experts again"
        );
        eprintln!("[live-residency-transition] status=success");
    })
    .await
    .expect("the live residency-transition regression must finish within 120 seconds");
}
