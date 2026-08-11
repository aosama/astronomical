//! End-to-end resident, paged, live-ceiling, and request-pressure journeys.
//!
//! Each ignored test owns a real validated sparse artifact and MLX runtime. The
//! assertions distinguish ownership by observable mode and native page counters,
//! rather than accepting successful token generation as sufficient evidence.

use std::time::Duration;

use astronomical_ipc_protocol::{ExpertMemoryMode, RequestId};
use astronomical_model_serving::{InferenceEngine, Qwen3_5InferenceRequest};
use tokio::time::timeout;

use super::automatic_residency_support::{
    CONSTRAINED_MLX_MEMORY_CEILING_BYTES, RESIDENCY_LIFECYCLE_PROMPT_TOKEN_COUNT,
    RESIDENCY_QUALIFICATION_PROMPT_TOKEN_COUNT, complete_started_romeo_and_juliet_request,
    construct_automatic_residency_engine, initialize_automatic_residency_tracing,
    serve_romeo_and_juliet_request,
};

#[tokio::test]
#[ignore = "loads a fitting sparse checkpoint and verifies fully resident expert execution"]
async fn should_keep_the_complete_sparse_model_resident_when_idle_memory_is_sufficient() {
    timeout(Duration::from_secs(120), async {
        initialize_automatic_residency_tracing();
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let mlx_memory_limits =
            crate::common::sample_machine_model_artifact_qualification_mlx_memory_limits().await;
        let model_directory = crate::common::configured_ornith_model_artifact_directory();
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
            RESIDENCY_QUALIFICATION_PROMPT_TOKEN_COUNT,
        );

        eprintln!("[automatic-residency 1/5] status=progress phase=model_load ETA_seconds=90");
        let engine_load_result = qwen3_5_engine
            .load()
            .await
            .expect("the automatic-residency model should load");
        assert_eq!(
            engine_load_result.expert_memory_mode(),
            Some(ExpertMemoryMode::Resident),
            "readiness metadata must report complete residency before the first request"
        );
        assert_eq!(
            qwen3_5_engine
                .expert_memory_mode_for_tests()
                .await
                .expect("the loaded model should expose its expert mode"),
            Some(ExpertMemoryMode::Resident),
            "a complete sparse model that fits idle MLX memory must become resident before serving"
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
            "automatic-residency",
        )
        .await;
        let native_cache_statistics_after_request = qwen3_5_engine
            .expert_weight_memory_cache_statistics_for_tests()
            .await
            .expect("the resident model should retain expert statistics");

        eprintln!("[automatic-residency 4/5] status=progress phase=residency_assertion");
        assert_eq!(
            generation_finalization.expert_memory_mode(),
            Some(ExpertMemoryMode::Resident),
            "a fitting sparse model must remain resident after request cleanup"
        );
        assert_eq!(
            native_cache_statistics_after_request, native_cache_statistics_before_request,
            "resident inference must not consult, populate, or read through the native expert cache"
        );
        eprintln!("[automatic-residency 5/5] status=success");
    })
    .await
    .expect("the automatic-residency regression must finish within 120 seconds");
}

#[tokio::test]
#[ignore = "loads the same depth-one MTP checkpoint under a constrained ceiling and verifies paging"]
async fn should_page_the_complete_sparse_model_when_idle_memory_is_insufficient() {
    timeout(Duration::from_secs(120), async {
        initialize_automatic_residency_tracing();
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let request_id = RequestId::new(1_004);
        let model_directory = super::configured_smallest_depth_one_mtp_model_artifact_directory();
        let (
            mut qwen3_5_engine,
            prompt_token_ids,
            image_pad_token_id,
            _context_memory_reservation_bytes,
        ) = construct_automatic_residency_engine(
            model_directory,
            CONSTRAINED_MLX_MEMORY_CEILING_BYTES,
            CONSTRAINED_MLX_MEMORY_CEILING_BYTES,
            request_id,
            RESIDENCY_QUALIFICATION_PROMPT_TOKEN_COUNT,
        );

        eprintln!("[automatic-paging 1/5] status=progress phase=model_load ETA_seconds=90");
        let engine_load_result = qwen3_5_engine
            .load()
            .await
            .expect("the constrained model should load through demand paging");
        assert_eq!(
            engine_load_result.expert_memory_mode(),
            Some(ExpertMemoryMode::Paged),
            "readiness metadata must report demand paging before the first request"
        );
        assert_eq!(
            qwen3_5_engine
                .expert_memory_mode_for_tests()
                .await
                .expect("the loaded model should expose its expert mode"),
            Some(ExpertMemoryMode::Paged),
            "a complete sparse model that does not fit idle MLX memory must remain paged"
        );
        let native_cache_statistics_before_request = qwen3_5_engine
            .expert_weight_memory_cache_statistics_for_tests()
            .await
            .expect("the paged model should expose expert statistics");

        let generation_finalization = serve_romeo_and_juliet_request(
            &mut qwen3_5_engine,
            request_id,
            prompt_token_ids,
            image_pad_token_id,
            "automatic-paging",
        )
        .await;
        let native_cache_statistics_after_request = qwen3_5_engine
            .expert_weight_memory_cache_statistics_for_tests()
            .await
            .expect("the paged model should retain expert statistics");

        eprintln!("[automatic-paging 4/5] status=progress phase=paging_assertion");
        assert_eq!(
            generation_finalization.expert_memory_mode(),
            Some(ExpertMemoryMode::Paged),
            "a constrained sparse model must remain paged after request cleanup"
        );
        assert!(
            native_cache_statistics_after_request.disk_page_load_count
                > native_cache_statistics_before_request.disk_page_load_count,
            "paged inference must demand-load routed expert weights"
        );
        assert!(
            native_cache_statistics_after_request.resident_payload_byte_count
                <= native_cache_statistics_after_request.maximum_resident_payload_byte_count,
            "paged inference must remain within its native expert-cache payload limit"
        );
        eprintln!("[automatic-paging 5/5] status=success");
    })
    .await
    .expect("the automatic-paging regression must finish within 120 seconds");
}

#[tokio::test]
#[ignore = "loads one sparse model and verifies idle lower-and-raise residency transitions"]
async fn should_transition_the_complete_sparse_model_across_live_memory_limits() {
    timeout(Duration::from_secs(120), async {
        initialize_automatic_residency_tracing();
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let mlx_memory_limits =
            crate::common::sample_machine_model_artifact_qualification_mlx_memory_limits().await;
        let raised_mlx_memory_ceiling_bytes =
            u64::try_from(mlx_memory_limits.active_memory_limit_bytes())
                .expect("the qualification memory ceiling should fit u64");
        let paged_request_id = RequestId::new(1_005);
        let model_directory = super::configured_smallest_depth_one_mtp_model_artifact_directory();
        let (
            mut qwen3_5_engine,
            prompt_token_ids,
            image_pad_token_id,
            _context_memory_reservation_bytes,
        ) = construct_automatic_residency_engine(
            model_directory,
            mlx_memory_limits.active_memory_limit_bytes(),
            mlx_memory_limits.allocator_cache_memory_limit_bytes(),
            paged_request_id,
            RESIDENCY_LIFECYCLE_PROMPT_TOKEN_COUNT,
        );

        eprintln!("[live-residency-transition 1/7] status=progress phase=model_load");
        qwen3_5_engine
            .load()
            .await
            .expect("the fitting model should load as resident");
        assert_eq!(
            qwen3_5_engine
                .expert_memory_mode_for_tests()
                .await
                .expect("the loaded model should expose its initial mode"),
            Some(ExpertMemoryMode::Resident)
        );

        eprintln!("[live-residency-transition 2/7] status=progress phase=ceiling_lower");
        qwen3_5_engine
            .update_mlx_memory_limit(CONSTRAINED_MLX_MEMORY_CEILING_BYTES as u64)
            .await
            .expect("the idle model should accept a safe constrained ceiling");
        assert_eq!(
            qwen3_5_engine
                .expert_memory_mode_for_tests()
                .await
                .expect("the lowered model should expose paged mode"),
            Some(ExpertMemoryMode::Paged)
        );
        serve_romeo_and_juliet_request(
            &mut qwen3_5_engine,
            paged_request_id,
            prompt_token_ids,
            image_pad_token_id,
            "live-residency-transition-paged",
        )
        .await;
        let native_cache_statistics_after_paged_request = qwen3_5_engine
            .expert_weight_memory_cache_statistics_for_tests()
            .await
            .expect("the paged request should expose expert statistics");

        eprintln!("[live-residency-transition 4/7] status=progress phase=ceiling_raise");
        qwen3_5_engine
            .update_mlx_memory_limit(raised_mlx_memory_ceiling_bytes)
            .await
            .expect("the idle model should accept the restored fitting ceiling");
        assert_eq!(
            qwen3_5_engine
                .expert_memory_mode_for_tests()
                .await
                .expect("the raised model should expose resident mode"),
            Some(ExpertMemoryMode::Resident)
        );
        let resident_statistics_after_ceiling_raise = qwen3_5_engine
            .expert_weight_memory_cache_statistics_for_tests()
            .await
            .expect("the repromoted model should expose expert statistics");
        assert_eq!(
            resident_statistics_after_ceiling_raise.resident_payload_byte_count,
            resident_statistics_after_ceiling_raise.maximum_resident_payload_byte_count,
            "repromotion must restore the complete resident expert payload"
        );
        assert_eq!(
            resident_statistics_after_ceiling_raise.disk_page_load_count,
            native_cache_statistics_after_paged_request.disk_page_load_count,
            "repromotion must load contiguous resident arrays without consulting the native pager"
        );
        assert_eq!(
            resident_statistics_after_ceiling_raise.disk_batch_load_count,
            native_cache_statistics_after_paged_request.disk_batch_load_count,
            "repromotion must not add native expert-page batches"
        );
        eprintln!("[live-residency-transition 7/7] status=success");
    })
    .await
    .expect("the live residency-transition regression must finish within 120 seconds");
}

#[tokio::test]
#[ignore = "loads one fitting sparse model and verifies pre-request pressure demotion and idle recovery"]
async fn should_page_for_a_large_request_then_recover_complete_idle_residency() {
    timeout(Duration::from_secs(120), async {
        initialize_automatic_residency_tracing();
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let mlx_memory_limits =
            crate::common::sample_machine_model_artifact_qualification_mlx_memory_limits().await;
        let request_id = RequestId::new(1_007);
        let model_directory = super::configured_smallest_depth_one_mtp_model_artifact_directory();
        let (
            mut qwen3_5_engine,
            prompt_token_ids,
            image_pad_token_id,
            context_memory_reservation_bytes,
        ) = construct_automatic_residency_engine(
            model_directory,
            mlx_memory_limits.active_memory_limit_bytes(),
            mlx_memory_limits.allocator_cache_memory_limit_bytes(),
            request_id,
            RESIDENCY_QUALIFICATION_PROMPT_TOKEN_COUNT,
        );
        qwen3_5_engine
            .load()
            .await
            .expect("the fitting model should load as resident");
        let idle_memory_telemetry = qwen3_5_engine
            .collect_mlx_memory_telemetry()
            .await
            .expect("idle model memory telemetry should be available")
            .expect("the loaded model should report idle memory");
        let machine_mlx_memory_ceiling_bytes =
            u64::try_from(mlx_memory_limits.active_memory_limit_bytes())
                .expect("the original memory ceiling should fit u64");
        let request_pressure_headroom_bytes = u64::try_from(context_memory_reservation_bytes / 2)
            .expect("the request context reservation should fit u64");
        assert!(
            request_pressure_headroom_bytes > 0,
            "qualification requires a nonzero request context reservation"
        );
        let request_pressure_ceiling_bytes = idle_memory_telemetry
            .active_memory_bytes
            .checked_add(request_pressure_headroom_bytes)
            .expect("the request-pressure ceiling should fit u64");
        assert!(
            request_pressure_ceiling_bytes < machine_mlx_memory_ceiling_bytes,
            "the request-pressure ceiling must remain below the sampled machine ceiling"
        );
        qwen3_5_engine
            .update_mlx_memory_limit(request_pressure_ceiling_bytes)
            .await
            .expect("the tighter ceiling should retain the idle resident model");
        assert_eq!(
            qwen3_5_engine
                .expert_memory_mode_for_tests()
                .await
                .expect("the tightened model should expose its mode"),
            Some(ExpertMemoryMode::Resident)
        );
        let expert_statistics_before_request = qwen3_5_engine
            .expert_weight_memory_cache_statistics_for_tests()
            .await
            .expect("the resident model should expose expert statistics");

        eprintln!("[request-pressure-residency 2/5] status=progress phase=request_admission");
        qwen3_5_engine
            .start_generation(
                Qwen3_5InferenceRequest::new(request_id, prompt_token_ids, 2)
                    .with_image_pad_token_id(image_pad_token_id),
            )
            .await
            .expect("the large request should fit after complete expert demotion");
        assert_eq!(
            qwen3_5_engine
                .expert_memory_mode_for_tests()
                .await
                .expect("the admitted request should expose its mode"),
            Some(ExpertMemoryMode::Paged),
            "request admission must demote before the first sparse forward"
        );
        let generation_finalization = complete_started_romeo_and_juliet_request(
            &mut qwen3_5_engine,
            request_id,
            "request-pressure-residency",
        )
        .await;
        let expert_statistics_after_request = qwen3_5_engine
            .expert_weight_memory_cache_statistics_for_tests()
            .await
            .expect("the recovered model should expose expert statistics");

        assert_eq!(
            generation_finalization.expert_memory_mode(),
            Some(ExpertMemoryMode::Resident),
            "request cleanup must restore a model that fits at idle"
        );
        assert!(
            expert_statistics_after_request.disk_page_load_count
                > expert_statistics_before_request.disk_page_load_count,
            "the pressure-demoted request must execute through native demand paging"
        );
        eprintln!("[request-pressure-residency 5/5] status=success");
    })
    .await
    .expect("the request-pressure residency regression must finish within 120 seconds");
}
