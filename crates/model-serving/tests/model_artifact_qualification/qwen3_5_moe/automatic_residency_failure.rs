//! Failure qualification for atomic resident promotion and paged recovery.
//!
//! The test removes only Rust's promotion descriptors after a valid paged load;
//! native C++ retains its independently copied source inventory. A failed ceiling
//! raise must therefore remain truthful and continue serving through paging.

use std::time::Duration;

use astronomical_ipc_protocol::{ExpertMemoryMode, RequestId};
use astronomical_model_serving::{InferenceEngine, InferenceEngineError};
use tokio::time::timeout;

use super::automatic_residency_support::{
    CONSTRAINED_MLX_MEMORY_CEILING_BYTES, RESIDENCY_LIFECYCLE_PROMPT_TOKEN_COUNT,
    construct_automatic_residency_engine, initialize_automatic_residency_tracing,
    serve_romeo_and_juliet_request,
};

#[tokio::test]
#[ignore = "removes resident source descriptors and proves failed promotion retains healthy paging"]
async fn should_resume_native_paging_when_resident_promotion_fails() {
    timeout(Duration::from_secs(120), async {
        initialize_automatic_residency_tracing();
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let machine_memory_limits =
            crate::common::sample_machine_model_artifact_qualification_mlx_memory_limits().await;
        let raised_mlx_memory_ceiling_bytes =
            u64::try_from(machine_memory_limits.active_memory_limit_bytes())
                .expect("the sampled machine ceiling should fit u64");
        let request_id = RequestId::new(1_008);
        let model_directory = crate::common::configured_ornith_model_artifact_directory();
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
            RESIDENCY_LIFECYCLE_PROMPT_TOKEN_COUNT,
        );

        eprintln!("[resident-promotion-failure 1/5] status=progress phase=model_load");
        let engine_load_result = qwen3_5_engine
            .load()
            .await
            .expect("the constrained sparse model should load through paging");
        assert_eq!(
            engine_load_result.expert_memory_mode(),
            Some(ExpertMemoryMode::Paged)
        );
        qwen3_5_engine
            .remove_resident_expert_source_files_for_tests()
            .await
            .expect("qualification should remove only resident-promotion source descriptors");

        eprintln!("[resident-promotion-failure 2/5] status=progress phase=failed_ceiling_raise");
        let promotion_error = qwen3_5_engine
            .update_mlx_memory_limit(raised_mlx_memory_ceiling_bytes)
            .await
            .expect_err("resident promotion must preserve a missing-source failure");
        assert!(
            matches!(promotion_error, InferenceEngineError::Fatal { .. }),
            "a malformed resident source inventory must remain a typed fatal transition error"
        );
        assert_eq!(
            qwen3_5_engine
                .expert_memory_mode_for_tests()
                .await
                .expect("the failed promotion should retain a truthful mode"),
            Some(ExpertMemoryMode::Paged)
        );
        let rollback_adjustment = qwen3_5_engine
            .update_mlx_memory_limit(CONSTRAINED_MLX_MEMORY_CEILING_BYTES as u64)
            .await
            .expect("the failed raise should leave the prior constrained ceiling usable");
        assert_eq!(
            rollback_adjustment.effective_mlx_memory_ceiling_bytes(),
            CONSTRAINED_MLX_MEMORY_CEILING_BYTES as u64,
            "failed resident promotion must restore the prior native and Rust ceilings"
        );
        let pager_statistics_before_request = qwen3_5_engine
            .expert_weight_memory_cache_statistics_for_tests()
            .await
            .expect("the healthy paged fallback should expose expert statistics");

        let generation_finalization = serve_romeo_and_juliet_request(
            &mut qwen3_5_engine,
            request_id,
            prompt_token_ids,
            image_pad_token_id,
            "resident-promotion-failure",
        )
        .await;
        let pager_statistics_after_request = qwen3_5_engine
            .expert_weight_memory_cache_statistics_for_tests()
            .await
            .expect("the paged fallback should retain expert statistics");
        assert_eq!(
            generation_finalization.expert_memory_mode(),
            Some(ExpertMemoryMode::Paged)
        );
        assert!(
            pager_statistics_after_request.disk_page_load_count
                > pager_statistics_before_request.disk_page_load_count,
            "the model must remain usable through demand paging after failed promotion"
        );
        eprintln!("[resident-promotion-failure 5/5] status=success");
    })
    .await
    .expect("the failed-promotion recovery journey must finish within 120 seconds");
}
