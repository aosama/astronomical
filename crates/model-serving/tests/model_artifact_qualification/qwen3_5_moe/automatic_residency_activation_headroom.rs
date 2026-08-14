//! Full-stack residency admission with activation headroom.
//!
//! User journey: load a large sparse checkpoint under a ceiling that still fits
//! the static complete-expert payload, serve Romeo and Juliet, and remain
//! non-resident under paged or hybrid retention because activation headroom
//! rejects complete residency.

use std::time::Duration;

use astronomical_ipc_protocol::{ExpertMemoryMode, RequestId};
use astronomical_model_serving::{
    InferenceEngine, complete_residency_exceeds_ceiling_with_activation_headroom,
    projected_active_memory_after_complete_expert_replacement,
    required_complete_residency_activation_headroom_bytes,
};
use tokio::time::timeout;

use super::automatic_residency_support::{
    RESIDENCY_LIFECYCLE_PROMPT_TOKEN_COUNT, construct_automatic_residency_engine,
    initialize_automatic_residency_tracing, serve_romeo_and_juliet_request,
};

/// Qualification ceiling: static Fable experts fit, activation headroom does not.
const STATIC_FIT_WITHOUT_HEADROOM_CEILING_BYTES: usize = 40_000_000_000;

/// Large sparse checkpoint whose complete experts nearly fill a 40 GB ceiling.
const LARGE_SPARSE_HEADROOM_REGRESSION_MODEL_ID: &str =
    "Qwen3.6-35B-A3B-Fable-Holo3.1-Qwopus-KAT-Coder-C-qx86-hi-mlx";

#[tokio::test]
#[ignore = "loads Fable under a static-fit 40 GB ceiling and proves activation headroom prevents complete residency"]
async fn should_keep_large_sparse_model_paged_when_static_experts_fit_but_activation_headroom_does_not()
 {
    timeout(Duration::from_secs(120), async {
        initialize_automatic_residency_tracing();
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let request_id = RequestId::new(1_041);
        let model_directory = crate::common::configured_model_artifact_directory_by_id(
            LARGE_SPARSE_HEADROOM_REGRESSION_MODEL_ID,
        );
        let (
            mut qwen3_5_engine,
            prompt_token_ids,
            image_pad_token_id,
            _context_memory_reservation_bytes,
        ) = construct_automatic_residency_engine(
            model_directory,
            STATIC_FIT_WITHOUT_HEADROOM_CEILING_BYTES,
            STATIC_FIT_WITHOUT_HEADROOM_CEILING_BYTES,
            request_id,
            RESIDENCY_LIFECYCLE_PROMPT_TOKEN_COUNT,
        );

        eprintln!(
            "[activation-headroom-paging 1/6] status=progress phase=model_load ceiling_bytes={STATIC_FIT_WITHOUT_HEADROOM_CEILING_BYTES} ETA_seconds=90"
        );
        let engine_load_result = qwen3_5_engine
            .load()
            .await
            .expect("the large sparse model should load under the static-fit ceiling");
        assert_eq!(
            engine_load_result.expert_memory_mode(),
            Some(ExpertMemoryMode::Paged),
            "readiness must report demand paging when activation headroom rejects complete residency"
        );
        assert_eq!(
            qwen3_5_engine
                .expert_memory_mode_for_tests()
                .await
                .expect("the loaded model should expose its expert mode"),
            Some(ExpertMemoryMode::Paged),
            "startup admission must keep the model paged under the static-fit ceiling"
        );

        let complete_expert_payload_bytes = qwen3_5_engine
            .complete_expert_payload_byte_count_for_tests()
            .await
            .expect("the sparse model should expose its complete expert payload");
        let minimum_mlx_memory_ceiling_bytes =
            engine_load_result.minimum_mlx_memory_ceiling_bytes();
        let projected_resident_active_memory_bytes =
            projected_active_memory_after_complete_expert_replacement(
                minimum_mlx_memory_ceiling_bytes,
                0,
                complete_expert_payload_bytes,
            )
            .expect("static complete-residency projection should succeed");
        let required_activation_headroom_bytes =
            required_complete_residency_activation_headroom_bytes(complete_expert_payload_bytes, 0);
        let stable_memory_ceiling_bytes =
            u64::try_from(STATIC_FIT_WITHOUT_HEADROOM_CEILING_BYTES)
                .expect("the static-fit ceiling should fit u64");

        eprintln!(
            "[activation-headroom-paging 2/6] status=progress phase=admission_arithmetic projected_resident_bytes={projected_resident_active_memory_bytes} headroom_bytes={required_activation_headroom_bytes} ceiling_bytes={stable_memory_ceiling_bytes}"
        );
        assert!(
            projected_resident_active_memory_bytes <= stable_memory_ceiling_bytes,
            "this qualification requires the static complete payload to fit the ceiling: projected={projected_resident_active_memory_bytes}, ceiling={stable_memory_ceiling_bytes}"
        );
        assert!(
            complete_residency_exceeds_ceiling_with_activation_headroom(
                projected_resident_active_memory_bytes,
                stable_memory_ceiling_bytes,
                required_activation_headroom_bytes,
            ),
            "activation headroom must reject complete residency for this ceiling"
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
            "activation-headroom-paging",
        )
        .await;
        let native_cache_statistics_after_request = qwen3_5_engine
            .expert_weight_memory_cache_statistics_for_tests()
            .await
            .expect("the paged model should retain expert statistics");

        eprintln!(
            "[activation-headroom-paging 5/6] status=progress phase=post_request_assertion"
        );
        // Hybrid mode retains a bounded subset of expert layers while continuing to
        // demand-page the remainder, so both paged modes prove complete residency was rejected.
        assert!(
            matches!(
                generation_finalization.expert_memory_mode(),
                Some(ExpertMemoryMode::Paged | ExpertMemoryMode::Hybrid)
            ),
            "request finalization must not promote complete residency without activation headroom"
        );
        assert!(
            matches!(
                qwen3_5_engine
                    .expert_memory_mode_for_tests()
                    .await
                    .expect("the model should still expose its expert mode after the request"),
                Some(ExpertMemoryMode::Paged | ExpertMemoryMode::Hybrid)
            ),
            "the engine must remain paged or hybrid after the first Romeo and Juliet request"
        );
        assert!(
            native_cache_statistics_after_request.disk_page_load_count
                > native_cache_statistics_before_request.disk_page_load_count,
            "paged inference must demand-load routed expert weights"
        );
        assert!(
            native_cache_statistics_after_request.resident_payload_byte_count
                <= native_cache_statistics_after_request.maximum_resident_payload_byte_count,
            "paged inference must remain within its streamed-expert payload limit"
        );
        eprintln!("[activation-headroom-paging 6/6] status=success");
    })
    .await
    .expect("the activation-headroom paging qualification must finish within 120 seconds");
}
