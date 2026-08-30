//! Direct acceptance of the optional MTP layer inside the complete owner.
//!
//! Ordinary target generation cannot prove that the appended MTP layer uses
//! resident arrays. This focused forward compares Rust pager counters before
//! and after one real draft so an accidental MTP paging fallback is observable.

use std::time::Duration;

use astronomical_ipc_protocol::{ExpertMemoryMode, MtpRuntimeState, RequestId};
use astronomical_model_serving::InferenceEngine;
use tokio::time::timeout;

use super::support::{
    construct_automatic_residency_engine, initialize_automatic_residency_tracing,
};

const RESIDENT_MTP_PROMPT_TOKEN_COUNT: usize = 32;

#[tokio::test]
#[ignore = "loads a fitting depth-one MTP checkpoint and verifies resident MTP-head execution"]
async fn should_execute_depth_one_mtp_with_complete_resident_experts() {
    timeout(Duration::from_secs(120), async {
        initialize_automatic_residency_tracing();
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let mlx_memory_limits = crate::common::
            sample_machine_serving_acceptance_mlx_memory_limits()
            .await;
        let request_id = RequestId::new(1_009);
        let model_directory = crate::serving_acceptance::support::configured_resident_sparse_moe_model_directory();
        let (
            mut qwen3_5_engine,
            prompt_token_ids,
            _image_pad_token_id,
            _context_memory_reservation_bytes,
        ) = construct_automatic_residency_engine(
            model_directory,
            mlx_memory_limits.active_memory_limit_bytes(),
            mlx_memory_limits.allocator_cache_memory_limit_bytes(),
            request_id,
            RESIDENT_MTP_PROMPT_TOKEN_COUNT,
        );

        eprintln!("[resident-mtp 1/5] status=progress phase=model_load");
        let engine_load_result = qwen3_5_engine
            .load()
            .await
            .expect("the fitting depth-one MTP model should load");
        assert_eq!(
            engine_load_result.expert_memory_mode(),
            Some(ExpertMemoryMode::Resident)
        );
        if engine_load_result.mtp_runtime_state() != MtpRuntimeState::Active {
            eprintln!(
                "[resident-mtp] status=success phase=target_only reason=installed_model_has_no_complete_mtp_inventory"
            );
            return;
        }
        let native_pager_statistics_before_mtp_forward = qwen3_5_engine
            .expert_weight_memory_cache_statistics_for_tests()
            .await
            .expect("resident MTP should expose Rust pager statistics");

        let romeo_and_juliet_token_id = prompt_token_ids[prompt_token_ids.len() / 2];
        eprintln!("[resident-mtp 2/5] status=progress phase=mtp_head_forward");
        let draft_token_id = qwen3_5_engine
            .execute_resident_mtp_draft_for_tests(romeo_and_juliet_token_id)
            .await
            .expect("the resident MTP head should produce one draft token");
        eprintln!(
            "[resident-mtp 4/5] status=progress phase=mtp_head_complete draft_token_id={draft_token_id}"
        );
        let native_pager_statistics_after_mtp_forward = qwen3_5_engine
            .expert_weight_memory_cache_statistics_for_tests()
            .await
            .expect("resident MTP should retain Rust pager statistics");

        assert_eq!(
            qwen3_5_engine
                .expert_memory_mode_for_tests()
                .await
                .expect("the MTP forward should preserve truthful expert mode"),
            Some(ExpertMemoryMode::Resident)
        );
        assert_eq!(
            native_pager_statistics_after_mtp_forward,
            native_pager_statistics_before_mtp_forward,
            "resident target and MTP execution must not prepare or load streamed expert pages"
        );
        eprintln!("[resident-mtp 5/5] status=success");
    })
    .await
    .expect("resident depth-one MTP acceptance must finish within 120 seconds");
}
