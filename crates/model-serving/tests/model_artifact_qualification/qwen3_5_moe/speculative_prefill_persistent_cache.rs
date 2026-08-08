use std::time::Duration;

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::{PersistentPromptCacheDiskStoreConfig, Qwen3_5ArtifactValidator};

use super::speculative_prefill::{
    RepresentativeGenerationMeasurement, SPECULATIVE_PREFILL_KEEP_PERCENTAGE,
    prepare_representative_prompt, run_representative_generation,
};

const REPRESENTATIVE_SOURCE_PROMPT_TOKEN_COUNT: usize = 8_192;
const INITIAL_CHAT_PROMPT_TOKEN_COUNT: usize = 40_960;
const FOLLOW_UP_MESSAGE_TOKEN_COUNT: usize = 10_240;

#[tokio::test]
#[ignore = "qualifies the complete 40K chat, SSD reuse, and memory lifecycle under the configured MLX ceiling"]
async fn should_reuse_target_and_drafter_ssd_state_for_a_long_chat_follow_up() {
    tokio::time::timeout(Duration::from_secs(115), async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let target_model_directory = crate::common::configured_ornith_model_artifact_directory();
        let (draft_model_directory, draft_model_id) =
            super::configured_speculative_prefill_draft_model_artifact(&target_model_directory);
        let representative_source_prompt = prepare_representative_prompt(&target_model_directory);
        assert_eq!(
            representative_source_prompt.prompt_token_ids.len(),
            REPRESENTATIVE_SOURCE_PROMPT_TOKEN_COUNT,
        );
        let initial_prompt = super::speculative_prefill::RepresentativePrompt {
            prompt_token_ids: representative_source_prompt.prompt_token_ids.repeat(5),
            image_pad_token_id: representative_source_prompt.image_pad_token_id,
            processed_visual_images: Vec::new(),
            ordinary_target_prefill_control_span_token_count: 0,
            sampling_temperature_thousandths: 0,
            sampling_top_p_thousandths: 1_000,
            sampling_seed: None,
        };
        assert_eq!(initial_prompt.prompt_token_ids.len(), INITIAL_CHAT_PROMPT_TOKEN_COUNT);
        let persistent_prompt_cache_root_directory =
            tempfile::tempdir().expect("the qualification should create a shared SSD cache root");
        let target_persistent_prompt_cache_directory =
            persistent_prompt_cache_root_directory.path().join("target");
        let prompt_cache_maximum_size_bytes =
            crate::common::configured_model_artifact_prompt_cache_maximum_size_bytes();
        let persistent_prompt_cache_disk_store_config = PersistentPromptCacheDiskStoreConfig::new(
            target_persistent_prompt_cache_directory.clone(),
            persistent_prompt_cache_root_directory.path().to_path_buf(),
            prompt_cache_maximum_size_bytes,
        );
        let mlx_memory_limits =
            crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;

        eprintln!(
            "[speculative-prefill-drafter-cache] status=progress phase=cold_drafter_population prompt_tokens={} ETA_seconds=115",
            initial_prompt.prompt_token_ids.len(),
        );
        let cold_measurement = run_representative_generation(
            &target_model_directory,
            &draft_model_directory,
            &draft_model_id,
            &initial_prompt,
            true,
            1,
            SPECULATIVE_PREFILL_KEEP_PERCENTAGE,
            RequestId::new(95_200),
            Some(persistent_prompt_cache_disk_store_config.clone()),
            mlx_memory_limits,
        )
        .await;
        assert_long_chat_cold_session_publishes_sparse_target_state(&cold_measurement);

        let mut follow_up_prompt_token_ids = initial_prompt.prompt_token_ids.clone();
        follow_up_prompt_token_ids.extend_from_slice(
            &representative_source_prompt.prompt_token_ids[..FOLLOW_UP_MESSAGE_TOKEN_COUNT.min(
                representative_source_prompt.prompt_token_ids.len(),
            )],
        );
        if follow_up_prompt_token_ids.len() < INITIAL_CHAT_PROMPT_TOKEN_COUNT + FOLLOW_UP_MESSAGE_TOKEN_COUNT {
            let missing_follow_up_token_count = INITIAL_CHAT_PROMPT_TOKEN_COUNT
                + FOLLOW_UP_MESSAGE_TOKEN_COUNT
                - follow_up_prompt_token_ids.len();
            follow_up_prompt_token_ids.extend_from_slice(
                &representative_source_prompt.prompt_token_ids[..missing_follow_up_token_count],
            );
        }
        let follow_up_prompt = super::speculative_prefill::RepresentativePrompt {
            prompt_token_ids: follow_up_prompt_token_ids,
            image_pad_token_id: representative_source_prompt.image_pad_token_id,
            processed_visual_images: Vec::new(),
            ordinary_target_prefill_control_span_token_count: 0,
            sampling_temperature_thousandths: 0,
            sampling_top_p_thousandths: 1_000,
            sampling_seed: None,
        };
        let target_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&target_model_directory, 20_480)
            .expect("the target artifact should validate for continuation qualification");
        assert!(
            follow_up_prompt.prompt_token_ids.len()
                < target_artifact.config().maximum_position_count() as usize,
        );
        eprintln!(
            "[speculative-prefill-drafter-cache] status=progress phase=independent_drafter_prefix_restart continuation_prompt_tokens={}"
            , follow_up_prompt.prompt_token_ids.len()
        );
        let follow_up_measurement = run_representative_generation(
            &target_model_directory,
            &draft_model_directory,
            &draft_model_id,
            &follow_up_prompt,
            true,
            1,
            SPECULATIVE_PREFILL_KEEP_PERCENTAGE,
            RequestId::new(95_201),
            Some(persistent_prompt_cache_disk_store_config),
            mlx_memory_limits,
        )
        .await;

        assert_long_chat_follow_up_session_preserves_speculative_prefill_rollback_state(
            &follow_up_measurement,
        );
        eprintln!(
            "[speculative-prefill-chat-journey] status=success target_restored_tokens={} drafter_restored_tokens={} drafter_suffix_tokens={}"
            , follow_up_measurement.speculative_prefill_target_persistent_state_restored_token_count
            , follow_up_measurement.speculative_prefill_draft_persistent_prefix_restored_token_count
            , follow_up_measurement.speculative_prefill_draft_scored_suffix_token_count
        );
    })
    .await
    .expect("the drafter SSD qualification should finish within 115 seconds");
}

fn assert_long_chat_follow_up_session_preserves_speculative_prefill_rollback_state(
    follow_up_measurement: &RepresentativeGenerationMeasurement,
) {
    const ZERO_THRESHOLD: u64 = 0;
    const MINIMUM_RESTORED_TARGET_STATE_TOKEN_COUNT: u64 = 1;
    const REQUIRED_DRAFTER_PREFIX_RESTORED_TOKEN_THRESHOLD: u64 =
        INITIAL_CHAT_PROMPT_TOKEN_COUNT as u64;
    const REQUIRED_FOLLOW_UP_MESSAGE_SUFFIX_TOKEN_COUNT: u64 = FOLLOW_UP_MESSAGE_TOKEN_COUNT as u64;

    assert_eq!(follow_up_measurement.speculative_prefill_fallback_count, 0);
    assert_eq!(
        follow_up_measurement.restored_target_persistent_prompt_cache_token_count, ZERO_THRESHOLD,
        "sparse target reuse must remain separate from exact dense target cache reuse"
    );
    assert!(
        follow_up_measurement.speculative_prefill_draft_persistent_prefix_restored_token_count
            >= REQUIRED_DRAFTER_PREFIX_RESTORED_TOKEN_THRESHOLD,
        "the follow-up should restore the complete dense drafter prefix"
    );
    assert!(
        follow_up_measurement.speculative_prefill_target_persistent_state_restored_token_count
            >= MINIMUM_RESTORED_TARGET_STATE_TOKEN_COUNT,
        "the follow-up must restore reusable sparse target state"
    );
    assert_eq!(
        follow_up_measurement.speculative_prefill_draft_scored_suffix_token_count,
        REQUIRED_FOLLOW_UP_MESSAGE_SUFFIX_TOKEN_COUNT,
        "the drafter must process only the uncached follow-up suffix"
    );
    if follow_up_measurement.speculative_prefill_context_target_expert_reclaimed_payload_bytes
        > ZERO_THRESHOLD
        || follow_up_measurement.speculative_prefill_draft_target_expert_reclaimed_payload_bytes
            > ZERO_THRESHOLD
    {
        assert!(
            follow_up_measurement.speculative_prefill_target_expert_repopulated_payload_bytes
                >= MINIMUM_RESTORED_TARGET_STATE_TOKEN_COUNT,
            "target experts that were reclaimed for drafting should repopulate before sparse target execution"
        );
    }
    assert!(
        follow_up_measurement.speculative_prefill_request_scoped_draft_release_elapsed_seconds
            > 0.0,
        "the request-scoped drafter must be released before sparse target execution"
    );
}

fn assert_long_chat_cold_session_publishes_sparse_target_state(
    cold_measurement: &RepresentativeGenerationMeasurement,
) {
    const ZERO_THRESHOLD: u64 = 0;
    const MINIMUM_PERSISTENT_STATE_WRITE_COUNT: u64 = 1;

    assert_eq!(
        cold_measurement.speculative_prefill_fallback_count,
        ZERO_THRESHOLD
    );
    assert_eq!(
        cold_measurement.restored_target_persistent_prompt_cache_token_count,
        ZERO_THRESHOLD,
    );
    assert!(
        cold_measurement.speculative_prefill_target_persistent_state_write_count
            >= MINIMUM_PERSISTENT_STATE_WRITE_COUNT,
        "the initial long chat must publish reusable sparse target state"
    );
}
