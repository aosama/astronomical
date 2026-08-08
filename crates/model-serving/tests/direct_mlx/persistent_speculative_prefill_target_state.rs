use astronomical_model_serving::{
    PersistentPromptCacheDiskStore, PersistentPromptCacheDiskStoreConfig,
    PersistentSpeculativePrefillTargetStateContract,
};

use crate::common::qwen3_5_moe::persistent_prompt_cache_model_contract;
use crate::direct_mlx::persistent_prompt_cache_disk_store_support::runtime_with_shared_limits;

const LONG_INITIAL_PROMPT_TOKEN_COUNT: usize = 40_960;
const LONG_CHAT_FOLLOW_UP_MESSAGE_TOKEN_COUNT: usize = 10_240;

#[test]
fn should_round_trip_the_longest_selection_bound_sparse_target_state_for_a_follow_up() {
    let temporary_cache_root_directory = tempfile::tempdir().expect("the test cache root exists");
    let runtime = runtime_with_shared_limits();
    let target_model_contract = persistent_prompt_cache_model_contract();
    let target_model_id = target_model_contract.model_id().to_owned();
    let target_model_revision = target_model_contract.model_revision().to_owned();
    let target_cache_store = PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            temporary_cache_root_directory.path().join("target"),
            temporary_cache_root_directory.path().to_path_buf(),
            100_000_000,
        ),
        target_model_contract,
    )
    .expect("the target cache store should open");
    let target_state_contract = PersistentSpeculativePrefillTargetStateContract::new(
        target_model_id,
        target_model_revision,
        "drafter-model".to_owned(),
        "drafter-revision".to_owned(),
        [7_u8; 32],
        20,
        32,
        512,
        8,
        13,
    );
    let initial_prompt_token_ids = (0..8_192)
        .map(|token_position| token_position as u32)
        .collect::<Vec<_>>();
    let mut follow_up_prompt_token_ids = initial_prompt_token_ids.clone();
    follow_up_prompt_token_ids.extend(8_192_u32..10_240_u32);
    let ordered_image_sha256_digests = [[3_u8; 32]];
    let selected_target_token_positions = runtime
        .array_from_u32(&[0, 32, 8_191], &[3])
        .expect("the selected target positions should upload");
    let target_decoder_state_tensor = runtime
        .array_from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])
        .expect("the target decoder state fixture should upload");

    target_cache_store
        .save_speculative_prefill_target_state(
            &runtime,
            &target_state_contract,
            &initial_prompt_token_ids,
            &ordered_image_sha256_digests,
            &selected_target_token_positions,
            &[("layer_0_attention.keys", &target_decoder_state_tensor)],
        )
        .expect("the sparse target state should save");

    let restored_target_state = target_cache_store
        .load_longest_speculative_prefill_target_state(
            &runtime,
            &target_state_contract,
            &follow_up_prompt_token_ids,
            &ordered_image_sha256_digests,
            None,
        )
        .expect("the sparse target lookup should remain valid")
        .expect("the initial chat should be the longest reusable follow-up prefix");

    assert_eq!(restored_target_state.prompt_prefix_token_count(), 8_192);
    assert_eq!(
        runtime
            .copy_u32_values(restored_target_state.selected_target_token_positions())
            .expect("the selected target positions should be readable"),
        vec![0, 32, 8_191]
    );
    assert_eq!(
        restored_target_state
            .decoder_state_tensors()
            .get("layer_0_attention.keys")
            .expect("the target state tensor should remain named")
            .shape(),
        [1, 1, 2, 2]
    );
    assert!(
        target_cache_store
            .load_longest_speculative_prefill_target_state(
                &runtime,
                &target_state_contract,
                &follow_up_prompt_token_ids,
                &[[9_u8; 32]],
                None,
            )
            .expect("a different image lookup should remain valid")
            .is_none(),
        "different image bytes must never restore sparse target state"
    );
}

#[test]
fn should_round_trip_the_longest_selection_bound_sparse_target_state_for_a_40k_initial_chat_follow_up()
 {
    let temporary_cache_root_directory = tempfile::tempdir().expect("the test cache root exists");
    let runtime = runtime_with_shared_limits();
    let target_model_contract = persistent_prompt_cache_model_contract();
    let target_model_id = target_model_contract.model_id().to_owned();
    let target_model_revision = target_model_contract.model_revision().to_owned();
    let target_cache_store = PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            temporary_cache_root_directory.path().join("target"),
            temporary_cache_root_directory.path().to_path_buf(),
            100_000_000,
        ),
        target_model_contract,
    )
    .expect("the target cache store should open");
    let target_state_contract = PersistentSpeculativePrefillTargetStateContract::new(
        target_model_id,
        target_model_revision,
        "drafter-model".to_owned(),
        "drafter-revision".to_owned(),
        [7_u8; 32],
        20,
        32,
        512,
        8,
        13,
    );
    let initial_prompt_token_ids = (0..LONG_INITIAL_PROMPT_TOKEN_COUNT)
        .map(|token_position| token_position as u32)
        .collect::<Vec<_>>();
    let mut follow_up_prompt_token_ids = initial_prompt_token_ids.clone();
    follow_up_prompt_token_ids.extend(
        LONG_INITIAL_PROMPT_TOKEN_COUNT as u32
            ..(LONG_INITIAL_PROMPT_TOKEN_COUNT + LONG_CHAT_FOLLOW_UP_MESSAGE_TOKEN_COUNT) as u32,
    );
    let ordered_image_sha256_digests = [[3_u8; 32]];
    let selected_target_token_positions = runtime
        .array_from_u32(&[0, 32, 8_191], &[3])
        .expect("the selected target positions should upload");
    let target_decoder_state_tensor = runtime
        .array_from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])
        .expect("the target decoder state fixture should upload");

    target_cache_store
        .save_speculative_prefill_target_state(
            &runtime,
            &target_state_contract,
            &initial_prompt_token_ids,
            &ordered_image_sha256_digests,
            &selected_target_token_positions,
            &[("layer_0_attention.keys", &target_decoder_state_tensor)],
        )
        .expect("the sparse target state should save");

    let restored_target_state = target_cache_store
        .load_longest_speculative_prefill_target_state(
            &runtime,
            &target_state_contract,
            &follow_up_prompt_token_ids,
            &ordered_image_sha256_digests,
            None,
        )
        .expect("the sparse target lookup should remain valid")
        .expect("the long chat follow-up should be the longest reusable prompt prefix");

    assert_eq!(
        restored_target_state.prompt_prefix_token_count(),
        LONG_INITIAL_PROMPT_TOKEN_COUNT,
    );
    assert_eq!(
        runtime
            .copy_u32_values(restored_target_state.selected_target_token_positions())
            .expect("the restored target positions should be readable"),
        vec![0, 32, 8_191]
    );
    assert_eq!(
        restored_target_state
            .decoder_state_tensors()
            .get("layer_0_attention.keys")
            .expect("the target state tensor should remain named")
            .shape(),
        [1, 1, 2, 2],
    );
}
