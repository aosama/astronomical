use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use astronomical_model_serving::{
    PersistentPromptCacheDiskStore, PersistentPromptCacheDiskStoreConfig,
    PersistentPromptCacheModelContract, PersistentSpeculativePrefillTargetStateContract,
};
use astronomical_runtime_integration::MlxRuntime;

use crate::common::qwen3_5_moe::persistent_prompt_cache_model_contract;
use crate::direct_mlx::persistent_prompt_cache_disk_store_support::runtime_with_shared_limits;

const LONG_INITIAL_PROMPT_TOKEN_COUNT: usize = 40_960;
const LONG_CHAT_FOLLOW_UP_MESSAGE_TOKEN_COUNT: usize = 10_240;
const SPARSE_TARGET_STATE_TEST_TIMEOUT: Duration = Duration::from_secs(115);
const SPARSE_TARGET_STATE_ACTIVE_CACHE_DIRECTORY: &str = "target";
const SPARSE_TARGET_STATES_DIRECTORY: &str = "speculative_prefill_target_states";
// The restore walk only considers prefixes strictly shorter than the load prompt, so the
// load prompt must extend the saved prefix to make the saved state a reachable candidate.
const SPARSE_TARGET_STATE_SAVED_PREFIX_TOKEN_COUNT: u32 = 4_096;
const SPARSE_TARGET_STATE_LOAD_PROMPT_TOKEN_COUNT: u32 = 4_224;

#[test]
fn should_round_trip_the_longest_selection_bound_sparse_target_state_for_a_follow_up() {
    let temporary_cache_root_directory = tempfile::tempdir().expect("the test cache root exists");
    let runtime = runtime_with_shared_limits();
    let target_model_contract = persistent_prompt_cache_model_contract();
    let target_model_id = target_model_contract.model_id().to_owned();
    let target_model_revision = target_model_contract.model_revision().to_owned();
    let target_cache_store_config = PersistentPromptCacheDiskStoreConfig::new(
        temporary_cache_root_directory.path().join("target"),
        temporary_cache_root_directory.path().to_path_buf(),
        100_000_000,
    );
    let target_cache_store = PersistentPromptCacheDiskStore::open(
        target_cache_store_config.clone(),
        target_model_contract.clone(),
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
    drop(target_cache_store);
    let target_cache_store =
        PersistentPromptCacheDiskStore::open(target_cache_store_config, target_model_contract)
            .expect("the target cache store should reopen after a process boundary");

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

#[test]
fn should_purge_only_obsolete_sparse_target_state_for_the_active_pairing() {
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
    let obsolete_active_pairing_contract = target_state_contract_for_policy(
        &target_model_id,
        &target_model_revision,
        "drafter-model",
        "drafter-revision",
        20,
    );
    let preserved_unrelated_pairing_contract = target_state_contract_for_policy(
        &target_model_id,
        &target_model_revision,
        "other-drafter",
        "drafter-revision",
        20,
    );
    let obsolete_prompt_token_ids = vec![10_u32, 11, 12, 13];
    let preserved_prompt_token_ids = vec![20_u32, 21, 22, 23];
    let selected_target_token_positions = runtime
        .array_from_u32(&[0, 2, 3], &[3])
        .expect("the selected target positions should upload");
    let target_decoder_state_tensor = runtime
        .array_from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])
        .expect("the target decoder state fixture should upload");
    for (target_state_contract, prompt_token_ids) in [
        (
            &obsolete_active_pairing_contract,
            &obsolete_prompt_token_ids,
        ),
        (
            &preserved_unrelated_pairing_contract,
            &preserved_prompt_token_ids,
        ),
    ] {
        target_cache_store
            .save_speculative_prefill_target_state(
                &runtime,
                target_state_contract,
                prompt_token_ids,
                &[],
                &selected_target_token_positions,
                &[("layer_0_attention.keys", &target_decoder_state_tensor)],
            )
            .expect("the policy-specific sparse target state should save");
    }

    let active_policy_identity = target_state_contract_for_policy(
        &target_model_id,
        &target_model_revision,
        "drafter-model",
        "drafter-revision",
        40,
    )
    .policy_identity();
    let purge_outcome = target_cache_store
        .purge_obsolete_speculative_prefill_keep_percentage_entries(&active_policy_identity)
        .expect("the sparse target keep-percentage purge should succeed");

    assert_eq!(purge_outcome.speculative_prefill_selection_count, 0);
    assert_eq!(purge_outcome.speculative_prefill_target_state_count, 1);
    assert!(
        target_cache_store
            .load_longest_speculative_prefill_target_state(
                &runtime,
                &obsolete_active_pairing_contract,
                &[10, 11, 12, 13, 14],
                &[],
                None,
            )
            .expect("the purged sparse target lookup should remain valid")
            .is_none()
    );
    assert!(
        target_cache_store
            .load_longest_speculative_prefill_target_state(
                &runtime,
                &preserved_unrelated_pairing_contract,
                &[20, 21, 22, 23, 24],
                &[],
                None,
            )
            .expect("the unrelated sparse target lookup should remain valid")
            .is_some()
    );
}

fn target_state_contract_for_policy(
    target_model_id: &str,
    target_model_revision: &str,
    drafter_model_id: &str,
    drafter_model_revision: &str,
    keep_percentage: u32,
) -> PersistentSpeculativePrefillTargetStateContract {
    PersistentSpeculativePrefillTargetStateContract::new(
        target_model_id.to_owned(),
        target_model_revision.to_owned(),
        drafter_model_id.to_owned(),
        drafter_model_revision.to_owned(),
        [7_u8; 32],
        keep_percentage,
        32,
        512,
        8,
        13,
    )
}

// Regression coverage for issue 617: the sparse target state is restored through the
// in-memory index without re-probing the disk, so a file removed out-of-band while the
// worker runs surfaces as an open failure. Restore must treat that as a lookup miss so
// the request falls back to a cold prefill instead of dying.

#[tokio::test]
async fn should_treat_a_vanished_sparse_target_state_file_as_a_lookup_miss() {
    tokio::time::timeout(SPARSE_TARGET_STATE_TEST_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let temporary_cache_root_directory =
            tempfile::tempdir().expect("the test cache root exists");
        let runtime = runtime_with_shared_limits();
        let target_model_contract = persistent_prompt_cache_model_contract();
        let target_cache_store = open_sparse_target_state_store(
            &temporary_cache_root_directory,
            &target_model_contract,
        );
        let target_state_contract = target_state_contract_for_policy(
            target_model_contract.model_id(),
            target_model_contract.model_revision(),
            "drafter-model",
            "drafter-revision",
            20,
        );
        let saved_prompt_token_ids: Vec<u32> =
            (0..SPARSE_TARGET_STATE_SAVED_PREFIX_TOKEN_COUNT).collect();
        let load_prompt_token_ids: Vec<u32> =
            (0..SPARSE_TARGET_STATE_LOAD_PROMPT_TOKEN_COUNT).collect();
        let ordered_image_sha256_digests = [[3_u8; 32]];
        save_sparse_target_state(
            &target_cache_store,
            &runtime,
            &target_state_contract,
            &saved_prompt_token_ids,
            &ordered_image_sha256_digests,
        );

        let vanished_target_state_file =
            sole_sparse_target_state_file(&temporary_cache_root_directory);
        fs::remove_file(&vanished_target_state_file)
            .expect("the test should remove the target state file out-of-band");

        let restored_target_state = target_cache_store
            .load_longest_speculative_prefill_target_state(
                &runtime,
                &target_state_contract,
                &load_prompt_token_ids,
                &ordered_image_sha256_digests,
                None,
            )
            .expect("a vanished sparse target state file must be a lookup miss, not an error");

        assert!(
            restored_target_state.is_none(),
            "a vanished sparse target state file must restore as a miss so the request can fall back to a cold prefill"
        );
    })
    .await
    .expect("the vanished-file miss lookup should finish within the test timeout");
}

#[tokio::test]
async fn should_treat_a_corrupted_sparse_target_state_file_as_a_lookup_miss_and_remove_it() {
    tokio::time::timeout(SPARSE_TARGET_STATE_TEST_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let temporary_cache_root_directory =
            tempfile::tempdir().expect("the test cache root exists");
        let runtime = runtime_with_shared_limits();
        let target_model_contract = persistent_prompt_cache_model_contract();
        let target_cache_store =
            open_sparse_target_state_store(&temporary_cache_root_directory, &target_model_contract);
        let target_state_contract = target_state_contract_for_policy(
            target_model_contract.model_id(),
            target_model_contract.model_revision(),
            "drafter-model",
            "drafter-revision",
            20,
        );
        let saved_prompt_token_ids: Vec<u32> =
            (0..SPARSE_TARGET_STATE_SAVED_PREFIX_TOKEN_COUNT).collect();
        let load_prompt_token_ids: Vec<u32> =
            (0..SPARSE_TARGET_STATE_LOAD_PROMPT_TOKEN_COUNT).collect();
        let ordered_image_sha256_digests = [[3_u8; 32]];
        save_sparse_target_state(
            &target_cache_store,
            &runtime,
            &target_state_contract,
            &saved_prompt_token_ids,
            &ordered_image_sha256_digests,
        );

        let corrupted_target_state_file =
            sole_sparse_target_state_file(&temporary_cache_root_directory);
        fs::write(&corrupted_target_state_file, [0_u8; 4])
            .expect("the test should corrupt the target state header out-of-band");

        let restored_target_state = target_cache_store
            .load_longest_speculative_prefill_target_state(
                &runtime,
                &target_state_contract,
                &load_prompt_token_ids,
                &ordered_image_sha256_digests,
                None,
            )
            .expect("a corrupted sparse target state file must be a lookup miss, not an error");

        assert!(
            restored_target_state.is_none(),
            "a corrupted sparse target state file must restore as a miss"
        );
        assert!(
            !corrupted_target_state_file.exists(),
            "the corrupted target state file must be removed from the cache"
        );
    })
    .await
    .expect("the corrupted-file miss lookup should finish within the test timeout");
}

#[tokio::test]
async fn should_fail_closed_when_a_sparse_target_state_payload_is_corrupted() {
    tokio::time::timeout(SPARSE_TARGET_STATE_TEST_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let temporary_cache_root_directory =
            tempfile::tempdir().expect("the test cache root exists");
        let runtime = runtime_with_shared_limits();
        let target_model_contract = persistent_prompt_cache_model_contract();
        let target_cache_store = open_sparse_target_state_store(
            &temporary_cache_root_directory,
            &target_model_contract,
        );
        let target_state_contract = target_state_contract_for_policy(
            target_model_contract.model_id(),
            target_model_contract.model_revision(),
            "drafter-model",
            "drafter-revision",
            20,
        );
        let saved_prompt_token_ids: Vec<u32> =
            (0..SPARSE_TARGET_STATE_SAVED_PREFIX_TOKEN_COUNT).collect();
        let load_prompt_token_ids: Vec<u32> =
            (0..SPARSE_TARGET_STATE_LOAD_PROMPT_TOKEN_COUNT).collect();
        let ordered_image_sha256_digests = [[3_u8; 32]];
        save_sparse_target_state(
            &target_cache_store,
            &runtime,
            &target_state_contract,
            &saved_prompt_token_ids,
            &ordered_image_sha256_digests,
        );

        truncate_sparse_target_state_payload(
            &sole_sparse_target_state_file(&temporary_cache_root_directory),
        );

        let load_outcome = target_cache_store
            .load_longest_speculative_prefill_target_state(
                &runtime,
                &target_state_contract,
                &load_prompt_token_ids,
                &ordered_image_sha256_digests,
                None,
            );

        assert!(
            load_outcome.is_err(),
            "a corrupted sparse target state payload must fail closed rather than restore partial state"
        );
    })
    .await
    .expect("the corrupted-payload fail-closed lookup should finish within the test timeout");
}

fn open_sparse_target_state_store(
    temporary_cache_root_directory: &tempfile::TempDir,
    target_model_contract: &PersistentPromptCacheModelContract,
) -> PersistentPromptCacheDiskStore {
    PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            temporary_cache_root_directory
                .path()
                .join(SPARSE_TARGET_STATE_ACTIVE_CACHE_DIRECTORY),
            temporary_cache_root_directory.path().to_path_buf(),
            100_000_000,
        ),
        target_model_contract.clone(),
    )
    .expect("the target cache store should open")
}

fn save_sparse_target_state(
    target_cache_store: &PersistentPromptCacheDiskStore,
    runtime: &MlxRuntime,
    target_state_contract: &PersistentSpeculativePrefillTargetStateContract,
    prompt_token_ids: &[u32],
    ordered_image_sha256_digests: &[[u8; 32]],
) {
    let selected_target_token_positions = runtime
        .array_from_u32(&[0, 32, 8_191], &[3])
        .expect("the selected target positions should upload");
    let target_decoder_state_tensor = runtime
        .array_from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])
        .expect("the target decoder state fixture should upload");
    target_cache_store
        .save_speculative_prefill_target_state(
            runtime,
            target_state_contract,
            prompt_token_ids,
            ordered_image_sha256_digests,
            &selected_target_token_positions,
            &[("layer_0_attention.keys", &target_decoder_state_tensor)],
        )
        .expect("the sparse target state should save");
}

fn sole_sparse_target_state_file(temporary_cache_root_directory: &tempfile::TempDir) -> PathBuf {
    let target_states_directory = sparse_target_states_directory(temporary_cache_root_directory);
    let target_state_files = fs::read_dir(&target_states_directory)
        .expect("the target state directory should exist")
        .map(|directory_entry| {
            directory_entry
                .expect("the target state directory entry should read")
                .path()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        target_state_files.len(),
        1,
        "the test should have saved exactly one sparse target state file"
    );
    target_state_files
        .into_iter()
        .next()
        .expect("one target state file was asserted")
}

fn sparse_target_states_directory(temporary_cache_root_directory: &tempfile::TempDir) -> PathBuf {
    temporary_cache_root_directory
        .path()
        .join(SPARSE_TARGET_STATE_ACTIVE_CACHE_DIRECTORY)
        .join(SPARSE_TARGET_STATES_DIRECTORY)
}

fn truncate_sparse_target_state_payload(target_state_file: &Path) {
    let serialized_target_state_bytes =
        fs::read(target_state_file).expect("the test should read the saved target state");
    let serialized_header_length = u64::from_le_bytes(
        serialized_target_state_bytes[0..8]
            .try_into()
            .expect("the safetensors header length prefix should be present"),
    ) as usize;
    // Keep the full header intact so the model-bound validation passes, then retain one
    // payload byte: the safetensors loader must fail before materializing any tensor.
    let truncated_target_state_bytes =
        &serialized_target_state_bytes[..serialized_header_length + 9];
    fs::write(target_state_file, truncated_target_state_bytes)
        .expect("the test should truncate the target state payload out-of-band");
}
