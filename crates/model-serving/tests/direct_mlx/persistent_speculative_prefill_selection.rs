use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use astronomical_model_serving::{
    PersistentPromptCacheBlockKey, PersistentPromptCacheDiskStore,
    PersistentPromptCacheDiskStoreConfig, PersistentPromptCacheModelContract,
    PersistentSpeculativePrefillSelectionContract, PersistentVisualEmbeddingKey,
};
use astronomical_runtime_integration::MlxDtype;

use crate::common::qwen3_5_moe::persistent_prompt_cache_model_contract;
use crate::direct_mlx::persistent_prompt_cache_disk_store_support::{
    block_tokens_for_seed, runtime_with_shared_limits, synthetic_kv_block_tensors,
    synthetic_recurrent_snapshot_tensors,
};

#[test]
fn should_reuse_drafter_selection_after_switching_to_another_drafter_and_back() {
    let temporary_cache_root_directory = tempfile::tempdir().expect("the test cache root exists");
    let runtime = runtime_with_shared_limits();
    let prompt_token_ids = vec![11_u32, 12, 13, 14];
    let selected_token_positions_on_gpu = runtime
        .array_from_u32(&[0, 2, 3], &[3])
        .expect("the selection tensor should be created");
    let shared_decoder_cache_layout = persistent_prompt_cache_model_contract()
        .decoder_cache_layout()
        .clone();
    let drafter_a_model_contract = resolve_model_contract(
        "drafter-a",
        "revision-a",
        shared_decoder_cache_layout.clone(),
    );
    let drafter_b_model_contract =
        resolve_model_contract("drafter-b", "revision-b", shared_decoder_cache_layout);
    let drafter_a_selection_contract = selection_contract("drafter-a", "revision-a");
    let drafter_b_selection_contract = selection_contract("drafter-b", "revision-b");

    let drafter_a_cache_store = open_drafter_cache_store(
        temporary_cache_root_directory.path(),
        &drafter_a_model_contract,
    );
    drafter_a_cache_store
        .save_speculative_prefill_selection(
            &runtime,
            &drafter_a_selection_contract,
            &prompt_token_ids,
            &selected_token_positions_on_gpu,
        )
        .expect("drafter A selection should be saved");
    drop(drafter_a_cache_store);

    let drafter_b_cache_store = open_drafter_cache_store(
        temporary_cache_root_directory.path(),
        &drafter_b_model_contract,
    );
    assert!(
        drafter_b_cache_store
            .load_speculative_prefill_selection(
                &runtime,
                &drafter_b_selection_contract,
                &prompt_token_ids,
                None,
            )
            .expect("drafter B lookup should be cache-local")
            .is_none()
    );
    drop(drafter_b_cache_store);

    let drafter_a_cache_store = open_drafter_cache_store(
        temporary_cache_root_directory.path(),
        &drafter_a_model_contract,
    );
    let restored_selection = drafter_a_cache_store
        .load_speculative_prefill_selection(
            &runtime,
            &drafter_a_selection_contract,
            &prompt_token_ids,
            None,
        )
        .expect("drafter A selection should load after switching back")
        .expect("drafter A selection should remain on SSD");

    assert_eq!(restored_selection.dtype(), MlxDtype::UInt32);
    assert_eq!(
        runtime
            .copy_u32_values(&restored_selection)
            .expect("the restored selection should be readable"),
        vec![0, 2, 3]
    );
}

#[test]
fn should_purge_only_obsolete_keep_percentage_selections_for_the_active_pairing() {
    let temporary_cache_root_directory = tempfile::tempdir().expect("the test cache root exists");
    let runtime = runtime_with_shared_limits();
    let prompt_token_ids = vec![11_u32, 12, 13, 14];
    let selected_token_positions_on_gpu = runtime
        .array_from_u32(&[0, 2, 3], &[3])
        .expect("the selection tensor should be created");
    let drafter_model_contract = resolve_model_contract(
        "drafter-a",
        "revision-a",
        persistent_prompt_cache_model_contract()
            .decoder_cache_layout()
            .clone(),
    );
    let drafter_cache_store = open_drafter_cache_store(
        temporary_cache_root_directory.path(),
        &drafter_model_contract,
    );
    let obsolete_active_pairing_contract = selection_contract_for_policy(
        "target-model",
        "target-revision",
        "drafter-a",
        "revision-a",
        20,
    );
    let preserved_unrelated_pairing_contract = selection_contract_for_policy(
        "other-target",
        "target-revision",
        "drafter-a",
        "revision-a",
        20,
    );
    for selection_contract in [
        &obsolete_active_pairing_contract,
        &preserved_unrelated_pairing_contract,
    ] {
        drafter_cache_store
            .save_speculative_prefill_selection(
                &runtime,
                selection_contract,
                &prompt_token_ids,
                &selected_token_positions_on_gpu,
            )
            .expect("the policy-specific selection should be saved");
    }
    let dense_drafter_prompt_state_block_key = PersistentPromptCacheBlockKey::for_root_block(
        &drafter_model_contract,
        &block_tokens_for_seed(0),
    )
    .expect("the dense drafter prompt-state identity should be valid");
    drafter_cache_store
        .save_kv_block_and_recurrent_snapshot(
            &runtime,
            &dense_drafter_prompt_state_block_key,
            None,
            &synthetic_kv_block_tensors(&runtime),
            &synthetic_recurrent_snapshot_tensors(&runtime),
        )
        .expect("the dense drafter prompt state should save before the policy purge");
    let visual_embedding_key = PersistentVisualEmbeddingKey::for_image(
        [9_u8; 32],
        drafter_model_contract.model_id(),
        drafter_model_contract.model_revision(),
    );
    let visual_embeddings = runtime
        .zeros(&[2, 2_048], MlxDtype::BFloat16)
        .expect("the persistent visual embedding fixture should allocate");
    drafter_cache_store
        .save_visual_embedding(&runtime, &visual_embedding_key, &visual_embeddings)
        .expect("the persistent visual embedding should save before the policy purge");

    let active_policy_identity = selection_contract_for_policy(
        "target-model",
        "target-revision",
        "drafter-a",
        "revision-a",
        40,
    )
    .policy_identity();
    let purge_outcome = drafter_cache_store
        .purge_obsolete_speculative_prefill_keep_percentage_entries(&active_policy_identity)
        .expect("the targeted keep-percentage purge should succeed");

    assert_eq!(purge_outcome.speculative_prefill_selection_count, 1);
    assert_eq!(purge_outcome.speculative_prefill_target_state_count, 0);
    assert_eq!(drafter_cache_store.sequence_state_block_count(), 1);
    assert_eq!(drafter_cache_store.boundary_state_snapshot_count(), 1);
    assert_eq!(drafter_cache_store.visual_embedding_count(), 1);
    assert!(
        drafter_cache_store.has_visual_embedding(&visual_embedding_key.visual_embedding_hash())
    );
    assert!(
        drafter_cache_store
            .load_speculative_prefill_selection(
                &runtime,
                &obsolete_active_pairing_contract,
                &prompt_token_ids,
                None,
            )
            .expect("the purged selection lookup should remain valid")
            .is_none()
    );
    assert!(
        drafter_cache_store
            .load_speculative_prefill_selection(
                &runtime,
                &preserved_unrelated_pairing_contract,
                &prompt_token_ids,
                None,
            )
            .expect("the unrelated selection lookup should remain valid")
            .is_some()
    );
}

#[test]
fn should_report_a_required_keep_percentage_purge_deletion_failure() {
    let temporary_cache_root_directory = tempfile::tempdir().expect("the test cache root exists");
    let runtime = runtime_with_shared_limits();
    let prompt_token_ids = vec![11_u32, 12, 13, 14];
    let selected_token_positions_on_gpu = runtime
        .array_from_u32(&[0, 2, 3], &[3])
        .expect("the selection tensor should be created");
    let drafter_model_contract = resolve_model_contract(
        "drafter-a",
        "revision-a",
        persistent_prompt_cache_model_contract()
            .decoder_cache_layout()
            .clone(),
    );
    let drafter_cache_store = open_drafter_cache_store(
        temporary_cache_root_directory.path(),
        &drafter_model_contract,
    );
    let obsolete_selection_contract = selection_contract_for_policy(
        "target-model",
        "target-revision",
        "drafter-a",
        "revision-a",
        20,
    );
    drafter_cache_store
        .save_speculative_prefill_selection(
            &runtime,
            &obsolete_selection_contract,
            &prompt_token_ids,
            &selected_token_positions_on_gpu,
        )
        .expect("the obsolete selection should save before forcing deletion failure");
    let selection_directory = temporary_cache_root_directory
        .path()
        .join("drafter-a")
        .join("speculative_prefill_selections");
    std::fs::set_permissions(&selection_directory, std::fs::Permissions::from_mode(0o500))
        .expect("the test should make the selection directory read-only");
    let active_policy_identity = selection_contract_for_policy(
        "target-model",
        "target-revision",
        "drafter-a",
        "revision-a",
        40,
    )
    .policy_identity();

    let purge_result = drafter_cache_store
        .purge_obsolete_speculative_prefill_keep_percentage_entries(&active_policy_identity);
    std::fs::set_permissions(&selection_directory, std::fs::Permissions::from_mode(0o700))
        .expect("the test should restore selection-directory cleanup permissions");

    let purge_error = purge_result.expect_err("a required deletion failure must stop the purge");
    assert!(purge_error.to_string().contains("remove"));
    assert!(
        drafter_cache_store
            .load_speculative_prefill_selection(
                &runtime,
                &obsolete_selection_contract,
                &prompt_token_ids,
                None,
            )
            .expect("the selection lookup should remain valid after failed deletion")
            .is_some(),
        "a failed required deletion must leave the previously valid selection readable",
    );
}

fn open_drafter_cache_store(
    global_prompt_cache_root_directory: &Path,
    drafter_model_contract: &PersistentPromptCacheModelContract,
) -> PersistentPromptCacheDiskStore {
    PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            global_prompt_cache_root_directory.join(drafter_model_contract.model_id()),
            global_prompt_cache_root_directory.to_path_buf(),
            10_000_000_000,
        ),
        drafter_model_contract.clone(),
    )
    .expect("the drafter cache store should open")
}

fn selection_contract(
    drafter_model_id: &str,
    drafter_model_revision: &str,
) -> PersistentSpeculativePrefillSelectionContract {
    selection_contract_for_policy(
        "target-model",
        "target-revision",
        drafter_model_id,
        drafter_model_revision,
        20,
    )
}

fn selection_contract_for_policy(
    target_model_id: &str,
    target_model_revision: &str,
    drafter_model_id: &str,
    drafter_model_revision: &str,
    keep_percentage: u32,
) -> PersistentSpeculativePrefillSelectionContract {
    PersistentSpeculativePrefillSelectionContract::new(
        target_model_id.to_owned(),
        target_model_revision.to_owned(),
        drafter_model_id.to_owned(),
        drafter_model_revision.to_owned(),
        [3_u8; 32],
        keep_percentage,
        32,
        2,
        2,
        3,
        0,
        4,
    )
}

fn resolve_model_contract(
    model_id: &str,
    model_revision: &str,
    decoder_cache_layout: astronomical_model_serving::DecoderCacheLayout,
) -> PersistentPromptCacheModelContract {
    PersistentPromptCacheModelContract::resolve(
        model_id.to_owned(),
        model_revision.to_owned(),
        decoder_cache_layout,
        16_384,
        20_000_000_000,
        10_000_000_000,
    )
    .expect("the drafter storage contract should resolve")
}
