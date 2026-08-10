//! Direct-MLX recovery journeys for format-11 ancestry and boundary retention.
//!
//! Tests close and reopen the store to exercise the same scan/reconciliation path
//! a user encounters after process restart or an interruption between commit and
//! post-commit retention cleanup.

use std::fs;
use std::fs::FileTimes;
use std::time::{Duration, UNIX_EPOCH};

use super::persistent_prompt_cache_disk_store_support::*;

const LARGE_CACHE_LIMIT_BYTES: u64 = 10 * 1024 * 1024 * 1024;

#[tokio::test]
async fn should_reopen_a_chain_with_one_compacted_non_checkpoint_ancestor() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let root_block_key = persistent_prompt_cache_block_key_for_seed(0);
    let child_block_key = root_block_key
        .for_child_block(&block_tokens_for_seed(10_000))
        .expect("the test should hash the child block");
    let grandchild_block_key = child_block_key
        .for_child_block(&block_tokens_for_seed(20_000))
        .expect("the test should hash the grandchild block");
    let sequence_state_tensors = synthetic_kv_block_tensors(&runtime);
    let boundary_state_tensors = synthetic_recurrent_snapshot_tensors(&runtime);
    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the persistent prompt cache should open");
    save_three_block_chain(
        &persistent_prompt_cache,
        &runtime,
        &root_block_key,
        &child_block_key,
        &grandchild_block_key,
        &sequence_state_tensors,
        &boundary_state_tensors,
    );
    let compacted_child_boundary_file_path =
        block_directory_path(&persistent_prompt_cache_directory, &child_block_key)
            .join("boundary.safetensors");
    fs::remove_file(&compacted_child_boundary_file_path)
        .expect("the test should compact the non-checkpoint child boundary");
    drop(persistent_prompt_cache);

    let reopened_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("a valid compacted ancestor should survive startup reconciliation");

    assert_eq!(reopened_prompt_cache.sequence_state_block_count(), 3);
    assert_eq!(reopened_prompt_cache.boundary_state_snapshot_count(), 2);
    assert!(reopened_prompt_cache.has_kv_block(&child_block_key.block_hash()));
    assert!(!reopened_prompt_cache.has_recurrent_snapshot(&child_block_key.block_hash()));
    assert!(reopened_prompt_cache.has_kv_block(&grandchild_block_key.block_hash()));
    assert!(reopened_prompt_cache.has_recurrent_snapshot(&grandchild_block_key.block_hash()));
}

#[tokio::test]
async fn should_recapture_only_a_compacted_boundary_without_rewriting_sequence_state() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let root_block_key = persistent_prompt_cache_block_key_for_seed(0);
    let child_block_key = root_block_key
        .for_child_block(&block_tokens_for_seed(10_000))
        .expect("the test should hash the child block");
    let grandchild_block_key = child_block_key
        .for_child_block(&block_tokens_for_seed(20_000))
        .expect("the test should hash the grandchild block");
    let sequence_state_tensors = synthetic_kv_block_tensors(&runtime);
    let boundary_state_tensors = synthetic_recurrent_snapshot_tensors(&runtime);
    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the persistent prompt cache should open");
    save_three_block_chain(
        &persistent_prompt_cache,
        &runtime,
        &root_block_key,
        &child_block_key,
        &grandchild_block_key,
        &sequence_state_tensors,
        &boundary_state_tensors,
    );
    let child_block_directory =
        block_directory_path(&persistent_prompt_cache_directory, &child_block_key);
    let child_sequence_file_path = child_block_directory.join("sequence.safetensors");
    let child_boundary_file_path = child_block_directory.join("boundary.safetensors");
    fs::remove_file(&child_boundary_file_path).expect("the test should compact the child boundary");
    fs::File::open(&child_sequence_file_path)
        .expect("the test should open the child sequence state")
        .set_times(FileTimes::new().set_modified(UNIX_EPOCH + Duration::from_secs(10)))
        .expect("the test should persist a stable sequence-state timestamp");
    drop(persistent_prompt_cache);
    let reopened_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the compacted chain should reopen");

    reopened_prompt_cache
        .publish_block(
            &runtime,
            &child_block_key,
            Some(&root_block_key),
            &sequence_state_tensors,
            &boundary_state_tensors,
        )
        .expect("recapturing the child should restore its missing boundary");

    assert!(child_boundary_file_path.is_file());
    assert_eq!(
        fs::metadata(&child_sequence_file_path)
            .expect("the child sequence state should remain")
            .modified()
            .expect("the test should read the sequence-state timestamp"),
        UNIX_EPOCH + Duration::from_secs(10),
        "recapturing a compacted boundary must not rewrite sequence state"
    );
}

#[tokio::test]
async fn should_complete_interrupted_parent_boundary_compaction_before_startup_eviction() {
    // The disk image represents a crash after child commit but before redundant
    // parent-boundary deletion. Tightening quota on reopen forces recovery order.
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let root_block_key = persistent_prompt_cache_block_key_for_seed(0);
    let child_block_key = root_block_key
        .for_child_block(&block_tokens_for_seed(10_000))
        .expect("the test should hash the child block");
    let grandchild_block_key = child_block_key
        .for_child_block(&block_tokens_for_seed(20_000))
        .expect("the test should hash the grandchild block");
    let sequence_state_tensors = synthetic_kv_block_tensors(&runtime);
    let boundary_state_tensors = synthetic_recurrent_snapshot_tensors(&runtime);
    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the persistent prompt cache should open");
    save_three_block_chain(
        &persistent_prompt_cache,
        &runtime,
        &root_block_key,
        &child_block_key,
        &grandchild_block_key,
        &sequence_state_tensors,
        &boundary_state_tensors,
    );
    let model_contract = persistent_prompt_cache.model_contract_ref().clone();
    let interrupted_transaction_size_bytes = persistent_prompt_cache.total_size_bytes();
    let post_compaction_quota_bytes = interrupted_transaction_size_bytes
        .checked_sub(model_contract.boundary_state_file_bytes())
        .expect("one parent boundary should fit inside the interrupted transaction");
    drop(persistent_prompt_cache);

    let reopened_prompt_cache = open_persistent_prompt_cache_disk_store_with_contract(
        &persistent_prompt_cache_directory,
        post_compaction_quota_bytes,
        model_contract,
    )
    .expect("startup should compact the eligible parent before evicting its chain");

    assert_eq!(reopened_prompt_cache.sequence_state_block_count(), 3);
    assert_eq!(reopened_prompt_cache.boundary_state_snapshot_count(), 2);
    assert!(!reopened_prompt_cache.has_recurrent_snapshot(&child_block_key.block_hash()));
    assert!(reopened_prompt_cache.has_kv_block(&grandchild_block_key.block_hash()));
}

#[tokio::test]
async fn should_evict_unprotected_content_before_compacting_the_active_startup_chain() {
    // Unrelated bytes can satisfy pressure without reducing useful restart points,
    // so they must be selected before any valid active-chain boundary is compacted.
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let root_block_key = persistent_prompt_cache_block_key_for_seed(0);
    let child_block_key = root_block_key
        .for_child_block(&block_tokens_for_seed(10_000))
        .expect("the test should hash the child block");
    let grandchild_block_key = child_block_key
        .for_child_block(&block_tokens_for_seed(20_000))
        .expect("the test should hash the grandchild block");
    let sequence_state_tensors = synthetic_kv_block_tensors(&runtime);
    let boundary_state_tensors = synthetic_recurrent_snapshot_tensors(&runtime);
    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the persistent prompt cache should open");
    save_three_block_chain(
        &persistent_prompt_cache,
        &runtime,
        &root_block_key,
        &child_block_key,
        &grandchild_block_key,
        &sequence_state_tensors,
        &boundary_state_tensors,
    );
    let model_contract = persistent_prompt_cache.model_contract_ref().clone();
    let active_chain_size_bytes = persistent_prompt_cache.total_size_bytes();
    drop(persistent_prompt_cache);
    let unprotected_file_path = persistent_prompt_cache_directory
        .path()
        .join("unprotected.bin");
    fs::write(&unprotected_file_path, vec![0_u8; 4_096])
        .expect("the unprotected file should consume global quota");

    let reopened_prompt_cache = open_persistent_prompt_cache_disk_store_with_contract(
        &persistent_prompt_cache_directory,
        active_chain_size_bytes,
        model_contract,
    )
    .expect("startup should evict unprotected content before active retention state");

    assert!(!unprotected_file_path.exists());
    assert_eq!(reopened_prompt_cache.sequence_state_block_count(), 3);
    assert_eq!(reopened_prompt_cache.boundary_state_snapshot_count(), 3);
}

#[tokio::test]
async fn should_remove_descendants_when_their_sequence_ancestor_is_missing_on_reopen() {
    // A child's content hash does not make it independently restorable. Removing
    // root models corruption/deletion and startup must prune the complete orphan chain.
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let root_block_key = persistent_prompt_cache_block_key_for_seed(0);
    let child_block_key = root_block_key
        .for_child_block(&block_tokens_for_seed(10_000))
        .expect("the test should hash the child block");
    let sequence_state_tensors = synthetic_kv_block_tensors(&runtime);
    let boundary_state_tensors = synthetic_recurrent_snapshot_tensors(&runtime);
    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the persistent prompt cache should open");
    persistent_prompt_cache
        .publish_block(
            &runtime,
            &root_block_key,
            None,
            &sequence_state_tensors,
            &boundary_state_tensors,
        )
        .expect("the root block should save");
    persistent_prompt_cache
        .publish_block(
            &runtime,
            &child_block_key,
            Some(&root_block_key),
            &sequence_state_tensors,
            &boundary_state_tensors,
        )
        .expect("the child block should save");
    drop(persistent_prompt_cache);
    fs::remove_dir_all(block_directory_path(
        &persistent_prompt_cache_directory,
        &root_block_key,
    ))
    .expect("the test should remove the root sequence ancestor");

    let reopened_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("startup reconciliation should prune the orphan subtree");

    assert_eq!(reopened_prompt_cache.sequence_state_block_count(), 0);
    assert_eq!(reopened_prompt_cache.boundary_state_snapshot_count(), 0);
    assert!(!block_directory_path(&persistent_prompt_cache_directory, &child_block_key).exists());
}

#[tokio::test]
async fn should_not_acknowledge_an_existing_block_after_its_state_is_corrupted() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let block_key = persistent_prompt_cache_block_key_for_seed(0);
    let sequence_state_tensors = synthetic_kv_block_tensors(&runtime);
    let boundary_state_tensors = synthetic_recurrent_snapshot_tensors(&runtime);
    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the persistent prompt cache should open");
    persistent_prompt_cache
        .publish_block(
            &runtime,
            &block_key,
            None,
            &sequence_state_tensors,
            &boundary_state_tensors,
        )
        .expect("the initial block should publish");
    fs::write(
        block_directory_path(&persistent_prompt_cache_directory, &block_key)
            .join("sequence.safetensors"),
        b"corrupted state",
    )
    .expect("the test should corrupt the published sequence state");

    let republish_error = persistent_prompt_cache
        .publish_block(
            &runtime,
            &block_key,
            None,
            &sequence_state_tensors,
            &boundary_state_tensors,
        )
        .expect_err("corrupt existing state must not receive idempotent acknowledgement");

    assert!(matches!(
        republish_error,
        astronomical_model_serving::PersistentPromptCacheDiskStoreError::ValidateBlock { .. }
    ));
}

fn save_three_block_chain(
    persistent_prompt_cache: &astronomical_model_serving::PersistentPromptCacheDiskStore,
    runtime: &astronomical_runtime_integration::MlxRuntime,
    root_block_key: &astronomical_model_serving::PersistentPromptCacheBlockKey,
    child_block_key: &astronomical_model_serving::PersistentPromptCacheBlockKey,
    grandchild_block_key: &astronomical_model_serving::PersistentPromptCacheBlockKey,
    sequence_state_tensors: &std::collections::HashMap<
        String,
        astronomical_runtime_integration::MlxArray,
    >,
    boundary_state_tensors: &std::collections::HashMap<
        String,
        astronomical_runtime_integration::MlxArray,
    >,
) {
    for (block_key, parent_block_key) in [
        (root_block_key, None),
        (child_block_key, Some(root_block_key)),
        (grandchild_block_key, Some(child_block_key)),
    ] {
        persistent_prompt_cache
            .publish_block(
                runtime,
                block_key,
                parent_block_key,
                sequence_state_tensors,
                boundary_state_tensors,
            )
            .expect("the test block chain should save");
    }
}

fn block_directory_path(
    persistent_prompt_cache_directory: &tempfile::TempDir,
    block_key: &astronomical_model_serving::PersistentPromptCacheBlockKey,
) -> std::path::PathBuf {
    persistent_prompt_cache_directory
        .path()
        .join("blocks")
        .join(hex::encode(block_key.block_hash()))
}
