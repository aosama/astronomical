//! Direct-MLX acceptance tests for the block publication transaction.
//!
//! These scenarios inject failure at different staging points and assert the
//! user-observable invariant: readers see either the old complete chain or the
//! new complete block, never a partially published directory.

use std::{collections::HashMap, fs, path::Path};

use astronomical_model_serving::PersistentPromptCacheDiskStoreError;
use astronomical_runtime_integration::{MlxArray, MlxDtype};
use serde_json::Value;

use super::persistent_prompt_cache_disk_store_support::*;

const LARGE_CACHE_LIMIT_BYTES: u64 = 10 * 1024 * 1024 * 1024;

#[tokio::test]
async fn should_remove_the_complete_staging_transaction_when_sequence_validation_fails() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let cache_directory = tempfile::tempdir().expect("the test should create a cache directory");
    let cache = open_persistent_prompt_cache_disk_store(&cache_directory, LARGE_CACHE_LIMIT_BYTES)
        .expect("the cache should open");
    let mut invalid_sequence_state_tensors = synthetic_kv_block_tensors(&runtime);
    replace_one_tensor_with_invalid_shape(&runtime, &mut invalid_sequence_state_tensors);

    let publication_error = cache
        .publish_block(
            &runtime,
            &persistent_prompt_cache_block_key_for_seed(0),
            None,
            &invalid_sequence_state_tensors,
            &synthetic_recurrent_snapshot_tensors(&runtime),
        )
        .expect_err("invalid sequence state must stop publication");

    assert!(matches!(
        publication_error,
        PersistentPromptCacheDiskStoreError::ValidateBlock { .. }
    ));
    assert_no_visible_or_staged_blocks(cache_directory.path());
}

#[tokio::test]
async fn should_remove_the_sequence_file_and_transaction_when_boundary_validation_fails() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let cache_directory = tempfile::tempdir().expect("the test should create a cache directory");
    let cache = open_persistent_prompt_cache_disk_store(&cache_directory, LARGE_CACHE_LIMIT_BYTES)
        .expect("the cache should open");
    let mut invalid_boundary_state_tensors = synthetic_recurrent_snapshot_tensors(&runtime);
    replace_one_tensor_with_invalid_shape(&runtime, &mut invalid_boundary_state_tensors);

    let publication_error = cache
        .publish_block(
            &runtime,
            &persistent_prompt_cache_block_key_for_seed(0),
            None,
            &synthetic_kv_block_tensors(&runtime),
            &invalid_boundary_state_tensors,
        )
        .expect_err("invalid boundary state must stop publication");

    assert!(matches!(
        publication_error,
        PersistentPromptCacheDiskStoreError::ValidateBlock { .. }
    ));
    assert_no_visible_or_staged_blocks(cache_directory.path());
}

#[tokio::test]
async fn should_leave_the_durable_parent_unchanged_after_a_child_storage_fault() {
    // This is the critical append-only failure journey. The parent was already
    // acknowledged to a previous request, so a child failure must not roll it back.
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let cache_directory = tempfile::tempdir().expect("the test should create a cache directory");
    let cache = open_persistent_prompt_cache_disk_store(&cache_directory, LARGE_CACHE_LIMIT_BYTES)
        .expect("the cache should open");
    let parent_block_key = persistent_prompt_cache_block_key_for_seed(0);
    let child_block_key = parent_block_key
        .for_child_block(&block_tokens_for_seed(10_000))
        .expect("the child identity should resolve");
    let sequence_state_tensors = synthetic_kv_block_tensors(&runtime);
    let boundary_state_tensors = synthetic_recurrent_snapshot_tensors(&runtime);
    cache
        .publish_block(
            &runtime,
            &parent_block_key,
            None,
            &sequence_state_tensors,
            &boundary_state_tensors,
        )
        .expect("the parent should publish");
    let mut invalid_child_boundary_state_tensors = synthetic_recurrent_snapshot_tensors(&runtime);
    replace_one_tensor_with_invalid_shape(&runtime, &mut invalid_child_boundary_state_tensors);

    cache
        .publish_block(
            &runtime,
            &child_block_key,
            Some(&parent_block_key),
            &sequence_state_tensors,
            &invalid_child_boundary_state_tensors,
        )
        .expect_err("the child storage fault must remain a hard error");

    assert!(cache.has_kv_block(&parent_block_key.block_hash()));
    assert!(cache.has_recurrent_snapshot(&parent_block_key.block_hash()));
    assert!(!cache.has_kv_block(&child_block_key.block_hash()));
    assert!(!cache.has_recurrent_snapshot(&child_block_key.block_hash()));
    assert_eq!(cache.sequence_state_block_count(), 1);
    assert_eq!(cache.boundary_state_snapshot_count(), 1);
    assert_no_staging_directories(cache_directory.path());
}

#[tokio::test]
async fn should_remove_staging_when_a_conflicting_final_directory_appears_before_commit() {
    // Simulate an out-of-process filesystem race after this store's startup scan.
    // Publication must preserve unknown final content rather than replacing it.
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let cache_directory = tempfile::tempdir().expect("the test should create a cache directory");
    let cache = open_persistent_prompt_cache_disk_store(&cache_directory, LARGE_CACHE_LIMIT_BYTES)
        .expect("the cache should open");
    let block_key = persistent_prompt_cache_block_key_for_seed(0);
    let conflicting_final_directory = cache_directory
        .path()
        .join("blocks")
        .join(hex::encode(block_key.block_hash()));
    fs::create_dir(&conflicting_final_directory)
        .expect("the conflicting final directory should be created");
    fs::write(
        conflicting_final_directory.join("foreign-marker"),
        b"foreign",
    )
    .expect("the conflicting directory marker should write");

    let publication_error = cache
        .publish_block(
            &runtime,
            &block_key,
            None,
            &synthetic_kv_block_tensors(&runtime),
            &synthetic_recurrent_snapshot_tensors(&runtime),
        )
        .expect_err("the conflicting final directory must prevent commit");

    assert!(matches!(
        publication_error,
        PersistentPromptCacheDiskStoreError::ExistingBlockTopologyMismatch { .. }
    ));
    assert!(conflicting_final_directory.join("foreign-marker").is_file());
    assert_no_staging_directories(cache_directory.path());
    assert_eq!(cache.sequence_state_block_count(), 0);
}

#[tokio::test]
async fn should_keep_state_file_metadata_bounded_to_the_storage_contract() {
    // Model identity is already represented by the opaque storage fingerprint.
    // Repeating names/revisions in every large state file leaks detail and grows headers.
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let cache_directory = tempfile::tempdir().expect("the test should create a cache directory");
    let cache = open_persistent_prompt_cache_disk_store(&cache_directory, LARGE_CACHE_LIMIT_BYTES)
        .expect("the cache should open");
    let block_key = persistent_prompt_cache_block_key_for_seed(0);
    cache
        .publish_block(
            &runtime,
            &block_key,
            None,
            &synthetic_kv_block_tensors(&runtime),
            &synthetic_recurrent_snapshot_tensors(&runtime),
        )
        .expect("the block should publish");

    let sequence_state_path = cache_directory
        .path()
        .join("blocks")
        .join(hex::encode(block_key.block_hash()))
        .join("sequence.safetensors");
    let sequence_state_bytes = fs::read(sequence_state_path)
        .expect("the published sequence-state file should be readable");
    let header_byte_count = u64::from_le_bytes(
        sequence_state_bytes[..8]
            .try_into()
            .expect("the sequence-state header length should be present"),
    ) as usize;
    let header: Value = serde_json::from_slice(&sequence_state_bytes[8..8 + header_byte_count])
        .expect("the sequence-state header should contain JSON");
    let metadata = header["__metadata__"]
        .as_object()
        .expect("the sequence-state header should contain metadata");
    assert_eq!(metadata.len(), 3);
    assert!(metadata.contains_key("format_version"));
    assert!(metadata.contains_key("block_token_count"));
    assert!(metadata.contains_key("storage_contract_fingerprint"));
    assert!(!metadata.contains_key("model_id"));
    assert!(!metadata.contains_key("model_revision"));
}

fn replace_one_tensor_with_invalid_shape(
    runtime: &astronomical_runtime_integration::MlxRuntime,
    tensors: &mut HashMap<String, MlxArray>,
) {
    let tensor_name = tensors
        .keys()
        .next()
        .cloned()
        .expect("the contract-derived tensor set should not be empty");
    tensors.insert(
        tensor_name,
        runtime
            .zeros(&[1], MlxDtype::BFloat16)
            .expect("the invalid-shape tensor should allocate"),
    );
}

fn assert_no_visible_or_staged_blocks(cache_directory: &Path) {
    let blocks_directory = cache_directory.join("blocks");
    assert_eq!(
        fs::read_dir(blocks_directory)
            .expect("the blocks directory should remain readable")
            .count(),
        0
    );
}

fn assert_no_staging_directories(cache_directory: &Path) {
    let blocks_directory = cache_directory.join("blocks");
    assert!(
        fs::read_dir(blocks_directory)
            .expect("the blocks directory should remain readable")
            .all(|directory_entry| {
                !directory_entry
                    .expect("the block entry should be readable")
                    .file_name()
                    .to_string_lossy()
                    .contains(".staging-")
            })
    );
}
