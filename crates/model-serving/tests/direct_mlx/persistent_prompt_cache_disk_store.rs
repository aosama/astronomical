use std::fs;
use std::sync::Arc;

use astronomical_model_serving::{
    ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID, ORNITH_1_0_35B_OPTIQ_4BIT_REVISION, PerformanceAttribution,
    PersistentPromptCacheDiskStoreError, PersistentPromptCacheWriteQueue,
    PersistentPromptCacheWriteQueueOutcome, PersistentVisualEmbeddingKey,
};
use astronomical_runtime_integration::MlxDtype;

use super::persistent_prompt_cache_disk_store_support::*;
use crate::common::qwen3_5_moe::persistent_visual_embedding_model_contract;

const LARGE_CACHE_LIMIT_BYTES: u64 = 10 * 1024 * 1024 * 1024;

#[tokio::test]
async fn should_publish_a_serialized_block_through_the_bounded_writer_queue() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let persistent_prompt_cache = Arc::new(
        open_persistent_prompt_cache_disk_store(
            &persistent_prompt_cache_directory,
            LARGE_CACHE_LIMIT_BYTES,
        )
        .expect("the persistent prompt cache should open an empty directory"),
    );
    let persistent_prompt_cache_block_key = persistent_prompt_cache_block_key_for_seed(0);
    let kv_block_tensors = synthetic_kv_block_tensors(&runtime);
    let recurrent_snapshot_tensors = synthetic_recurrent_snapshot_tensors(&runtime);
    let write_queue =
        PersistentPromptCacheWriteQueue::new(Arc::clone(&persistent_prompt_cache), Some(10))
            .expect("the test should start the bounded writer queue");
    let mut performance_attribution = PerformanceAttribution::disabled();

    let write_queue_outcome = write_queue
        .serialize_and_enqueue(
            &runtime,
            &persistent_prompt_cache_block_key,
            None,
            &kv_block_tensors,
            &recurrent_snapshot_tensors,
            &mut performance_attribution,
        )
        .expect("the test should serialize and enqueue the prompt-cache block");
    assert_eq!(
        write_queue_outcome,
        PersistentPromptCacheWriteQueueOutcome::Queued
    );
    let pending_serialized_bytes_after_first_enqueue = write_queue.pending_serialized_bytes();
    let duplicate_write_queue_outcome = write_queue
        .serialize_and_enqueue(
            &runtime,
            &persistent_prompt_cache_block_key,
            None,
            &kv_block_tensors,
            &recurrent_snapshot_tensors,
            &mut performance_attribution,
        )
        .expect("the test should recognize the already queued prompt-cache block");
    assert_eq!(
        duplicate_write_queue_outcome,
        PersistentPromptCacheWriteQueueOutcome::AlreadyQueued
    );
    assert_eq!(
        write_queue.pending_serialized_bytes(),
        pending_serialized_bytes_after_first_enqueue,
        "deduplication must not retain a second serialized payload"
    );
    drop(write_queue);

    assert_eq!(persistent_prompt_cache.sequence_state_block_count(), 1);
    assert_eq!(persistent_prompt_cache.boundary_state_snapshot_count(), 1);
}

#[tokio::test]
async fn should_save_and_load_kv_block_and_recurrent_snapshot_as_separate_files() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the persistent prompt cache should open an empty directory");
    let persistent_prompt_cache_block_key = persistent_prompt_cache_block_key_for_seed(0);
    let kv_block_tensors = synthetic_kv_block_tensors(&runtime);
    let recurrent_snapshot_tensors = synthetic_recurrent_snapshot_tensors(&runtime);

    persistent_prompt_cache
        .save_kv_block_and_recurrent_snapshot(
            &runtime,
            &persistent_prompt_cache_block_key,
            None,
            &kv_block_tensors,
            &recurrent_snapshot_tensors,
        )
        .expect("the persistent prompt cache should save split files");

    assert_eq!(persistent_prompt_cache.sequence_state_block_count(), 1);
    assert_eq!(persistent_prompt_cache.boundary_state_snapshot_count(), 1);
    assert!(
        persistent_prompt_cache_directory
            .path()
            .join("kv_blocks")
            .join(format!(
                "{}.safetensors",
                hex::encode(persistent_prompt_cache_block_key.block_hash())
            ))
            .is_file()
    );
    assert!(
        persistent_prompt_cache_directory
            .path()
            .join("recurrent_snapshots")
            .join(format!(
                "{}.safetensors",
                hex::encode(persistent_prompt_cache_block_key.block_hash())
            ))
            .is_file()
    );

    let loaded_kv_block_tensors = persistent_prompt_cache
        .load_kv_block(&runtime, &persistent_prompt_cache_block_key)
        .expect("the persistent prompt cache should load the saved KV block")
        .expect("the saved KV block should be present");
    let loaded_recurrent_snapshot_tensors = persistent_prompt_cache
        .load_recurrent_snapshot(&runtime, &persistent_prompt_cache_block_key)
        .expect("the persistent prompt cache should load the saved recurrent snapshot")
        .expect("the saved recurrent snapshot should be present");

    assert_eq!(loaded_kv_block_tensors.len(), 20);
    assert_eq!(loaded_recurrent_snapshot_tensors.len(), 60);
    assert_split_tensor_shapes_match(&loaded_kv_block_tensors, &kv_block_tensors);
    assert_split_tensor_shapes_match(
        &loaded_recurrent_snapshot_tensors,
        &recurrent_snapshot_tensors,
    );
}

#[tokio::test]
async fn should_report_zero_files_for_an_empty_prompt_cache_directory() {
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the persistent prompt cache should open an empty directory");

    assert_eq!(persistent_prompt_cache.sequence_state_block_count(), 0);
    assert_eq!(persistent_prompt_cache.boundary_state_snapshot_count(), 0);
    assert_eq!(persistent_prompt_cache.total_size_bytes(), 0);
}

#[tokio::test]
async fn should_save_and_load_one_visual_embedding_file() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the persistent prompt cache should open an empty directory");
    let visual_embedding_key = PersistentVisualEmbeddingKey::for_image(
        [7_u8; 32],
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    );
    let visual_embeddings = runtime
        .zeros(&[2, 2_048], MlxDtype::BFloat16)
        .expect("the test should create visual embeddings");

    persistent_prompt_cache
        .save_visual_embedding(&runtime, &visual_embedding_key, &visual_embeddings)
        .expect("the persistent prompt cache should save visual embeddings");

    assert_eq!(persistent_prompt_cache.visual_embedding_count(), 1);
    assert!(persistent_prompt_cache.visual_embedding_total_size_bytes() > 0);
    assert!(
        persistent_prompt_cache.has_visual_embedding(&visual_embedding_key.visual_embedding_hash())
    );
    assert!(persistent_prompt_cache.total_size_bytes() > 0);
    assert_eq!(
        persistent_prompt_cache.visual_embedding_total_size_bytes(),
        persistent_prompt_cache.total_size_bytes()
    );
    let loaded_visual_embeddings = persistent_prompt_cache
        .load_visual_embedding(
            &runtime,
            &visual_embedding_key,
            &persistent_visual_embedding_model_contract(),
        )
        .expect("the persistent prompt cache should load visual embeddings")
        .expect("the saved visual embedding should be present");
    assert_eq!(loaded_visual_embeddings.shape(), vec![2, 2_048]);
    assert_eq!(loaded_visual_embeddings.dtype(), MlxDtype::BFloat16);
}

#[tokio::test]
async fn should_retain_root_recurrent_snapshot_after_child_snapshot_is_saved() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the persistent prompt cache should open an empty directory");
    let parent_persistent_prompt_cache_block_key = persistent_prompt_cache_block_key_for_seed(0);
    let child_persistent_prompt_cache_block_key = parent_persistent_prompt_cache_block_key
        .for_child_block(&block_tokens_for_seed(10_000))
        .expect("the test should hash the child block");
    let kv_block_tensors = synthetic_kv_block_tensors(&runtime);
    let recurrent_snapshot_tensors = synthetic_recurrent_snapshot_tensors(&runtime);

    persistent_prompt_cache
        .save_kv_block_and_recurrent_snapshot(
            &runtime,
            &parent_persistent_prompt_cache_block_key,
            None,
            &kv_block_tensors,
            &recurrent_snapshot_tensors,
        )
        .expect("the parent split files should save");
    persistent_prompt_cache
        .save_kv_block_and_recurrent_snapshot(
            &runtime,
            &child_persistent_prompt_cache_block_key,
            Some(&parent_persistent_prompt_cache_block_key),
            &kv_block_tensors,
            &recurrent_snapshot_tensors,
        )
        .expect("the child split files should save");

    assert_eq!(persistent_prompt_cache.sequence_state_block_count(), 2);
    assert_eq!(persistent_prompt_cache.boundary_state_snapshot_count(), 2);
    assert!(
        persistent_prompt_cache
            .has_recurrent_snapshot(&parent_persistent_prompt_cache_block_key.block_hash())
    );
    assert!(
        persistent_prompt_cache
            .has_recurrent_snapshot(&child_persistent_prompt_cache_block_key.block_hash())
    );
}

#[tokio::test]
async fn should_delete_invalid_current_format_file_during_scan() {
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let kv_blocks_directory = persistent_prompt_cache_directory.path().join("kv_blocks");
    fs::create_dir_all(&kv_blocks_directory)
        .expect("the test should create the KV blocks directory");
    let invalid_kv_block_file_path =
        kv_blocks_directory.join(format!("{}.safetensors", "0".repeat(64)));
    fs::write(&invalid_kv_block_file_path, b"not a safetensors file")
        .expect("the test should write an invalid current-format file");

    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the persistent prompt cache should delete invalid current files");

    assert_eq!(persistent_prompt_cache.sequence_state_block_count(), 0);
    assert!(
        !invalid_kv_block_file_path.exists(),
        "invalid cache-owned .safetensors files must be deleted during scan"
    );
}

#[tokio::test]
async fn should_delete_incompatible_format_file_during_scan() {
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let kv_blocks_directory = persistent_prompt_cache_directory.path().join("kv_blocks");
    fs::create_dir_all(&kv_blocks_directory)
        .expect("the test should create the KV blocks directory");
    let format_four_kv_block_file_path =
        kv_blocks_directory.join(format!("{}.safetensors", "1".repeat(64)));
    write_format_four_cache_file(&format_four_kv_block_file_path);

    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the persistent prompt cache should remove incompatible format files");

    assert_eq!(persistent_prompt_cache.sequence_state_block_count(), 0);
    assert!(
        !format_four_kv_block_file_path.exists(),
        "cache-owned state from an incompatible format must not consume disk capacity"
    );
}

#[tokio::test]
async fn should_reject_split_save_when_the_combined_files_exceed_the_size_bound() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let persistent_prompt_cache =
        open_persistent_prompt_cache_disk_store(&persistent_prompt_cache_directory, 1)
            .expect("the persistent prompt cache should open with a one-byte bound");

    let save_result = persistent_prompt_cache.save_kv_block_and_recurrent_snapshot(
        &runtime,
        &persistent_prompt_cache_block_key_for_seed(0),
        None,
        &synthetic_kv_block_tensors(&runtime),
        &synthetic_recurrent_snapshot_tensors(&runtime),
    );

    assert!(matches!(
        save_result,
        Err(PersistentPromptCacheDiskStoreError::SizeBoundExceeded { .. })
    ));
}

#[tokio::test]
async fn should_evict_the_oldest_visual_embedding_under_shared_quota_pressure() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");

    let first_visual_embedding_key = PersistentVisualEmbeddingKey::for_image(
        [7_u8; 32],
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    );
    let second_visual_embedding_key = PersistentVisualEmbeddingKey::for_image(
        [8_u8; 32],
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    );
    let first_visual_embeddings = runtime
        .zeros(&[8, 2_048], MlxDtype::BFloat16)
        .expect("the test should create first visual embeddings");
    let second_visual_embeddings = runtime
        .zeros(&[8, 2_048], MlxDtype::BFloat16)
        .expect("the test should create second visual embeddings");

    // One projected image fits the estimate check and actual quota, but two
    // projected images do not. This proves visual files participate in the
    // same byte accounting and oldest-file eviction loop as prompt-state files.
    let one_visual_embedding_quota_bytes = u64::try_from(first_visual_embeddings.byte_count())
        .unwrap_or(0)
        .saturating_add(17 * 1024);
    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        one_visual_embedding_quota_bytes,
    )
    .expect("the persistent prompt cache should open an empty directory");

    persistent_prompt_cache
        .save_visual_embedding(
            &runtime,
            &first_visual_embedding_key,
            &first_visual_embeddings,
        )
        .expect("the persistent prompt cache should save the first visual embedding");
    assert_eq!(persistent_prompt_cache.visual_embedding_count(), 1);
    assert!(
        persistent_prompt_cache
            .has_visual_embedding(&first_visual_embedding_key.visual_embedding_hash())
    );
    assert!(persistent_prompt_cache.total_size_bytes() <= one_visual_embedding_quota_bytes);

    tokio::time::sleep(std::time::Duration::from_millis(5)).await;

    persistent_prompt_cache
        .save_visual_embedding(
            &runtime,
            &second_visual_embedding_key,
            &second_visual_embeddings,
        )
        .expect("the persistent prompt cache should save the second visual embedding");

    assert_eq!(persistent_prompt_cache.visual_embedding_count(), 1);
    assert!(
        !persistent_prompt_cache
            .has_visual_embedding(&first_visual_embedding_key.visual_embedding_hash()),
        "the oldest visual embedding should be evicted first"
    );
    assert!(
        persistent_prompt_cache
            .has_visual_embedding(&second_visual_embedding_key.visual_embedding_hash())
    );
    assert!(
        persistent_prompt_cache.total_size_bytes() <= one_visual_embedding_quota_bytes,
        "visual embedding eviction must keep the shared quota satisfied"
    );
}

#[tokio::test]
async fn should_untrack_invalid_visual_embedding_load_and_accept_replacement() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the persistent prompt cache should open an empty directory");
    let visual_embedding_key = PersistentVisualEmbeddingKey::for_image(
        [9_u8; 32],
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    );
    let visual_embeddings = runtime
        .zeros(&[2, 2_048], MlxDtype::BFloat16)
        .expect("the test should create visual embeddings");

    persistent_prompt_cache
        .save_visual_embedding(&runtime, &visual_embedding_key, &visual_embeddings)
        .expect("the persistent prompt cache should save visual embeddings");
    assert_eq!(persistent_prompt_cache.visual_embedding_count(), 1);

    let visual_embedding_file_path = persistent_prompt_cache_directory
        .path()
        .join("visual_embeddings")
        .join(format!(
            "{}.safetensors",
            hex::encode(visual_embedding_key.visual_embedding_hash())
        ));
    fs::write(&visual_embedding_file_path, b"not a safetensors file")
        .expect("the test should corrupt the visual embedding file");

    let load_result = persistent_prompt_cache.load_visual_embedding(
        &runtime,
        &visual_embedding_key,
        &persistent_visual_embedding_model_contract(),
    );

    assert!(matches!(
        load_result,
        Err(PersistentPromptCacheDiskStoreError::ValidateModelSpecificArtifact { .. })
    ));
    assert_eq!(persistent_prompt_cache.visual_embedding_count(), 0);
    assert!(
        !persistent_prompt_cache
            .has_visual_embedding(&visual_embedding_key.visual_embedding_hash())
    );

    persistent_prompt_cache
        .save_visual_embedding(&runtime, &visual_embedding_key, &visual_embeddings)
        .expect("the persistent prompt cache should replace the invalid visual embedding file");
    let loaded_visual_embeddings = persistent_prompt_cache
        .load_visual_embedding(
            &runtime,
            &visual_embedding_key,
            &persistent_visual_embedding_model_contract(),
        )
        .expect("the replacement visual embedding should load")
        .expect("the replacement visual embedding should be tracked");
    assert_eq!(loaded_visual_embeddings.shape(), vec![2, 2_048]);
    assert_eq!(loaded_visual_embeddings.dtype(), MlxDtype::BFloat16);
}
