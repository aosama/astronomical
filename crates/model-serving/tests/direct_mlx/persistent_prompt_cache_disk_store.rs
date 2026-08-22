use std::fs;
use std::sync::Arc;
use std::time::Instant;

use astronomical_model_serving::{
    PersistentPromptCacheDiskStoreError, PersistentPromptCachePublicationOutcome,
};
use astronomical_runtime_integration::PositionalFileReadMetrics;

use super::persistent_prompt_cache_disk_store_support::*;
use crate::common::qwen3_5_moe::persistent_prompt_cache_model_contract;

const LARGE_CACHE_LIMIT_BYTES: u64 = 10 * 1024 * 1024 * 1024;

#[tokio::test]
async fn should_publish_a_block_directly_and_report_idempotent_republication() {
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
    let publication_outcome = persistent_prompt_cache
        .publish_block(
            &runtime,
            &persistent_prompt_cache_block_key,
            None,
            &kv_block_tensors,
            &recurrent_snapshot_tensors,
        )
        .expect("the test should durably publish the prompt-cache block");
    assert_eq!(
        publication_outcome,
        PersistentPromptCachePublicationOutcome::Published
    );
    let duplicate_publication_outcome = persistent_prompt_cache
        .publish_block(
            &runtime,
            &persistent_prompt_cache_block_key,
            None,
            &kv_block_tensors,
            &recurrent_snapshot_tensors,
        )
        .expect("the test should validate the already published prompt-cache block");
    assert_eq!(
        duplicate_publication_outcome,
        PersistentPromptCachePublicationOutcome::AlreadyPublished
    );

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
        .publish_block(
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
            .join("blocks")
            .join(hex::encode(persistent_prompt_cache_block_key.block_hash()))
            .join("sequence.safetensors")
            .is_file()
    );
    assert!(
        persistent_prompt_cache_directory
            .path()
            .join("blocks")
            .join(hex::encode(persistent_prompt_cache_block_key.block_hash()))
            .join("boundary.safetensors")
            .is_file()
    );
    assert!(
        persistent_prompt_cache_directory
            .path()
            .join("blocks")
            .join(hex::encode(persistent_prompt_cache_block_key.block_hash()))
            .join("manifest.json")
            .is_file()
    );
    let published_block_directory = persistent_prompt_cache_directory
        .path()
        .join("blocks")
        .join(hex::encode(persistent_prompt_cache_block_key.block_hash()));
    let model_contract = persistent_prompt_cache_model_contract();
    let actual_sequence_state_file_bytes =
        fs::metadata(published_block_directory.join("sequence.safetensors"))
            .expect("the test should read sequence-state metadata")
            .len();
    let actual_boundary_state_file_bytes =
        fs::metadata(published_block_directory.join("boundary.safetensors"))
            .expect("the test should read boundary-state metadata")
            .len();
    let actual_manifest_file_bytes = fs::metadata(published_block_directory.join("manifest.json"))
        .expect("the test should read manifest metadata")
        .len();
    assert_eq!(
        actual_sequence_state_file_bytes,
        model_contract.sequence_state_file_bytes()
    );
    assert_eq!(
        actual_boundary_state_file_bytes,
        model_contract.boundary_state_file_bytes()
    );
    assert!(actual_manifest_file_bytes <= model_contract.maximum_block_manifest_file_bytes());
    assert!(
        actual_sequence_state_file_bytes
            .saturating_add(actual_boundary_state_file_bytes)
            .saturating_add(actual_manifest_file_bytes)
            <= model_contract.maximum_committed_block_bytes()
    );

    let positional_file_read_metrics = Arc::new(PositionalFileReadMetrics::default());
    let loaded_kv_block_tensors = persistent_prompt_cache
        .load_kv_block(
            &runtime,
            &persistent_prompt_cache_block_key,
            Some(Arc::clone(&positional_file_read_metrics)),
        )
        .expect("the persistent prompt cache should load the saved KV block")
        .expect("the saved KV block should be present");
    let loaded_recurrent_snapshot_tensors = persistent_prompt_cache
        .load_recurrent_snapshot(
            &runtime,
            &persistent_prompt_cache_block_key,
            Some(Arc::clone(&positional_file_read_metrics)),
        )
        .expect("the persistent prompt cache should load the saved recurrent snapshot")
        .expect("the saved recurrent snapshot should be present");

    assert_eq!(loaded_kv_block_tensors.len(), 20);
    assert_eq!(loaded_recurrent_snapshot_tensors.len(), 60);
    assert_split_tensor_shapes_match(&loaded_kv_block_tensors, &kv_block_tensors);
    assert_split_tensor_shapes_match(
        &loaded_recurrent_snapshot_tensors,
        &recurrent_snapshot_tensors,
    );
    drop(kv_block_tensors);
    drop(recurrent_snapshot_tensors);
    runtime
        .synchronize_gpu_stream_and_clear_allocator_cache()
        .expect("the test should release serialized source tensors before restore evaluation");
    let loaded_state_arrays = loaded_kv_block_tensors
        .values()
        .chain(loaded_recurrent_snapshot_tensors.values())
        .collect::<Vec<_>>();
    let restore_evaluation_started_at = Instant::now();
    runtime
        .evaluate_arrays(&loaded_state_arrays)
        .expect("the restored prompt-cache tensors should evaluate");
    let restore_evaluation_elapsed = restore_evaluation_started_at.elapsed();
    let positional_file_read_snapshot = positional_file_read_metrics.snapshot();
    assert_eq!(positional_file_read_snapshot.read_call_count, 80);
    assert!(positional_file_read_snapshot.read_byte_count > 0);
    assert_eq!(positional_file_read_snapshot.read_failure_count, 0);
    eprintln!(
        "[prompt-cache-positional-read] status=success tensors=80 read_calls={} read_bytes={} maximum_concurrent_reads={} summed_read_milliseconds={:.3} evaluation_milliseconds={:.3}",
        positional_file_read_snapshot.read_call_count,
        positional_file_read_snapshot.read_byte_count,
        positional_file_read_snapshot.maximum_concurrent_read_count,
        positional_file_read_snapshot.total_read_elapsed_nanoseconds as f64 / 1_000_000.0,
        restore_evaluation_elapsed.as_secs_f64() * 1_000.0,
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
        .publish_block(
            &runtime,
            &parent_persistent_prompt_cache_block_key,
            None,
            &kv_block_tensors,
            &recurrent_snapshot_tensors,
        )
        .expect("the parent split files should save");
    persistent_prompt_cache
        .publish_block(
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
    let block_hash_hex = "0".repeat(64);
    let block_directory = persistent_prompt_cache_directory
        .path()
        .join("blocks")
        .join(&block_hash_hex);
    fs::create_dir_all(&block_directory).expect("the test should create the block directory");
    write_block_manifest_for_hash(&block_directory, &block_hash_hex, 0);
    let invalid_sequence_state_file_path = block_directory.join("sequence.safetensors");
    fs::write(&invalid_sequence_state_file_path, b"not a safetensors file")
        .expect("the test should write an invalid current-format file");

    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the persistent prompt cache should delete invalid current files");

    assert_eq!(persistent_prompt_cache.sequence_state_block_count(), 0);
    assert!(
        !block_directory.exists(),
        "invalid cache-owned block directories must be deleted during scan"
    );
}

#[tokio::test]
async fn should_delete_incompatible_format_file_during_scan() {
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let block_hash_hex = "1".repeat(64);
    let block_directory = persistent_prompt_cache_directory
        .path()
        .join("blocks")
        .join(&block_hash_hex);
    fs::create_dir_all(&block_directory).expect("the test should create the block directory");
    write_block_manifest_for_hash_with_format(&block_directory, &block_hash_hex, 1, "11");

    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the persistent prompt cache should remove incompatible format files");

    assert_eq!(persistent_prompt_cache.sequence_state_block_count(), 0);
    assert!(
        !block_directory.exists(),
        "cache-owned state from an incompatible format must not consume disk capacity"
    );
}

fn write_block_manifest_for_hash(
    block_directory: &std::path::Path,
    block_hash_hex: &str,
    block_index: u32,
) {
    write_block_manifest_for_hash_with_format(block_directory, block_hash_hex, block_index, "12");
}

fn write_block_manifest_for_hash_with_format(
    block_directory: &std::path::Path,
    block_hash_hex: &str,
    block_index: u32,
    format_version: &str,
) {
    let persistent_prompt_cache_model_contract = persistent_prompt_cache_model_contract();
    let manifest_json = serde_json::json!({
        "format_version": format_version,
        "block_hash": block_hash_hex,
        "block_index": block_index,
        "parent_block_hash": null,
        "storage_contract_fingerprint": persistent_prompt_cache_model_contract.storage_contract_fingerprint_hex(),
        "has_sequence_state": true,
        "has_boundary_state": true,
    });
    fs::write(
        block_directory.join("manifest.json"),
        manifest_json.to_string(),
    )
    .expect("the test should write the block manifest");
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

    let save_result = persistent_prompt_cache.publish_block(
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
