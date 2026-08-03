use std::collections::HashMap;
use std::fs;
use std::fs::FileTimes;
use std::os::unix::fs::PermissionsExt;
use std::time::{Duration, UNIX_EPOCH};

use astronomical_model_serving::{
    ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID, ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    PersistentPromptCacheDiskStoreError, PersistentVisualEmbeddingKey,
};
use astronomical_runtime_integration::MlxDtype;

use super::persistent_prompt_cache_disk_store_support::*;

const ONE_MEBIBYTE_QUOTA_BYTES: u64 = 1024 * 1024;
const LARGE_CACHE_LIMIT_BYTES: u64 = 10 * 1024 * 1024 * 1024;

#[tokio::test]
async fn should_rollback_a_new_visual_embedding_when_global_eviction_fails() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let global_prompt_cache_root_directory =
        tempfile::tempdir().expect("the test should create a global prompt-cache root");
    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &global_prompt_cache_root_directory,
        ONE_MEBIBYTE_QUOTA_BYTES,
    )
    .expect("the persistent prompt cache should open an empty directory");
    let cross_model_directory = global_prompt_cache_root_directory
        .path()
        .join("cross-model")
        .join("revision")
        .join("kv_blocks");
    fs::create_dir_all(&cross_model_directory)
        .expect("the test should create a cross-model cache directory");
    let undeletable_cross_model_file_path =
        cross_model_directory.join(format!("{}.safetensors", "a".repeat(64)));
    fs::write(
        &undeletable_cross_model_file_path,
        vec![0_u8; ONE_MEBIBYTE_QUOTA_BYTES as usize],
    )
    .expect("the test should fill the global quota with a cross-model file");
    fs::File::open(&undeletable_cross_model_file_path)
        .expect("the test should open the cross-model file")
        .set_times(FileTimes::new().set_modified(UNIX_EPOCH + Duration::from_secs(10)))
        .expect("the test should make the cross-model file oldest");
    let original_cross_model_directory_mode = fs::metadata(&cross_model_directory)
        .expect("the test should read cross-model directory metadata")
        .permissions()
        .mode();
    fs::set_permissions(&cross_model_directory, fs::Permissions::from_mode(0o555))
        .expect("the test should prevent cross-model eviction");
    let visual_embedding_key = PersistentVisualEmbeddingKey::for_image(
        [17_u8; 32],
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    );
    let visual_embeddings = runtime
        .zeros(&[2, 2_048], MlxDtype::BFloat16)
        .expect("the test should create visual embeddings");

    let save_result = persistent_prompt_cache.save_visual_embedding(
        &runtime,
        &visual_embedding_key,
        &visual_embeddings,
    );

    fs::set_permissions(
        &cross_model_directory,
        fs::Permissions::from_mode(original_cross_model_directory_mode),
    )
    .expect("the test should restore cross-model directory permissions");
    assert!(matches!(
        save_result,
        Err(PersistentPromptCacheDiskStoreError::RemovePromptCacheFile {
            persistent_prompt_cache_file_path,
            ..
        }) if persistent_prompt_cache_file_path == undeletable_cross_model_file_path
    ));
    assert!(
        !persistent_prompt_cache
            .has_visual_embedding(&visual_embedding_key.visual_embedding_hash())
    );
    assert_eq!(
        persistent_prompt_cache.total_size_bytes(),
        ONE_MEBIBYTE_QUOTA_BYTES
    );
}

#[tokio::test]
async fn should_rollback_both_new_split_files_when_parent_snapshot_deletion_fails() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the persistent prompt cache should open an empty directory");
    let root_persistent_prompt_cache_block_key = persistent_prompt_cache_block_key_for_seed(0);
    let child_persistent_prompt_cache_block_key = root_persistent_prompt_cache_block_key
        .for_child_block(&block_tokens_for_seed(10_000))
        .expect("the test should hash the child block");
    let grandchild_persistent_prompt_cache_block_key = child_persistent_prompt_cache_block_key
        .for_child_block(&block_tokens_for_seed(20_000))
        .expect("the test should hash the grandchild block");
    let kv_block_tensors = HashMap::from([(
        "filesystem_rollback_kv".to_owned(),
        runtime
            .zeros(&[1], MlxDtype::BFloat16)
            .expect("the test should create one tiny KV tensor"),
    )]);
    let recurrent_snapshot_tensors = HashMap::from([(
        "filesystem_rollback_recurrent".to_owned(),
        runtime
            .zeros(&[1], MlxDtype::Float32)
            .expect("the test should create one tiny recurrent tensor"),
    )]);
    persistent_prompt_cache
        .save_kv_block_and_recurrent_snapshot(
            &runtime,
            &root_persistent_prompt_cache_block_key,
            None,
            &kv_block_tensors,
            &recurrent_snapshot_tensors,
        )
        .expect("the root split files should save");
    persistent_prompt_cache
        .save_kv_block_and_recurrent_snapshot(
            &runtime,
            &child_persistent_prompt_cache_block_key,
            Some(&root_persistent_prompt_cache_block_key),
            &kv_block_tensors,
            &recurrent_snapshot_tensors,
        )
        .expect("the child split files should save");
    let child_recurrent_snapshot_file_path = persistent_prompt_cache_directory
        .path()
        .join("recurrent_snapshots")
        .join(format!(
            "{}.safetensors",
            hex::encode(child_persistent_prompt_cache_block_key.block_hash())
        ));
    fs::remove_file(&child_recurrent_snapshot_file_path)
        .expect("the test should remove the child snapshot file");
    fs::create_dir(&child_recurrent_snapshot_file_path)
        .expect("the test should replace the child snapshot with an undeletable directory");

    let save_result = persistent_prompt_cache.save_kv_block_and_recurrent_snapshot(
        &runtime,
        &grandchild_persistent_prompt_cache_block_key,
        Some(&child_persistent_prompt_cache_block_key),
        &kv_block_tensors,
        &recurrent_snapshot_tensors,
    );

    assert!(matches!(
        save_result,
        Err(PersistentPromptCacheDiskStoreError::RemovePromptCacheFile {
            persistent_prompt_cache_file_path,
            ..
        }) if persistent_prompt_cache_file_path == child_recurrent_snapshot_file_path
    ));
    assert!(
        !persistent_prompt_cache
            .has_kv_block(&grandchild_persistent_prompt_cache_block_key.block_hash())
    );
    assert!(
        !persistent_prompt_cache
            .has_recurrent_snapshot(&grandchild_persistent_prompt_cache_block_key.block_hash())
    );
    assert!(
        !persistent_prompt_cache_directory
            .path()
            .join("kv_blocks")
            .join(format!(
                "{}.safetensors",
                hex::encode(grandchild_persistent_prompt_cache_block_key.block_hash())
            ))
            .exists()
    );
    assert!(
        !persistent_prompt_cache_directory
            .path()
            .join("recurrent_snapshots")
            .join(format!(
                "{}.safetensors",
                hex::encode(grandchild_persistent_prompt_cache_block_key.block_hash())
            ))
            .exists()
    );
}
