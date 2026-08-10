use std::fs;
use std::os::unix::fs::symlink;

use astronomical_model_serving::{
    PersistentPromptCacheDiskStore, PersistentPromptCacheDiskStoreConfig,
    PersistentPromptCacheDiskStoreError,
};

use crate::common::qwen3_5_moe::persistent_prompt_cache_model_contract;
use crate::direct_mlx::persistent_prompt_cache_disk_store_support::{
    persistent_prompt_cache_block_key_for_seed, runtime_with_shared_limits,
    synthetic_kv_block_tensors, synthetic_recurrent_snapshot_tensors,
};

const LARGE_CACHE_LIMIT_BYTES: u64 = 10 * 1024 * 1024 * 1024;

#[tokio::test]
async fn should_recreate_deleted_active_model_directories_before_replacement_write() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let global_prompt_cache_root_directory =
        tempfile::tempdir().expect("the test should create a global prompt-cache root");
    let active_model_prompt_cache_directory = global_prompt_cache_root_directory
        .path()
        .join("active-model")
        .join("revision");
    let persistent_prompt_cache = PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            active_model_prompt_cache_directory.clone(),
            global_prompt_cache_root_directory.path().to_path_buf(),
            LARGE_CACHE_LIMIT_BYTES,
        ),
        persistent_prompt_cache_model_contract(),
    )
    .expect("the persistent prompt cache should open");
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
        .expect("the initial split files should save");

    fs::remove_dir_all(&active_model_prompt_cache_directory)
        .expect("the test should remove the active model cache while it remains open");
    let stale_lookup =
        persistent_prompt_cache.load_kv_block(&runtime, &persistent_prompt_cache_block_key, None);
    assert!(matches!(
        stale_lookup,
        Err(PersistentPromptCacheDiskStoreError::OpenBlockFile { .. })
    ));
    assert_eq!(persistent_prompt_cache.sequence_state_block_count(), 0);
    assert!(
        !persistent_prompt_cache
            .has_recurrent_snapshot(&persistent_prompt_cache_block_key.block_hash()),
        "missing recurrent snapshot files must not remain present in the live index"
    );
    assert_eq!(persistent_prompt_cache.boundary_state_snapshot_count(), 0);

    persistent_prompt_cache
        .publish_block(
            &runtime,
            &persistent_prompt_cache_block_key,
            None,
            &kv_block_tensors,
            &recurrent_snapshot_tensors,
        )
        .expect("the replacement write should recreate deleted cache directories");

    assert!(active_model_prompt_cache_directory.join("blocks").is_dir());
    assert!(
        persistent_prompt_cache
            .load_kv_block(&runtime, &persistent_prompt_cache_block_key, None)
            .expect("the replacement KV block should load")
            .is_some()
    );
}

#[test]
fn should_never_follow_a_global_prompt_cache_symlink_outside_the_root() {
    let global_prompt_cache_root_directory =
        tempfile::tempdir().expect("the test should create a global prompt-cache root");
    let external_directory =
        tempfile::tempdir().expect("the test should create an external directory");
    let external_file_path = external_directory.path().join("must-remain.bin");
    fs::write(&external_file_path, vec![7_u8; 1_024])
        .expect("the test should write an external file");
    let root_owned_symlink_path = global_prompt_cache_root_directory
        .path()
        .join("outside-link");
    symlink(external_directory.path(), &root_owned_symlink_path)
        .expect("the test should create a root-owned symlink");

    PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            global_prompt_cache_root_directory
                .path()
                .join("active-model")
                .join("revision"),
            global_prompt_cache_root_directory.path().to_path_buf(),
            0,
        ),
        persistent_prompt_cache_model_contract(),
    )
    .expect("global quota enforcement should remove only the symlink itself");

    assert!(external_file_path.exists());
    assert!(fs::symlink_metadata(&root_owned_symlink_path).is_err());
}

#[test]
fn should_never_follow_an_active_model_safetensors_symlink() {
    let global_prompt_cache_root_directory =
        tempfile::tempdir().expect("the test should create a global prompt-cache root");
    let active_model_kv_blocks_directory = global_prompt_cache_root_directory
        .path()
        .join("active-model")
        .join("revision")
        .join("kv_blocks");
    fs::create_dir_all(&active_model_kv_blocks_directory)
        .expect("the test should create the active-model KV blocks directory");
    let external_directory =
        tempfile::tempdir().expect("the test should create an external directory");
    let external_file_path = external_directory.path().join("must-remain.safetensors");
    fs::write(&external_file_path, b"external").expect("the test should write an external file");
    let active_model_symlink_path =
        active_model_kv_blocks_directory.join(format!("{}.safetensors", "e".repeat(64)));
    symlink(&external_file_path, &active_model_symlink_path)
        .expect("the test should create an active-model safetensors symlink");

    PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            global_prompt_cache_root_directory
                .path()
                .join("active-model")
                .join("revision"),
            global_prompt_cache_root_directory.path().to_path_buf(),
            0,
        ),
        persistent_prompt_cache_model_contract(),
    )
    .expect("startup should remove the symlink itself without following it");

    assert!(external_file_path.exists());
    assert!(fs::symlink_metadata(&active_model_symlink_path).is_err());
}

#[test]
fn should_reject_a_symlink_as_the_global_prompt_cache_root() {
    let configured_prompt_cache_parent_directory =
        tempfile::tempdir().expect("the test should create a configured cache parent");
    let external_directory =
        tempfile::tempdir().expect("the test should create an external directory");
    let external_marker_file_path = external_directory.path().join("must-remain.bin");
    fs::write(&external_marker_file_path, b"external")
        .expect("the test should write an external marker file");
    let symlinked_global_prompt_cache_root_directory = configured_prompt_cache_parent_directory
        .path()
        .join("cache-root");
    symlink(
        external_directory.path(),
        &symlinked_global_prompt_cache_root_directory,
    )
    .expect("the test should symlink the configured global root");

    let open_result = PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            symlinked_global_prompt_cache_root_directory
                .join("active-model")
                .join("revision"),
            symlinked_global_prompt_cache_root_directory,
            0,
        ),
        persistent_prompt_cache_model_contract(),
    );

    assert!(matches!(
        open_result,
        Err(PersistentPromptCacheDiskStoreError::UnsafePromptCacheDirectory { .. })
    ));
    assert!(external_marker_file_path.exists());
    assert!(!external_directory.path().join("active-model").exists());
}

#[test]
fn should_reject_a_symlinked_active_model_directory_component() {
    let global_prompt_cache_root_directory =
        tempfile::tempdir().expect("the test should create a global prompt-cache root");
    let external_directory =
        tempfile::tempdir().expect("the test should create an external directory");
    let symlinked_model_directory = global_prompt_cache_root_directory
        .path()
        .join("active-model");
    symlink(external_directory.path(), &symlinked_model_directory)
        .expect("the test should symlink one active-model path component");

    let open_result = PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            symlinked_model_directory.join("revision"),
            global_prompt_cache_root_directory.path().to_path_buf(),
            0,
        ),
        persistent_prompt_cache_model_contract(),
    );

    assert!(matches!(
        open_result,
        Err(PersistentPromptCacheDiskStoreError::UnsafePromptCacheDirectory { .. })
    ));
    assert!(!external_directory.path().join("revision").exists());
}

#[test]
fn should_reject_an_active_model_directory_outside_the_global_root() {
    let global_prompt_cache_root_directory =
        tempfile::tempdir().expect("the test should create a global prompt-cache root");
    let unrelated_active_model_directory =
        tempfile::tempdir().expect("the test should create an unrelated active-model directory");

    let open_result = PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            unrelated_active_model_directory.path().to_path_buf(),
            global_prompt_cache_root_directory.path().to_path_buf(),
            0,
        ),
        persistent_prompt_cache_model_contract(),
    );

    assert!(matches!(
        open_result,
        Err(
            PersistentPromptCacheDiskStoreError::ActivePromptCacheDirectoryOutsideGlobalRoot { .. }
        )
    ));
}

#[test]
fn should_reject_parent_directory_components_inside_the_active_model_path() {
    let global_prompt_cache_root_directory =
        tempfile::tempdir().expect("the test should create a global prompt-cache root");
    let active_model_prompt_cache_directory = global_prompt_cache_root_directory
        .path()
        .join("model")
        .join("..")
        .join("escaped-revision");

    let open_result = PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            active_model_prompt_cache_directory,
            global_prompt_cache_root_directory.path().to_path_buf(),
            0,
        ),
        persistent_prompt_cache_model_contract(),
    );

    assert!(matches!(
        open_result,
        Err(PersistentPromptCacheDiskStoreError::UnsafePromptCacheDirectory { .. })
    ));
    assert!(
        !global_prompt_cache_root_directory
            .path()
            .join("escaped-revision")
            .exists()
    );
}
