//! Hermetic filesystem-only tests for persistent prompt-cache disk-store
//! startup scan: invalid-file deletion and quota enforcement.
//!
//! These tests exercise `PersistentPromptCacheDiskStore::open()` and quota
//! accounting without allocating MLX tensors. The disk store type is gated
//! behind `direct-mlx` because it depends on `MlxRuntime` for save/load,
//! but `open()` itself is pure filesystem I/O.

use std::fs;
use std::fs::FileTimes;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, UNIX_EPOCH};

use astronomical_model_serving::{
    PersistentPromptCacheDiskStore, PersistentPromptCacheDiskStoreConfig,
    PersistentPromptCacheDiskStoreError,
};

use crate::common::qwen3_5_moe::persistent_prompt_cache_model_contract;

const ONE_BYTE_QUOTA: u64 = 1;
const FIRST_CROSS_MODEL_FILE_BYTE_COUNT: usize = 1_024;
const SECOND_CROSS_MODEL_FILE_BYTE_COUNT: usize = 1_024;

fn write_cross_model_prompt_cache_file(
    global_prompt_cache_root_directory: &Path,
    model_directory_name: &str,
    file_name_hex_digit: char,
    file_byte_count: usize,
) -> PathBuf {
    let cross_model_kv_blocks_directory = global_prompt_cache_root_directory
        .join(model_directory_name)
        .join("revision")
        .join("kv_blocks");
    fs::create_dir_all(&cross_model_kv_blocks_directory)
        .expect("the test should create a cross-model KV blocks directory");
    let cross_model_prompt_cache_file_path = cross_model_kv_blocks_directory.join(format!(
        "{}.safetensors",
        file_name_hex_digit.to_string().repeat(64)
    ));
    fs::write(
        &cross_model_prompt_cache_file_path,
        vec![0_u8; file_byte_count],
    )
    .expect("the test should write a cross-model prompt-cache file");
    cross_model_prompt_cache_file_path
}

fn recursive_non_directory_byte_count(directory: &Path) -> u64 {
    let mut pending_directories = vec![directory.to_path_buf()];
    let mut total_non_directory_bytes = 0_u64;
    while let Some(pending_directory) = pending_directories.pop() {
        for directory_entry in
            fs::read_dir(&pending_directory).expect("the test should read a cache directory")
        {
            let directory_entry = directory_entry.expect("the test should read a cache entry");
            let directory_entry_file_type = directory_entry
                .file_type()
                .expect("the test should read the cache entry type");
            if directory_entry_file_type.is_dir() {
                pending_directories.push(directory_entry.path());
            } else {
                total_non_directory_bytes = total_non_directory_bytes.saturating_add(
                    directory_entry
                        .metadata()
                        .expect("the test should read cache entry metadata")
                        .len(),
                );
            }
        }
    }
    total_non_directory_bytes
}

#[test]
fn should_enforce_one_global_prompt_cache_quota_across_model_directories() {
    let global_prompt_cache_root_directory =
        tempfile::tempdir().expect("the test should create a global prompt-cache root");
    let first_cross_model_prompt_cache_file_path = write_cross_model_prompt_cache_file(
        global_prompt_cache_root_directory.path(),
        "z-older-model",
        'a',
        FIRST_CROSS_MODEL_FILE_BYTE_COUNT,
    );
    let second_cross_model_prompt_cache_file_path = write_cross_model_prompt_cache_file(
        global_prompt_cache_root_directory.path(),
        "a-newer-model",
        'b',
        SECOND_CROSS_MODEL_FILE_BYTE_COUNT,
    );
    assert!(first_cross_model_prompt_cache_file_path.exists());
    assert!(second_cross_model_prompt_cache_file_path.exists());

    let global_prompt_cache_maximum_size_bytes =
        u64::try_from(FIRST_CROSS_MODEL_FILE_BYTE_COUNT + SECOND_CROSS_MODEL_FILE_BYTE_COUNT - 1)
            .expect("the test byte count should fit u64");
    let active_model_prompt_cache_directory = global_prompt_cache_root_directory
        .path()
        .join("third-active-model")
        .join("revision");

    let persistent_prompt_cache = PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            active_model_prompt_cache_directory,
            global_prompt_cache_root_directory.path().to_path_buf(),
            global_prompt_cache_maximum_size_bytes,
        ),
        persistent_prompt_cache_model_contract(),
    )
    .expect("the active model prompt cache should enforce the global root quota");

    let actual_global_prompt_cache_size_bytes =
        recursive_non_directory_byte_count(global_prompt_cache_root_directory.path());
    assert!(
        actual_global_prompt_cache_size_bytes <= global_prompt_cache_maximum_size_bytes,
        "global prompt-cache bytes must not exceed one configured maximum: actual={actual_global_prompt_cache_size_bytes}, maximum={global_prompt_cache_maximum_size_bytes}"
    );
    assert_eq!(
        persistent_prompt_cache.total_size_bytes(),
        actual_global_prompt_cache_size_bytes,
        "reported prompt-cache bytes must describe global root usage"
    );
}

#[test]
fn should_evict_the_oldest_written_cross_model_file_first() {
    let global_prompt_cache_root_directory =
        tempfile::tempdir().expect("the test should create a global prompt-cache root");
    let older_cross_model_prompt_cache_file_path = write_cross_model_prompt_cache_file(
        global_prompt_cache_root_directory.path(),
        "z-older-model",
        'f',
        FIRST_CROSS_MODEL_FILE_BYTE_COUNT,
    );
    let newer_cross_model_prompt_cache_file_path = write_cross_model_prompt_cache_file(
        global_prompt_cache_root_directory.path(),
        "a-newer-model",
        'a',
        SECOND_CROSS_MODEL_FILE_BYTE_COUNT,
    );
    fs::File::open(&older_cross_model_prompt_cache_file_path)
        .expect("the test should open the older cross-model file")
        .set_times(FileTimes::new().set_modified(UNIX_EPOCH + Duration::from_secs(10)))
        .expect("the test should persist the older modification time");
    fs::File::open(&newer_cross_model_prompt_cache_file_path)
        .expect("the test should open the newer cross-model file")
        .set_times(FileTimes::new().set_modified(UNIX_EPOCH + Duration::from_secs(20)))
        .expect("the test should persist the newer modification time");
    assert!(older_cross_model_prompt_cache_file_path.exists());
    assert!(newer_cross_model_prompt_cache_file_path.exists());

    let global_prompt_cache_maximum_size_bytes = u64::try_from(SECOND_CROSS_MODEL_FILE_BYTE_COUNT)
        .expect("the test byte count should fit u64");
    PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            global_prompt_cache_root_directory
                .path()
                .join("third-active-model")
                .join("revision"),
            global_prompt_cache_root_directory.path().to_path_buf(),
            global_prompt_cache_maximum_size_bytes,
        ),
        persistent_prompt_cache_model_contract(),
    )
    .expect("the active model prompt cache should evict the oldest-written file");

    assert!(!older_cross_model_prompt_cache_file_path.exists());
    assert!(newer_cross_model_prompt_cache_file_path.exists());
}

#[test]
fn should_return_typed_error_when_cross_model_global_eviction_fails() {
    let global_prompt_cache_root_directory =
        tempfile::tempdir().expect("the test should create a global prompt-cache root");
    let cross_model_prompt_cache_file_path = write_cross_model_prompt_cache_file(
        global_prompt_cache_root_directory.path(),
        "cross-model",
        'c',
        FIRST_CROSS_MODEL_FILE_BYTE_COUNT,
    );
    let cross_model_kv_blocks_directory = cross_model_prompt_cache_file_path
        .parent()
        .expect("the cross-model file should have a parent");
    let original_directory_mode = fs::metadata(cross_model_kv_blocks_directory)
        .expect("the test should read cross-model directory metadata")
        .permissions()
        .mode();
    fs::set_permissions(
        cross_model_kv_blocks_directory,
        fs::Permissions::from_mode(0o555),
    )
    .expect("the test should prevent cross-model deletion");

    let open_result = PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            global_prompt_cache_root_directory
                .path()
                .join("active-model")
                .join("revision"),
            global_prompt_cache_root_directory.path().to_path_buf(),
            0,
        ),
        persistent_prompt_cache_model_contract(),
    );

    fs::set_permissions(
        cross_model_kv_blocks_directory,
        fs::Permissions::from_mode(original_directory_mode),
    )
    .expect("the test should restore cross-model directory permissions");
    match open_result {
        Err(PersistentPromptCacheDiskStoreError::RemovePromptCacheFile {
            persistent_prompt_cache_file_path,
            ..
        }) => assert_eq!(
            persistent_prompt_cache_file_path,
            cross_model_prompt_cache_file_path
        ),
        Err(other_error) => panic!("expected typed removal error, got {other_error}"),
        Ok(_) => panic!("global quota enforcement must not hide deletion failure"),
    }
    assert!(cross_model_prompt_cache_file_path.exists());
    fs::remove_file(&cross_model_prompt_cache_file_path)
        .expect("the test should remove the retained cross-model file");
}

#[test]
fn should_delete_cross_model_stale_writer_temp_below_global_quota() {
    let global_prompt_cache_root_directory =
        tempfile::tempdir().expect("the test should create a global prompt-cache root");
    let cross_model_kv_blocks_directory = global_prompt_cache_root_directory
        .path()
        .join("cross-model")
        .join("revision")
        .join("kv_blocks");
    fs::create_dir_all(&cross_model_kv_blocks_directory)
        .expect("the test should create a cross-model KV blocks directory");
    let stale_writer_temp_file_path =
        cross_model_kv_blocks_directory.join(format!("{}.safetensors.tmp", "d".repeat(64)));
    fs::write(&stale_writer_temp_file_path, vec![0_u8; 128])
        .expect("the test should write a stale writer temp file");
    assert!(stale_writer_temp_file_path.exists());

    PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            global_prompt_cache_root_directory
                .path()
                .join("active-model")
                .join("revision"),
            global_prompt_cache_root_directory.path().to_path_buf(),
            10_000,
        ),
        persistent_prompt_cache_model_contract(),
    )
    .expect("the global cache should remove stale writer temp files");

    assert!(!stale_writer_temp_file_path.exists());
}

/// REGRESSION: a valid-hash-named but invalid-content `.safetensors` file
/// under a one-byte quota must be deleted during `open()` so that
/// cache-owned bytes cannot silently escape accounting.
#[test]
fn should_delete_invalid_content_safetensors_file_under_one_byte_quota() {
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let kv_blocks_directory = persistent_prompt_cache_directory.path().join("kv_blocks");
    fs::create_dir_all(&kv_blocks_directory)
        .expect("the test should create the KV blocks directory");

    let invalid_kv_block_file_path =
        kv_blocks_directory.join(format!("{}.safetensors", "0".repeat(64)));
    fs::write(&invalid_kv_block_file_path, b"not a safetensors file")
        .expect("the test should write an invalid current-format file");

    let persistent_prompt_cache = PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            persistent_prompt_cache_directory.path().to_path_buf(),
            persistent_prompt_cache_directory.path().to_path_buf(),
            ONE_BYTE_QUOTA,
        ),
        persistent_prompt_cache_model_contract(),
    )
    .expect("the persistent prompt cache should open and delete invalid files");

    assert_eq!(persistent_prompt_cache.sequence_state_block_count(), 0);
    assert_eq!(
        persistent_prompt_cache.total_size_bytes(),
        0,
        "cache-owned bytes must be zero after deleting the only invalid file"
    );
    assert!(
        !invalid_kv_block_file_path.exists(),
        "invalid cache-owned .safetensors file must be deleted so it does not consume disk capacity beyond the quota"
    );
}

/// When `open()` encounters a cache-owned `.safetensors` file that cannot be
/// deleted (permission denied on the parent directory), it must return
/// `RemovePromptCacheFile` rather than silently leaving the file untracked.
#[test]
fn should_return_remove_prompt_cache_file_error_when_deletion_fails() {
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let kv_blocks_directory = persistent_prompt_cache_directory.path().join("kv_blocks");
    fs::create_dir_all(&kv_blocks_directory)
        .expect("the test should create the KV blocks directory");

    let invalid_kv_block_file_path =
        kv_blocks_directory.join(format!("{}.safetensors", "0".repeat(64)));
    fs::write(&invalid_kv_block_file_path, b"not a safetensors file")
        .expect("the test should write an invalid current-format file");

    let original_directory_mode = fs::metadata(&kv_blocks_directory)
        .expect("the test should read directory metadata")
        .permissions()
        .mode();

    fs::set_permissions(&kv_blocks_directory, fs::Permissions::from_mode(0o555))
        .expect("the test should set directory to read-execute only");

    let open_result = PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            persistent_prompt_cache_directory.path().to_path_buf(),
            persistent_prompt_cache_directory.path().to_path_buf(),
            ONE_BYTE_QUOTA,
        ),
        persistent_prompt_cache_model_contract(),
    );

    fs::set_permissions(
        &kv_blocks_directory,
        fs::Permissions::from_mode(original_directory_mode),
    )
    .expect("the test must restore directory permissions for tempdir cleanup");
    fs::remove_file(&invalid_kv_block_file_path)
        .expect("the test must remove the invalid file after restoring permissions");

    match open_result {
        Err(PersistentPromptCacheDiskStoreError::RemovePromptCacheFile {
            persistent_prompt_cache_file_path,
            ..
        }) => assert_eq!(
            persistent_prompt_cache_file_path,
            invalid_kv_block_file_path
        ),
        Err(other_error) => {
            panic!("expected RemovePromptCacheFile, got: {other_error}");
        }
        Ok(_) => {
            panic!("expected RemovePromptCacheFile because directory was read-only");
        }
    }
}
