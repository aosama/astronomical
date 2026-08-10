//! Global-quota acceptance tests for foreign namespaces and adversarial topology.
//!
//! The active store must safely account and evict cache artifacts belonging to
//! other models without interpreting matching hashes as shared ancestry.

use std::{fs, path::Path, time::UNIX_EPOCH};

use astronomical_model_serving::{
    PersistentPromptCacheDiskStore, PersistentPromptCacheDiskStoreConfig,
};

use super::persistent_prompt_cache_disk_store_support::*;
use crate::common::qwen3_5_moe::persistent_prompt_cache_model_contract;

const LARGE_CACHE_LIMIT_BYTES: u64 = 10 * 1024 * 1024 * 1024;

#[tokio::test]
async fn should_scope_protected_ancestry_to_its_model_namespace_when_hashes_match() {
    // Duplicate one real block under a foreign model/revision. If protection were
    // hash-only, the foreign bytes would become unevictable with the active parent.
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let measurement_directory =
        tempfile::tempdir().expect("the test should create a measurement directory");
    let measurement_cache =
        open_persistent_prompt_cache_disk_store(&measurement_directory, LARGE_CACHE_LIMIT_BYTES)
            .expect("the measurement cache should open");
    let parent_block_key = persistent_prompt_cache_block_key_for_seed(0);
    let child_block_key = parent_block_key
        .for_child_block(&block_tokens_for_seed(10_000))
        .expect("the child identity should resolve");
    let sequence_state_tensors = synthetic_kv_block_tensors(&runtime);
    let boundary_state_tensors = synthetic_recurrent_snapshot_tensors(&runtime);
    measurement_cache
        .publish_block(
            &runtime,
            &parent_block_key,
            None,
            &sequence_state_tensors,
            &boundary_state_tensors,
        )
        .expect("the measurement block should publish");
    let measured_block_size_bytes = measurement_cache.total_size_bytes();
    drop(measurement_cache);

    let two_block_quota_bytes = measured_block_size_bytes
        .checked_mul(2)
        .and_then(|two_blocks| two_blocks.checked_add(1_024))
        .expect("the two-block quota should fit");
    let global_prompt_cache_root =
        tempfile::tempdir().expect("the test should create a global cache root");
    let active_cache =
        open_persistent_prompt_cache_disk_store(&global_prompt_cache_root, two_block_quota_bytes)
            .expect("the active cache should open");
    active_cache
        .publish_block(
            &runtime,
            &parent_block_key,
            None,
            &sequence_state_tensors,
            &boundary_state_tensors,
        )
        .expect("the active parent should publish");
    let parent_block_directory = global_prompt_cache_root
        .path()
        .join("blocks")
        .join(hex::encode(parent_block_key.block_hash()));
    let foreign_same_hash_directory = global_prompt_cache_root
        .path()
        .join("foreign-model")
        .join("foreign-revision")
        .join("blocks")
        .join(hex::encode(parent_block_key.block_hash()));
    fs::create_dir_all(&foreign_same_hash_directory)
        .expect("the foreign block directory should be created");
    fs::copy(
        parent_block_directory.join("manifest.json"),
        foreign_same_hash_directory.join("manifest.json"),
    )
    .expect("the foreign block should copy the same-hash manifest");
    fs::write(
        foreign_same_hash_directory.join("foreign-payload.bin"),
        vec![0_u8; 4_096],
    )
    .expect("the foreign block should consume the remaining quota");
    fs::File::open(&foreign_same_hash_directory)
        .expect("the foreign block directory should open")
        .set_times(std::fs::FileTimes::new().set_modified(UNIX_EPOCH))
        .expect("the foreign block should become the oldest candidate");

    active_cache
        .publish_block(
            &runtime,
            &child_block_key,
            Some(&parent_block_key),
            &sequence_state_tensors,
            &boundary_state_tensors,
        )
        .expect("the child should evict only the foreign same-hash namespace");

    assert!(parent_block_directory.is_dir());
    assert!(
        global_prompt_cache_root
            .path()
            .join("blocks")
            .join(hex::encode(child_block_key.block_hash()))
            .is_dir()
    );
    assert!(!foreign_same_hash_directory.exists());
}

#[test]
fn should_bound_global_quota_topology_walk_for_cyclic_foreign_manifests() {
    // Foreign files bypass active-model startup validation. The global scanner
    // therefore needs its own iterative cycle guard before applying quota.
    let global_prompt_cache_root =
        tempfile::tempdir().expect("the test should create a global cache root");
    let foreign_blocks_directory = global_prompt_cache_root
        .path()
        .join("foreign-model")
        .join("foreign-revision")
        .join("blocks");
    let first_block_hash = "a".repeat(64);
    let second_block_hash = "b".repeat(64);
    write_cyclic_foreign_block(
        &foreign_blocks_directory,
        &first_block_hash,
        &second_block_hash,
    );
    write_cyclic_foreign_block(
        &foreign_blocks_directory,
        &second_block_hash,
        &first_block_hash,
    );
    let active_model_directory = global_prompt_cache_root
        .path()
        .join("active-model")
        .join("active-revision");

    PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            active_model_directory,
            global_prompt_cache_root.path().to_path_buf(),
            1,
        ),
        persistent_prompt_cache_model_contract(),
    )
    .expect("cyclic foreign manifests should be evicted without recursive overflow");

    assert!(!foreign_blocks_directory.join(first_block_hash).exists());
    assert!(!foreign_blocks_directory.join(second_block_hash).exists());
}

#[test]
fn should_not_join_foreign_block_subtrees_across_storage_contract_fingerprints() {
    // The child names the parent's hash but declares incompatible tensor geometry.
    // Evicting the parent must not sweep that independently scoped child.
    let global_prompt_cache_root =
        tempfile::tempdir().expect("the test should create a global cache root");
    let foreign_blocks_directory = global_prompt_cache_root
        .path()
        .join("foreign-model")
        .join("foreign-revision")
        .join("blocks");
    let parent_block_hash = "a".repeat(64);
    let child_block_hash = "b".repeat(64);
    write_foreign_block(
        &foreign_blocks_directory,
        &parent_block_hash,
        None,
        &"1".repeat(64),
        128,
        UNIX_EPOCH,
    );
    write_foreign_block(
        &foreign_blocks_directory,
        &child_block_hash,
        Some(&parent_block_hash),
        &"2".repeat(64),
        128,
        UNIX_EPOCH + std::time::Duration::from_secs(10),
    );
    let child_directory = foreign_blocks_directory.join(&child_block_hash);
    let child_file_size_bytes = directory_file_size_bytes(&child_directory);

    open_foreign_quota_trigger(&global_prompt_cache_root, child_file_size_bytes);

    assert!(!foreign_blocks_directory.join(parent_block_hash).exists());
    assert!(child_directory.is_dir());
}

#[test]
fn should_remove_stale_transactions_before_evicting_committed_blocks() {
    // Make durable content older than staging content to prove transaction class,
    // rather than timestamp alone, controls the first eviction decision.
    let global_prompt_cache_root =
        tempfile::tempdir().expect("the test should create a global cache root");
    let foreign_blocks_directory = global_prompt_cache_root
        .path()
        .join("foreign-model")
        .join("foreign-revision")
        .join("blocks");
    let committed_block_hash = "c".repeat(64);
    write_foreign_block(
        &foreign_blocks_directory,
        &committed_block_hash,
        None,
        &"3".repeat(64),
        128,
        UNIX_EPOCH,
    );
    let committed_block_directory = foreign_blocks_directory.join(&committed_block_hash);
    let committed_block_size_bytes = directory_file_size_bytes(&committed_block_directory);
    let stale_transaction_directory =
        foreign_blocks_directory.join(format!("{}.staging-interrupted", "d".repeat(64)));
    fs::create_dir_all(&stale_transaction_directory)
        .expect("the stale transaction directory should be created");
    fs::write(
        stale_transaction_directory.join("sequence.safetensors.tmp"),
        vec![0_u8; 512],
    )
    .expect("the stale transaction should consume quota");

    open_foreign_quota_trigger(&global_prompt_cache_root, committed_block_size_bytes);

    assert!(committed_block_directory.is_dir());
    assert!(!stale_transaction_directory.exists());
}

fn write_cyclic_foreign_block(
    foreign_blocks_directory: &Path,
    block_hash: &str,
    parent_block_hash: &str,
) {
    let block_directory = foreign_blocks_directory.join(block_hash);
    fs::create_dir_all(&block_directory).expect("the cyclic block directory should be created");
    let manifest = serde_json::json!({
        "format_version": "11",
        "block_hash": block_hash,
        "block_index": 1,
        "parent_block_hash": parent_block_hash,
        "storage_contract_fingerprint": "foreign-contract",
        "has_sequence_state": true,
        "has_boundary_state": true,
    });
    fs::write(block_directory.join("manifest.json"), manifest.to_string())
        .expect("the cyclic manifest should write");
    fs::write(block_directory.join("payload.bin"), vec![0_u8; 128])
        .expect("the cyclic block payload should write");
}

fn write_foreign_block(
    foreign_blocks_directory: &Path,
    block_hash: &str,
    parent_block_hash: Option<&str>,
    storage_contract_fingerprint: &str,
    payload_byte_count: usize,
    modified_at: std::time::SystemTime,
) {
    let block_directory = foreign_blocks_directory.join(block_hash);
    fs::create_dir_all(&block_directory).expect("the foreign block directory should be created");
    let manifest = serde_json::json!({
        "format_version": "11",
        "block_hash": block_hash,
        "block_index": usize::from(parent_block_hash.is_some()),
        "parent_block_hash": parent_block_hash,
        "storage_contract_fingerprint": storage_contract_fingerprint,
        "has_sequence_state": true,
        "has_boundary_state": true,
    });
    fs::write(block_directory.join("manifest.json"), manifest.to_string())
        .expect("the foreign manifest should write");
    fs::write(
        block_directory.join("payload.bin"),
        vec![0_u8; payload_byte_count],
    )
    .expect("the foreign block payload should write");
    fs::File::open(&block_directory)
        .expect("the foreign block directory should open")
        .set_times(std::fs::FileTimes::new().set_modified(modified_at))
        .expect("the foreign block timestamp should write");
}

fn directory_file_size_bytes(directory_path: &Path) -> u64 {
    fs::read_dir(directory_path)
        .expect("the foreign block directory should be readable")
        .map(|directory_entry| {
            directory_entry
                .expect("the foreign block entry should be readable")
                .metadata()
                .expect("the foreign block metadata should be readable")
                .len()
        })
        .sum()
}

fn open_foreign_quota_trigger(
    global_prompt_cache_root: &tempfile::TempDir,
    global_prompt_cache_maximum_size_bytes: u64,
) {
    let active_model_directory = global_prompt_cache_root
        .path()
        .join("active-model")
        .join("active-revision");
    PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            active_model_directory,
            global_prompt_cache_root.path().to_path_buf(),
            global_prompt_cache_maximum_size_bytes,
        ),
        persistent_prompt_cache_model_contract(),
    )
    .expect("global quota reconciliation should complete");
}
