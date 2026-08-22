//! Filesystem acceptance coverage for bounded startup-cleanup attribution.

use std::{fs, path::Path};

use astronomical_model_serving::{
    PersistentPromptCacheDiskStore, PersistentPromptCacheDiskStoreConfig,
};

use crate::common::qwen3_5_moe::persistent_prompt_cache_model_contract;

const LARGE_CACHE_LIMIT_BYTES: u64 = 1_000_000_000;

#[test]
fn should_classify_startup_cleanup_and_consume_evidence_once() {
    let global_prompt_cache_root =
        tempfile::tempdir().expect("the test should create a global prompt-cache root");
    let active_model_directory = global_prompt_cache_root
        .path()
        .join("fictional-model")
        .join("fictional-revision");
    let blocks_directory = active_model_directory.join("blocks");
    fs::create_dir_all(&blocks_directory).expect("the test should create the blocks directory");

    let abandoned_transaction_directory =
        blocks_directory.join(format!("{}.staging-interrupted", "a".repeat(64)));
    fs::create_dir_all(&abandoned_transaction_directory)
        .expect("the test should create an abandoned transaction");
    fs::write(
        abandoned_transaction_directory.join("sequence.safetensors.tmp"),
        vec![1_u8; 31],
    )
    .expect("the test should write abandoned transaction bytes");
    let abandoned_transaction_byte_count =
        directory_file_size_bytes(&abandoned_transaction_directory);

    let corrupt_block_directory = blocks_directory.join("b".repeat(64));
    fs::create_dir_all(&corrupt_block_directory)
        .expect("the test should create a corrupt current-format block");
    fs::write(corrupt_block_directory.join("manifest.json"), b"not-json")
        .expect("the test should write a corrupt manifest");
    fs::write(corrupt_block_directory.join("payload.bin"), vec![2_u8; 37])
        .expect("the test should write corrupt block bytes");
    let corrupt_block_byte_count = directory_file_size_bytes(&corrupt_block_directory);

    let obsolete_sequence_directory = active_model_directory.join("kv_blocks");
    let obsolete_boundary_directory = active_model_directory.join("recurrent_snapshots");
    fs::create_dir_all(&obsolete_sequence_directory)
        .expect("the test should create obsolete sequence storage");
    fs::create_dir_all(&obsolete_boundary_directory)
        .expect("the test should create obsolete boundary storage");
    let obsolete_sequence_file =
        obsolete_sequence_directory.join(format!("{}.safetensors", "c".repeat(64)));
    let obsolete_boundary_file =
        obsolete_boundary_directory.join(format!("{}.safetensors", "d".repeat(64)));
    fs::write(&obsolete_sequence_file, vec![3_u8; 41])
        .expect("the test should write obsolete sequence bytes");
    fs::write(&obsolete_boundary_file, vec![4_u8; 43])
        .expect("the test should write obsolete boundary bytes");

    let prompt_cache = open_store(
        global_prompt_cache_root.path(),
        active_model_directory,
        LARGE_CACHE_LIMIT_BYTES,
    );
    let startup_cleanup_evidence = prompt_cache
        .startup_cleanup_evidence()
        .expect("startup cleanup should retain bounded evidence");

    assert_eq!(
        startup_cleanup_evidence
            .interrupted_transaction_recovery
            .block_count,
        1
    );
    assert_eq!(
        startup_cleanup_evidence
            .interrupted_transaction_recovery
            .byte_count,
        abandoned_transaction_byte_count
    );
    assert_eq!(startup_cleanup_evidence.obsolete_format.artifact_count, 2);
    assert_eq!(startup_cleanup_evidence.obsolete_format.byte_count, 84);
    assert_eq!(
        startup_cleanup_evidence.corrupt_current_format.block_count,
        1
    );
    assert_eq!(
        startup_cleanup_evidence.corrupt_current_format.byte_count,
        corrupt_block_byte_count
    );
    assert!(!abandoned_transaction_directory.exists());
    assert!(!corrupt_block_directory.exists());
    assert!(!obsolete_sequence_file.exists());
    assert!(!obsolete_boundary_file.exists());

    assert_eq!(
        prompt_cache.take_startup_cleanup_evidence(),
        Some(startup_cleanup_evidence)
    );
    assert_eq!(prompt_cache.take_startup_cleanup_evidence(), None);
}

#[test]
fn should_count_startup_quota_eviction_by_artifact_and_block() {
    let global_prompt_cache_root =
        tempfile::tempdir().expect("the test should create a global prompt-cache root");
    let foreign_revision_directory = global_prompt_cache_root
        .path()
        .join("foreign-model")
        .join("foreign-revision");
    let foreign_visual_embedding_directory = foreign_revision_directory.join("visual_embeddings");
    fs::create_dir_all(&foreign_visual_embedding_directory)
        .expect("the test should create foreign standalone storage");
    let foreign_standalone_file =
        foreign_visual_embedding_directory.join(format!("{}.safetensors", "e".repeat(64)));
    fs::write(&foreign_standalone_file, vec![5_u8; 47])
        .expect("the test should write a foreign standalone artifact");

    let foreign_block_directory = foreign_revision_directory
        .join("blocks")
        .join("f".repeat(64));
    fs::create_dir_all(&foreign_block_directory)
        .expect("the test should create a foreign block directory");
    let foreign_manifest = serde_json::json!({
        "format_version": "12",
        "block_hash": "f".repeat(64),
        "block_index": 0,
        "parent_block_hash": null,
        "storage_contract_fingerprint": "fictional-foreign-contract",
        "has_sequence_state": true,
        "has_boundary_state": true,
    });
    fs::write(
        foreign_block_directory.join("manifest.json"),
        foreign_manifest.to_string(),
    )
    .expect("the test should write a foreign block manifest");
    fs::write(foreign_block_directory.join("payload.bin"), vec![6_u8; 53])
        .expect("the test should write a foreign block payload");
    let foreign_block_byte_count = directory_file_size_bytes(&foreign_block_directory);

    let prompt_cache = open_store(
        global_prompt_cache_root.path(),
        global_prompt_cache_root
            .path()
            .join("active-model")
            .join("active-revision"),
        0,
    );
    let quota_eviction = prompt_cache
        .startup_cleanup_evidence()
        .expect("quota cleanup should retain evidence")
        .quota_eviction;

    assert_eq!(quota_eviction.artifact_count, 1);
    assert_eq!(quota_eviction.block_count, 1);
    assert_eq!(quota_eviction.byte_count, 47 + foreign_block_byte_count);
    assert!(!foreign_standalone_file.exists());
    assert!(!foreign_block_directory.exists());
}

fn open_store(
    global_prompt_cache_root_directory: &Path,
    active_model_directory: std::path::PathBuf,
    global_prompt_cache_maximum_size_bytes: u64,
) -> PersistentPromptCacheDiskStore {
    PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            active_model_directory,
            global_prompt_cache_root_directory.to_path_buf(),
            global_prompt_cache_maximum_size_bytes,
        ),
        persistent_prompt_cache_model_contract(),
    )
    .expect("the prompt cache should open")
}

fn directory_file_size_bytes(directory_path: &Path) -> u64 {
    let mut pending_directories = vec![directory_path.to_path_buf()];
    let mut total_size_bytes = 0_u64;
    while let Some(pending_directory) = pending_directories.pop() {
        for directory_entry in
            fs::read_dir(pending_directory).expect("the test should read the directory")
        {
            let directory_entry = directory_entry.expect("the test should read an entry");
            let file_type = directory_entry
                .file_type()
                .expect("the test should read the entry type");
            if file_type.is_dir() {
                pending_directories.push(directory_entry.path());
            } else {
                total_size_bytes = total_size_bytes.saturating_add(
                    directory_entry
                        .metadata()
                        .expect("the test should read entry metadata")
                        .len(),
                );
            }
        }
    }
    total_size_bytes
}
