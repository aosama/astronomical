use std::fs;
use std::path::Path;

use super::{
    BlockDirectoryDiscardReason, BlockDirectoryVerdict, classify_block_directory,
    scan_current_format_block_directories,
};
use crate::persistent_cache::block_key::PersistentPromptCacheBlockKey;
use crate::persistent_cache::block_manifest::PersistentPromptCacheBlockManifest;
use crate::persistent_cache::disk_store_index::PersistentPromptCacheDiskStoreIndex;
use crate::persistent_cache::restore_support_tests_support::test_model_contract;
use crate::persistent_cache::startup_cleanup_evidence::PersistentPromptCacheStartupCleanupEvidence;

const BLOCK_HASH_HEX: &str = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";

fn create_block_directory(blocks_directory: &Path, directory_name: &str) -> std::path::PathBuf {
    let block_directory_path = blocks_directory.join(directory_name);
    fs::create_dir(&block_directory_path).expect("block directory");
    fs::write(
        block_directory_path.join("partial.bin"),
        b"to be or not to be",
    )
    .expect("block payload");
    block_directory_path
}

/// The discard reason of a verdict, or `None` when it accepted a candidate.
fn discard_reason(verdict: BlockDirectoryVerdict) -> Option<BlockDirectoryDiscardReason> {
    match verdict {
        BlockDirectoryVerdict::Discard(discard_reason) => Some(discard_reason),
        BlockDirectoryVerdict::Candidate { .. } => None,
    }
}

/// A root block key over one block of a single repeated token id, hashed
/// under the test model's storage contract.
fn root_block_key() -> PersistentPromptCacheBlockKey {
    let model_contract = test_model_contract();
    let block_tokens = vec![1_601_u32; model_contract.block_token_count()];
    PersistentPromptCacheBlockKey::for_root_block(&model_contract, &block_tokens)
        .expect("a root block key")
}

/// Writes the manifest of `block_key` into a directory named `directory_name`.
fn create_manifest_directory(
    blocks_directory: &Path,
    directory_name: &str,
    block_key: &PersistentPromptCacheBlockKey,
) -> std::path::PathBuf {
    let block_directory_path = blocks_directory.join(directory_name);
    fs::create_dir(&block_directory_path).expect("block directory");
    PersistentPromptCacheBlockManifest::new(block_key, None, &test_model_contract())
        .write_to_staging_directory(&block_directory_path)
        .expect("a written manifest");
    block_directory_path
}

#[test]
fn should_judge_a_staging_directory_an_interrupted_transaction() {
    let blocks_directory = tempfile::tempdir().expect("blocks directory");
    let staging_directory = create_block_directory(
        blocks_directory.path(),
        &format!("{BLOCK_HASH_HEX}.staging-4242"),
    );
    let verdict = classify_block_directory(&staging_directory, &test_model_contract())
        .expect("a staging directory is judged, not an error");
    assert_eq!(
        discard_reason(verdict),
        Some(BlockDirectoryDiscardReason::InterruptedTransaction)
    );
}

#[test]
fn should_judge_a_directory_not_named_by_a_hash_corrupt() {
    let blocks_directory = tempfile::tempdir().expect("blocks directory");
    for foreign_name in [
        "lost+found",
        &BLOCK_HASH_HEX[..63],
        &BLOCK_HASH_HEX.to_ascii_uppercase(),
    ] {
        let foreign_directory = create_block_directory(blocks_directory.path(), foreign_name);
        let verdict = classify_block_directory(&foreign_directory, &test_model_contract())
            .expect("a foreign directory is judged, not an error");
        assert_eq!(
            discard_reason(verdict),
            Some(BlockDirectoryDiscardReason::CorruptCurrentFormat),
            "{foreign_name}"
        );
    }
}

#[test]
fn should_judge_a_hash_named_directory_without_a_manifest_corrupt() {
    let blocks_directory = tempfile::tempdir().expect("blocks directory");
    let manifestless_directory = create_block_directory(blocks_directory.path(), BLOCK_HASH_HEX);
    let verdict = classify_block_directory(&manifestless_directory, &test_model_contract())
        .expect("a missing manifest is judged, not an error");
    assert_eq!(
        discard_reason(verdict),
        Some(BlockDirectoryDiscardReason::CorruptCurrentFormat)
    );
}

#[test]
fn should_remove_rejected_directories_and_leave_stray_files_alone() {
    let blocks_directory = tempfile::tempdir().expect("blocks directory");
    let staging_directory = create_block_directory(
        blocks_directory.path(),
        &format!("{BLOCK_HASH_HEX}.staging-7"),
    );
    let manifestless_directory = create_block_directory(blocks_directory.path(), BLOCK_HASH_HEX);
    let stray_file = blocks_directory.path().join("notes.txt");
    fs::write(&stray_file, b"stray").expect("stray file");
    let mut tracked_files = PersistentPromptCacheDiskStoreIndex::default();
    let mut startup_cleanup_evidence = PersistentPromptCacheStartupCleanupEvidence::default();

    scan_current_format_block_directories(
        blocks_directory.path(),
        &mut tracked_files,
        &test_model_contract(),
        &mut startup_cleanup_evidence,
    )
    .expect("a scan of rejected directories succeeds");

    assert!(!staging_directory.exists() && !manifestless_directory.exists());
    assert!(stray_file.is_file());
    assert!(tracked_files.tracked_blocks().is_empty());
    assert_eq!(
        startup_cleanup_evidence
            .interrupted_transaction_recovery
            .block_count,
        1
    );
    assert_eq!(
        startup_cleanup_evidence.corrupt_current_format.block_count,
        1
    );
}

#[test]
fn should_scan_an_empty_blocks_directory_without_evidence() {
    let blocks_directory = tempfile::tempdir().expect("blocks directory");
    let mut tracked_files = PersistentPromptCacheDiskStoreIndex::default();
    let mut startup_cleanup_evidence = PersistentPromptCacheStartupCleanupEvidence::default();
    scan_current_format_block_directories(
        blocks_directory.path(),
        &mut tracked_files,
        &test_model_contract(),
        &mut startup_cleanup_evidence,
    )
    .expect("an empty scan succeeds");
    assert_eq!(
        startup_cleanup_evidence,
        PersistentPromptCacheStartupCleanupEvidence::default()
    );
}

#[test]
fn should_fail_the_scan_when_the_blocks_directory_is_missing() {
    let parent_directory = tempfile::tempdir().expect("parent directory");
    let mut tracked_files = PersistentPromptCacheDiskStoreIndex::default();
    let mut startup_cleanup_evidence = PersistentPromptCacheStartupCleanupEvidence::default();
    assert!(
        scan_current_format_block_directories(
            &parent_directory.path().join("blocks"),
            &mut tracked_files,
            &test_model_contract(),
            &mut startup_cleanup_evidence,
        )
        .is_err()
    );
}

#[test]
fn should_judge_a_directory_whose_manifest_names_its_own_hash_a_candidate() {
    let blocks_directory = tempfile::tempdir().expect("blocks directory");
    let block_key = root_block_key();
    let block_directory_path = create_manifest_directory(
        blocks_directory.path(),
        &astronomical_lowercase_hex::encode_lowercase_hex(&block_key.block_hash()),
        &block_key,
    );
    let verdict = classify_block_directory(&block_directory_path, &test_model_contract())
        .expect("a well-formed block is judged");
    let BlockDirectoryVerdict::Candidate {
        block_hash,
        block_candidate,
    } = verdict
    else {
        panic!("a manifest naming its own directory must be a candidate");
    };
    assert_eq!(block_hash, block_key.block_hash());
    assert_eq!(block_candidate.block_index, 0);
    assert_eq!(block_candidate.parent_block_hash, None);
}

#[test]
fn should_judge_a_manifest_naming_another_hash_corrupt() {
    let blocks_directory = tempfile::tempdir().expect("blocks directory");
    let block_key = root_block_key();
    let misnamed_directory =
        create_manifest_directory(blocks_directory.path(), BLOCK_HASH_HEX, &block_key);
    assert_ne!(
        astronomical_lowercase_hex::encode_lowercase_hex(&block_key.block_hash()),
        BLOCK_HASH_HEX
    );
    let verdict = classify_block_directory(&misnamed_directory, &test_model_contract())
        .expect("a misnamed block is judged, not an error");
    assert_eq!(
        discard_reason(verdict),
        Some(BlockDirectoryDiscardReason::CorruptCurrentFormat)
    );
}
