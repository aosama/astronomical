use astronomical_model_serving::{
    PersistentPromptCacheBlockSaveAdmission,
    persistent_prompt_cache_recurrent_snapshot_is_common_prefix_checkpoint,
    persistent_prompt_cache_save_admission,
};

const KV_BLOCK_BYTES: u64 = 40_000_000;
const RECURRENT_SNAPSHOT_BYTES: u64 = 61_500_000;
const RETENTION_POLICY_CHILD_BLOCK_INDEX: u32 = 100;
const RETENTION_POLICY_MAXIMUM_SIZE_BYTES: u64 =
    (RETENTION_POLICY_CHILD_BLOCK_INDEX as u64 + 2) * KV_BLOCK_BYTES;

#[test]
fn should_mark_common_prefix_recurrent_snapshot_checkpoints() {
    assert!(persistent_prompt_cache_recurrent_snapshot_is_common_prefix_checkpoint(0));
    assert!(!persistent_prompt_cache_recurrent_snapshot_is_common_prefix_checkpoint(1));
    assert!(!persistent_prompt_cache_recurrent_snapshot_is_common_prefix_checkpoint(2));
    assert!(persistent_prompt_cache_recurrent_snapshot_is_common_prefix_checkpoint(3));
    assert!(!persistent_prompt_cache_recurrent_snapshot_is_common_prefix_checkpoint(4));
    assert!(persistent_prompt_cache_recurrent_snapshot_is_common_prefix_checkpoint(7));
}

#[test]
fn should_credit_only_reclaimable_parent_snapshot_bytes() {
    let save_admission = persistent_prompt_cache_save_admission(
        RETENTION_POLICY_MAXIMUM_SIZE_BYTES - 20_000_000,
        KV_BLOCK_BYTES,
        RECURRENT_SNAPSHOT_BYTES,
        RECURRENT_SNAPSHOT_BYTES,
        RETENTION_POLICY_MAXIMUM_SIZE_BYTES,
        RETENTION_POLICY_CHILD_BLOCK_INDEX,
        false,
    );

    assert_eq!(
        save_admission,
        PersistentPromptCacheBlockSaveAdmission::SaveAndEvictOldBlocksToFit,
        "saving a child should account for deleting the parent recurrent snapshot after the child snapshot is safe"
    );
}

#[test]
fn should_skip_child_when_combined_kv_and_snapshot_cannot_fit_without_parent_snapshot_credit() {
    let save_admission = persistent_prompt_cache_save_admission(
        RETENTION_POLICY_MAXIMUM_SIZE_BYTES - 20_000_000,
        KV_BLOCK_BYTES,
        RECURRENT_SNAPSHOT_BYTES,
        0,
        RETENTION_POLICY_MAXIMUM_SIZE_BYTES,
        223,
        false,
    );

    assert_eq!(
        save_admission,
        PersistentPromptCacheBlockSaveAdmission::SkipBecauseCacheIsFull,
        "a later child is worthless if saving it cannot preserve a restorable prefix"
    );
}

#[test]
fn should_admit_a_root_block_even_when_the_cache_is_full() {
    let save_admission = persistent_prompt_cache_save_admission(
        RETENTION_POLICY_MAXIMUM_SIZE_BYTES - 20_000_000,
        KV_BLOCK_BYTES,
        RECURRENT_SNAPSHOT_BYTES,
        0,
        RETENTION_POLICY_MAXIMUM_SIZE_BYTES,
        0,
        false,
    );

    assert_eq!(
        save_admission,
        PersistentPromptCacheBlockSaveAdmission::SaveAndEvictOldBlocksToFit,
        "a new root block must be able to displace stale cache files so a useful prefix can be rebuilt"
    );
}

#[test]
fn should_admit_a_child_block_when_split_files_fit_without_eviction() {
    let save_admission = persistent_prompt_cache_save_admission(
        RETENTION_POLICY_MAXIMUM_SIZE_BYTES - 140_000_000,
        KV_BLOCK_BYTES,
        RECURRENT_SNAPSHOT_BYTES,
        RECURRENT_SNAPSHOT_BYTES,
        RETENTION_POLICY_MAXIMUM_SIZE_BYTES,
        RETENTION_POLICY_CHILD_BLOCK_INDEX,
        false,
    );

    assert_eq!(
        save_admission,
        PersistentPromptCacheBlockSaveAdmission::SaveWithoutEviction
    );
}

#[test]
fn should_admit_an_existing_kv_block_when_it_replaces_itself() {
    let save_admission = persistent_prompt_cache_save_admission(
        RETENTION_POLICY_MAXIMUM_SIZE_BYTES - 20_000_000,
        KV_BLOCK_BYTES,
        RECURRENT_SNAPSHOT_BYTES,
        RECURRENT_SNAPSHOT_BYTES,
        RETENTION_POLICY_MAXIMUM_SIZE_BYTES,
        RETENTION_POLICY_CHILD_BLOCK_INDEX,
        true,
    );

    assert_eq!(
        save_admission,
        PersistentPromptCacheBlockSaveAdmission::SaveWithoutEviction,
        "rewriting an already tracked KV block should not be treated as appending a new tail block"
    );
}
