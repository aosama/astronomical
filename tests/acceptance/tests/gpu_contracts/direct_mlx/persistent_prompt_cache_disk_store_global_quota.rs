use super::persistent_prompt_cache_disk_store_support::*;

const LARGE_CACHE_LIMIT_BYTES: u64 = 10 * 1024 * 1024 * 1024;

#[tokio::test]
async fn should_protect_the_active_parent_chain_and_evict_unrelated_blocks_under_quota_pressure() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let measurement_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a measurement prompt-cache directory");
    let measurement_prompt_cache = open_persistent_prompt_cache_disk_store(
        &measurement_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the measurement prompt cache should open");
    let parent_persistent_prompt_cache_block_key = persistent_prompt_cache_block_key_for_seed(0);
    let child_persistent_prompt_cache_block_key = parent_persistent_prompt_cache_block_key
        .for_child_block(&block_tokens_for_seed(10_000))
        .expect("the test should hash the child block");
    let unrelated_persistent_prompt_cache_block_key =
        persistent_prompt_cache_block_key_for_seed(99_000);
    let kv_block_tensors = synthetic_kv_block_tensors(&runtime);
    let recurrent_snapshot_tensors = synthetic_recurrent_snapshot_tensors(&runtime);
    measurement_prompt_cache
        .publish_block(
            &runtime,
            &parent_persistent_prompt_cache_block_key,
            None,
            &kv_block_tensors,
            &recurrent_snapshot_tensors,
        )
        .expect("the measurement prompt cache should save one block");
    let measured_single_block_size_bytes = measurement_prompt_cache.total_size_bytes();
    drop(measurement_prompt_cache);
    let two_block_quota_bytes = measured_single_block_size_bytes
        .checked_mul(2)
        .and_then(|two_blocks| two_blocks.checked_add(1024))
        .expect("the test block size should fit the quota arithmetic");
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        two_block_quota_bytes,
    )
    .expect("the persistent prompt cache should open with a two-block quota");

    persistent_prompt_cache
        .publish_block(
            &runtime,
            &parent_persistent_prompt_cache_block_key,
            None,
            &kv_block_tensors,
            &recurrent_snapshot_tensors,
        )
        .expect("the parent block should save");
    persistent_prompt_cache
        .publish_block(
            &runtime,
            &unrelated_persistent_prompt_cache_block_key,
            None,
            &kv_block_tensors,
            &recurrent_snapshot_tensors,
        )
        .expect("the unrelated block should save while the two-block quota has room");

    persistent_prompt_cache
        .publish_block(
            &runtime,
            &child_persistent_prompt_cache_block_key,
            Some(&parent_persistent_prompt_cache_block_key),
            &kv_block_tensors,
            &recurrent_snapshot_tensors,
        )
        .expect("the child publication should evict the unrelated block, not the active chain");

    assert!(
        persistent_prompt_cache
            .has_kv_block(&parent_persistent_prompt_cache_block_key.block_hash()),
        "the protected parent sequence state must remain restorable"
    );
    assert!(
        persistent_prompt_cache.has_kv_block(&child_persistent_prompt_cache_block_key.block_hash()),
        "the newly published child sequence state must remain restorable"
    );
    assert!(
        !persistent_prompt_cache
            .has_kv_block(&unrelated_persistent_prompt_cache_block_key.block_hash()),
        "quota pressure should evict the unrelated block instead of the active chain"
    );
    assert!(
        persistent_prompt_cache.total_size_bytes() <= two_block_quota_bytes,
        "committed cache bytes must satisfy the configured two-block quota"
    );
}
