use std::collections::HashMap;

use mlx_bridge::MlxArray;
use model_serving::{AstroSsdPromptCacheBlockKey, AstroSsdPromptCacheDiskStore};
use runtime_integration::MlxRuntime;

use super::astro_ssd_prompt_cache_disk_store_support::*;

const LARGE_ASTRO_SSD_PROMPT_CACHE_LIMIT_BYTES: u64 = 10 * 1024 * 1024 * 1024;

#[tokio::test]
async fn should_protect_the_active_parent_chain_and_evict_unrelated_blocks_under_quota_pressure() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let measurement_astro_ssd_prompt_cache_directory = tempfile::tempdir()
        .expect("the test should create a measurement astro-ssd-prompt-cache directory");
    let measurement_astro_ssd_prompt_cache = open_astro_ssd_prompt_cache_disk_store(
        &measurement_astro_ssd_prompt_cache_directory,
        LARGE_ASTRO_SSD_PROMPT_CACHE_LIMIT_BYTES,
    )
    .expect("the measurement astro SSD prompt cache should open");
    let parent_astro_ssd_prompt_cache_block_key = astro_ssd_prompt_cache_block_key_for_seed(0);
    let child_astro_ssd_prompt_cache_block_key = parent_astro_ssd_prompt_cache_block_key
        .for_child_block(&block_tokens_for_seed(10_000))
        .expect("the test should hash the child block");
    let unrelated_astro_ssd_prompt_cache_block_key =
        astro_ssd_prompt_cache_block_key_for_seed(99_000);
    let kv_block_tensors = synthetic_kv_block_tensors(&runtime);
    let recurrent_snapshot_tensors = synthetic_recurrent_snapshot_tensors(&runtime);
    let measured_single_block_size_bytes = measure_single_block_size_bytes(
        &runtime,
        measurement_astro_ssd_prompt_cache,
        &parent_astro_ssd_prompt_cache_block_key,
        (&kv_block_tensors, &recurrent_snapshot_tensors),
    );
    let two_block_quota_bytes = measured_single_block_size_bytes
        .checked_mul(2)
        .and_then(|two_blocks| two_blocks.checked_add(1024))
        .expect("the test block size should fit the quota arithmetic");
    let astro_ssd_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create an astro SSD prompt cache directory");
    let astro_ssd_prompt_cache = open_astro_ssd_prompt_cache_disk_store(
        &astro_ssd_prompt_cache_directory,
        two_block_quota_bytes,
    )
    .expect("the astro SSD prompt cache should open with a two-block quota");

    astro_ssd_prompt_cache
        .publish_block(
            &runtime,
            &parent_astro_ssd_prompt_cache_block_key,
            None,
            &kv_block_tensors,
            &recurrent_snapshot_tensors,
        )
        .expect("the parent block should save");
    astro_ssd_prompt_cache
        .publish_block(
            &runtime,
            &unrelated_astro_ssd_prompt_cache_block_key,
            None,
            &kv_block_tensors,
            &recurrent_snapshot_tensors,
        )
        .expect("the unrelated block should save while the two-block quota has room");

    astro_ssd_prompt_cache
        .publish_block(
            &runtime,
            &child_astro_ssd_prompt_cache_block_key,
            Some(&parent_astro_ssd_prompt_cache_block_key),
            &kv_block_tensors,
            &recurrent_snapshot_tensors,
        )
        .expect("the child publication should evict the unrelated block, not the active chain");

    assert_quota_pressure_evicted_only_the_unrelated_block(
        &astro_ssd_prompt_cache,
        [
            &parent_astro_ssd_prompt_cache_block_key,
            &child_astro_ssd_prompt_cache_block_key,
            &unrelated_astro_ssd_prompt_cache_block_key,
        ],
        two_block_quota_bytes,
    );
}

fn measure_single_block_size_bytes(
    runtime: &MlxRuntime,
    measurement_astro_ssd_prompt_cache: AstroSsdPromptCacheDiskStore,
    block_key: &AstroSsdPromptCacheBlockKey,
    block_tensors: (&HashMap<String, MlxArray>, &HashMap<String, MlxArray>),
) -> u64 {
    let (kv_block_tensors, recurrent_snapshot_tensors) = block_tensors;
    measurement_astro_ssd_prompt_cache
        .publish_block(
            runtime,
            block_key,
            None,
            kv_block_tensors,
            recurrent_snapshot_tensors,
        )
        .expect("the measurement astro SSD prompt cache should save one block");
    measurement_astro_ssd_prompt_cache.total_size_bytes()
}

fn assert_quota_pressure_evicted_only_the_unrelated_block(
    astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
    parent_child_and_unrelated_block_keys: [&AstroSsdPromptCacheBlockKey; 3],
    two_block_quota_bytes: u64,
) {
    let [
        parent_astro_ssd_prompt_cache_block_key,
        child_astro_ssd_prompt_cache_block_key,
        unrelated_astro_ssd_prompt_cache_block_key,
    ] = parent_child_and_unrelated_block_keys;
    assert!(
        astro_ssd_prompt_cache.has_kv_block(&parent_astro_ssd_prompt_cache_block_key.block_hash()),
        "the protected parent sequence state must remain restorable"
    );
    assert!(
        astro_ssd_prompt_cache.has_kv_block(&child_astro_ssd_prompt_cache_block_key.block_hash()),
        "the newly published child sequence state must remain restorable"
    );
    assert!(
        !astro_ssd_prompt_cache
            .has_kv_block(&unrelated_astro_ssd_prompt_cache_block_key.block_hash()),
        "quota pressure should evict the unrelated block instead of the active chain"
    );
    assert!(
        astro_ssd_prompt_cache.total_size_bytes() <= two_block_quota_bytes,
        "committed astro SSD prompt cache bytes must satisfy the configured two-block quota"
    );
}
