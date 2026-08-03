use std::time::Duration;

use std::sync::Arc;

use astronomical_model_serving::{
    PersistentPromptCacheDiskStore, PersistentPromptCacheDiskStoreConfig,
    PersistentPromptCacheWriteQueue, PersistentPromptCacheWriteRateLimiter,
    persistent_prompt_cache_write_queue_can_accept,
};

use crate::common::qwen3_5_moe::persistent_prompt_cache_model_contract;

#[test]
fn should_convert_decimal_megabytes_per_second_to_bytes_per_second() {
    let write_rate_limiter = PersistentPromptCacheWriteRateLimiter::new(Some(3));

    assert_eq!(
        write_rate_limiter.maximum_bytes_per_second(),
        Some(3_000_000)
    );
}

#[test]
fn should_not_artificially_throttle_the_default_prompt_cache_writer() {
    let write_rate_limiter = PersistentPromptCacheWriteRateLimiter::new(None);

    assert_eq!(
        write_rate_limiter.minimum_elapsed_for_bytes(1_000_000_000),
        Duration::ZERO
    );
}

#[test]
fn should_calculate_minimum_elapsed_time_for_written_bytes() {
    let write_rate_limiter = PersistentPromptCacheWriteRateLimiter::new(Some(2));

    assert_eq!(
        write_rate_limiter.minimum_elapsed_for_bytes(5_000_000),
        Duration::from_millis(2_500)
    );
}

#[test]
fn should_treat_a_zero_internal_rate_as_uncapped() {
    let write_rate_limiter = PersistentPromptCacheWriteRateLimiter::new(Some(0));

    assert_eq!(write_rate_limiter.maximum_bytes_per_second(), None);
}

#[test]
fn should_reject_serialized_bytes_beyond_the_bounded_queue_capacity() {
    assert!(persistent_prompt_cache_write_queue_can_accept(
        128_000_000,
        128_000_000
    ));
    assert!(!persistent_prompt_cache_write_queue_can_accept(
        256_000_000,
        1
    ));
    assert!(!persistent_prompt_cache_write_queue_can_accept(u64::MAX, 1));
}

#[test]
fn should_stop_an_idle_writer_when_the_queue_is_dropped() {
    let prompt_cache_root =
        tempfile::tempdir().expect("the test should create a prompt-cache root");
    let disk_store = PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            prompt_cache_root.path().join("model/revision"),
            prompt_cache_root.path().to_path_buf(),
            1_000_000_000,
        ),
        persistent_prompt_cache_model_contract(),
    )
    .expect("the test should open the prompt-cache store");

    let write_queue = PersistentPromptCacheWriteQueue::new(Arc::new(disk_store), None)
        .expect("the test should start the prompt-cache writer");
    drop(write_queue);
}
