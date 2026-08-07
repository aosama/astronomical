use std::collections::HashMap;
use std::time::Duration;

use std::sync::Arc;

use astronomical_model_serving::{
    PerformanceAttribution, PerformanceOperation, PersistentPromptCacheBlockKey,
    PersistentPromptCacheDiskStore, PersistentPromptCacheDiskStoreConfig,
    PersistentPromptCacheWriteQueue, PersistentPromptCacheWriteQueueOutcome,
    PersistentPromptCacheWriteRateLimiter, persistent_prompt_cache_write_queue_can_accept,
};
use astronomical_runtime_integration::MlxDtype;

use crate::common::qwen3_5_moe::persistent_prompt_cache_model_contract;

use super::persistent_prompt_cache_disk_store_support::{
    block_tokens_for_seed, open_persistent_prompt_cache_disk_store,
    persistent_prompt_cache_block_key_for_seed, runtime_with_shared_limits,
    synthetic_kv_block_tensors, synthetic_recurrent_snapshot_tensors,
};

const LARGE_CACHE_LIMIT_BYTES: u64 = 10 * 1024 * 1024 * 1024;
const OVERSIZED_QUEUE_CAPTURE_ELEMENT_COUNT: i32 = 70_000_000;

#[tokio::test]
async fn should_reject_an_oversized_queue_capture_before_mlx_serialization() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let persistent_prompt_cache = Arc::new(
        open_persistent_prompt_cache_disk_store(
            &persistent_prompt_cache_directory,
            LARGE_CACHE_LIMIT_BYTES,
        )
        .expect("the persistent prompt cache should open an empty directory"),
    );
    let write_queue = PersistentPromptCacheWriteQueue::new(persistent_prompt_cache, None)
        .expect("the test should start the bounded writer queue");
    let oversized_kv_block_tensors = HashMap::from([(
        "layer.0.full_attention.key".to_owned(),
        runtime
            .zeros(&[OVERSIZED_QUEUE_CAPTURE_ELEMENT_COUNT], MlxDtype::BFloat16)
            .expect("the test should construct a lazy oversized KV tensor"),
    )]);
    let oversized_recurrent_snapshot_tensors = HashMap::from([(
        "layer.0.linear_attention.recurrent_state".to_owned(),
        runtime
            .zeros(&[OVERSIZED_QUEUE_CAPTURE_ELEMENT_COUNT], MlxDtype::BFloat16)
            .expect("the test should construct a lazy oversized recurrent tensor"),
    )]);
    let mut performance_attribution = PerformanceAttribution::enabled();

    let write_queue_outcome = write_queue
        .serialize_and_enqueue(
            &runtime,
            &persistent_prompt_cache_block_key_for_seed(0),
            None,
            &oversized_kv_block_tensors,
            &oversized_recurrent_snapshot_tensors,
            &mut performance_attribution,
        )
        .expect("the oversized capture should be rejected without serialization");

    assert_eq!(
        write_queue_outcome,
        PersistentPromptCacheWriteQueueOutcome::DroppedBecauseQueueIsFull
    );
    assert_eq!(
        performance_attribution
            .operation_measurement(PerformanceOperation::PersistentPromptCacheKvBlockSerialization),
        None
    );
    assert_eq!(
        performance_attribution.operation_measurement(
            PerformanceOperation::PersistentPromptCacheRecurrentSnapshotSerialization
        ),
        None
    );
}

#[tokio::test]
async fn should_publish_four_production_sized_boundaries_with_bounded_backpressure() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let persistent_prompt_cache = Arc::new(
        open_persistent_prompt_cache_disk_store(
            &persistent_prompt_cache_directory,
            LARGE_CACHE_LIMIT_BYTES,
        )
        .expect("the persistent prompt cache should open an empty directory"),
    );
    let write_queue =
        PersistentPromptCacheWriteQueue::new(Arc::clone(&persistent_prompt_cache), Some(100))
            .expect("the test should start the bounded writer queue");
    let kv_block_tensors = synthetic_kv_block_tensors(&runtime);
    let recurrent_snapshot_tensors = synthetic_recurrent_snapshot_tensors(&runtime);
    let mut parent_persistent_prompt_cache_block_key: Option<PersistentPromptCacheBlockKey> = None;
    let mut performance_attribution = PerformanceAttribution::disabled();

    for boundary_index in 0_u32..4 {
        let persistent_prompt_cache_block_key =
            match parent_persistent_prompt_cache_block_key.as_ref() {
                None => persistent_prompt_cache_block_key_for_seed(0),
                Some(parent_persistent_prompt_cache_block_key) => {
                    parent_persistent_prompt_cache_block_key
                        .for_child_block(&block_tokens_for_seed(boundary_index * 10_000))
                        .expect("the test should construct the next child block identity")
                }
            };
        let write_queue_outcome = write_queue
            .serialize_and_enqueue(
                &runtime,
                &persistent_prompt_cache_block_key,
                parent_persistent_prompt_cache_block_key.as_ref(),
                &kv_block_tensors,
                &recurrent_snapshot_tensors,
                &mut performance_attribution,
            )
            .expect("each production-sized boundary should serialize and enqueue");

        assert_eq!(
            write_queue_outcome,
            PersistentPromptCacheWriteQueueOutcome::Queued,
            "boundary {boundary_index} should wait for bounded writer capacity instead of being dropped"
        );
        parent_persistent_prompt_cache_block_key = Some(persistent_prompt_cache_block_key);
    }

    drop(write_queue);
    assert_eq!(persistent_prompt_cache.sequence_state_block_count(), 4);
}

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

#[test]
fn should_preflight_all_projected_boundary_captures_before_model_execution() {
    let prompt_cache_root =
        tempfile::tempdir().expect("the test should create a prompt-cache root");
    let disk_store = PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            prompt_cache_root.path().join("model/revision"),
            prompt_cache_root.path().to_path_buf(),
            2_000_000,
        ),
        persistent_prompt_cache_model_contract(),
    )
    .expect("the test should open the prompt-cache store");
    let write_queue = PersistentPromptCacheWriteQueue::new(Arc::new(disk_store), None)
        .expect("the test should start the prompt-cache writer");

    assert!(write_queue.can_accept_projected_captures(800_000, 2));
    assert!(!write_queue.can_accept_projected_captures(1_000_000, 2));
    assert!(!write_queue.can_accept_projected_captures(usize::MAX, 1));
}

#[test]
fn should_bound_sequential_capture_memory_by_one_block_and_disk_quota_by_all_blocks() {
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

    assert!(
        write_queue.can_accept_projected_captures(70_000_000, 4),
        "four sequential captures must not be treated as four simultaneously pending payloads"
    );
    assert!(
        !write_queue.can_accept_projected_captures(256_000_000, 1),
        "one capture plus serialization overhead must remain below the pending byte bound"
    );
}

#[test]
fn should_reject_projected_captures_after_writer_disconnection_or_shutdown() {
    let disconnected_prompt_cache_root =
        tempfile::tempdir().expect("the test should create a disconnected cache root");
    let disconnected_disk_store = PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            disconnected_prompt_cache_root.path().join("model/revision"),
            disconnected_prompt_cache_root.path().to_path_buf(),
            1_000_000_000,
        ),
        persistent_prompt_cache_model_contract(),
    )
    .expect("the disconnected test store should open");
    let mut disconnected_write_queue =
        PersistentPromptCacheWriteQueue::new(Arc::new(disconnected_disk_store), None)
            .expect("the disconnected test writer should start");
    disconnected_write_queue.disconnect_writer_for_tests();
    assert!(!disconnected_write_queue.can_accept_projected_captures(1, 1));

    let stopped_prompt_cache_root =
        tempfile::tempdir().expect("the test should create a stopped cache root");
    let stopped_disk_store = PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            stopped_prompt_cache_root.path().join("model/revision"),
            stopped_prompt_cache_root.path().to_path_buf(),
            1_000_000_000,
        ),
        persistent_prompt_cache_model_contract(),
    )
    .expect("the stopped test store should open");
    let stopped_write_queue =
        PersistentPromptCacheWriteQueue::new(Arc::new(stopped_disk_store), None)
            .expect("the stopped test writer should start");
    stopped_write_queue.request_writer_shutdown_for_tests();
    assert!(!stopped_write_queue.can_accept_projected_captures(1, 1));
}
