use super::MlxBufferCacheReplyRecord;

const MEBIBYTE: u64 = 1 << 20;

#[test]
fn should_report_the_smallest_largest_and_mean_mlx_buffer_cache_over_a_reply() {
    let mut reply_record = MlxBufferCacheReplyRecord::default();

    for mlx_buffer_cache_bytes in [300 * MEBIBYTE, 100 * MEBIBYTE, 200 * MEBIBYTE] {
        reply_record.record_token(mlx_buffer_cache_bytes);
    }

    assert_eq!(reply_record.sampled_token_count(), 3);
    assert_eq!(reply_record.empty_token_count(), 0);
    assert_eq!(reply_record.smallest_bytes(), 100 * MEBIBYTE);
    assert_eq!(reply_record.largest_bytes(), 300 * MEBIBYTE);
    assert_eq!(reply_record.mean_bytes(), 200 * MEBIBYTE);
}

#[test]
fn should_count_every_token_after_which_the_mlx_buffer_cache_was_empty() {
    let mut reply_record = MlxBufferCacheReplyRecord::default();

    for mlx_buffer_cache_bytes in [0, 64 * MEBIBYTE, 0, 0] {
        reply_record.record_token(mlx_buffer_cache_bytes);
    }

    assert_eq!(reply_record.empty_token_count(), 3);
    assert_eq!(reply_record.smallest_bytes(), 0);
    assert_eq!(reply_record.mean_bytes(), 16 * MEBIBYTE);
}

#[test]
fn should_report_zeros_for_a_reply_that_sampled_no_token() {
    let reply_record = MlxBufferCacheReplyRecord::default();

    assert_eq!(reply_record.sampled_token_count(), 0);
    assert_eq!(reply_record.empty_token_count(), 0);
    assert_eq!(reply_record.smallest_bytes(), 0);
    assert_eq!(reply_record.largest_bytes(), 0);
    assert_eq!(reply_record.mean_bytes(), 0);
}

#[test]
fn should_keep_the_total_from_overflowing_on_impossible_sizes() {
    let mut reply_record = MlxBufferCacheReplyRecord::default();

    reply_record.record_token(u64::MAX);
    reply_record.record_token(u64::MAX);

    assert_eq!(reply_record.largest_bytes(), u64::MAX);
    assert_eq!(reply_record.mean_bytes(), u64::MAX / 2);
}

#[test]
fn should_carry_the_readings_and_each_token_storage_reads_into_the_reply_memory_record() {
    let mut reply_record = MlxBufferCacheReplyRecord::default();
    for mlx_buffer_cache_bytes in [0, 200 * MEBIBYTE, 400 * MEBIBYTE] {
        reply_record.record_token(mlx_buffer_cache_bytes);
    }

    let mut route_record =
        crate::memory::reply_expert_route_record::ReplyExpertRouteRecord::default();
    for (memory_served_route_count, storage_read_route_count) in [(377, 7), (384, 0), (382, 2)] {
        route_record.record_token(
            crate::memory::reply_expert_route_record::SessionExpertRouteSample {
                memory_served_expert_count: memory_served_route_count,
                storage_read_expert_count: storage_read_route_count,
                decode_storage_read_bytes: storage_read_route_count * MEBIBYTE,
                decode_storage_read_wait_nanoseconds: storage_read_route_count * 1_000,
            },
        );
    }

    let reply_memory_record = reply_record.reply_memory_record(route_record);

    assert_eq!(
        reply_memory_record.storage_read_expert_counts_by_token,
        vec![7, 0, 2]
    );
    assert_eq!(reply_memory_record.routed_expert_count_per_token, 384);
    assert_eq!(reply_memory_record.expert_hit_count, 1_143);
    assert_eq!(reply_memory_record.expert_miss_count, 9);
    assert_eq!(reply_memory_record.storage_read_bytes, 9 * MEBIBYTE);
    assert_eq!(reply_memory_record.storage_read_wait_nanoseconds, 9_000);
    assert_eq!(reply_memory_record.mlx_buffer_cache_sampled_token_count, 3);
    assert_eq!(reply_memory_record.mlx_buffer_cache_empty_token_count, 1);
    assert_eq!(reply_memory_record.smallest_mlx_buffer_cache_bytes, 0);
    assert_eq!(
        reply_memory_record.mean_mlx_buffer_cache_bytes,
        200 * MEBIBYTE
    );
    assert_eq!(
        reply_memory_record.largest_mlx_buffer_cache_bytes,
        400 * MEBIBYTE
    );
}

#[test]
fn should_build_an_empty_reply_memory_record_for_a_reply_without_tokens() {
    let reply_memory_record =
        MlxBufferCacheReplyRecord::default().reply_memory_record(Default::default());

    assert_eq!(
        reply_memory_record,
        ipc_protocol::ReplyMemoryRecord::default()
    );
}
