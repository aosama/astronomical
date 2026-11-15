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
