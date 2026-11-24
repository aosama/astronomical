use super::{ReplyExpertRouteRecord, SessionExpertRouteSample};

fn token_change(memory_served: u64, storage_read: u64) -> SessionExpertRouteSample {
    SessionExpertRouteSample {
        memory_served_expert_count: memory_served,
        storage_read_expert_count: storage_read,
        decode_storage_read_bytes: storage_read * 1_000,
        decode_storage_read_wait_nanoseconds: storage_read * 10,
    }
}

#[test]
fn should_keep_each_token_storage_reads_in_order_and_the_routed_width() {
    let mut route_record = ReplyExpertRouteRecord::default();

    route_record.record_token(token_change(362, 22));
    route_record.record_token(token_change(384, 0));
    route_record.record_token(token_change(354, 30));

    assert_eq!(route_record.routed_expert_count_per_token(), 384);
    assert_eq!(
        route_record.into_storage_read_expert_counts_by_token(),
        vec![22, 0, 30]
    );
}

#[test]
fn should_total_the_reply_hits_misses_bytes_and_wait() {
    let mut route_record = ReplyExpertRouteRecord::default();

    route_record.record_token(token_change(362, 22));
    route_record.record_token(token_change(354, 30));

    assert_eq!(
        route_record.reply_totals(),
        SessionExpertRouteSample {
            memory_served_expert_count: 716,
            storage_read_expert_count: 52,
            decode_storage_read_bytes: 52_000,
            decode_storage_read_wait_nanoseconds: 520,
        }
    );
}

#[test]
fn should_report_no_tokens_and_no_width_before_the_first_token() {
    let route_record = ReplyExpertRouteRecord::default();

    assert_eq!(route_record.routed_expert_count_per_token(), 0);
    assert_eq!(
        route_record.reply_totals(),
        SessionExpertRouteSample::default()
    );
    assert!(
        route_record
            .into_storage_read_expert_counts_by_token()
            .is_empty()
    );
}

#[test]
fn should_saturate_counts_too_large_for_the_record() {
    let mut route_record = ReplyExpertRouteRecord::default();

    route_record.record_token(SessionExpertRouteSample {
        memory_served_expert_count: u64::MAX,
        storage_read_expert_count: u64::MAX,
        decode_storage_read_bytes: u64::MAX,
        decode_storage_read_wait_nanoseconds: u64::MAX,
    });
    route_record.record_token(token_change(1, 1));

    assert_eq!(route_record.routed_expert_count_per_token(), u16::MAX);
    assert_eq!(
        route_record.reply_totals().storage_read_expert_count,
        u64::MAX
    );
    assert_eq!(
        route_record.into_storage_read_expert_counts_by_token(),
        vec![u16::MAX, 1]
    );
}

#[test]
fn should_record_a_token_of_a_model_that_streams_no_experts_as_zero() {
    let mut route_record = ReplyExpertRouteRecord::default();

    route_record.record_token(SessionExpertRouteSample::default());

    assert!(!route_record.is_empty());
    assert_eq!(
        route_record.into_storage_read_expert_counts_by_token(),
        vec![0]
    );
}

#[test]
fn should_read_a_token_change_as_the_difference_between_two_samples() {
    let before = token_change(100, 10);
    let after = token_change(140, 15);

    assert_eq!(after.change_since(&before), token_change(40, 5));
}

#[test]
fn should_read_a_counter_that_went_backwards_as_no_change() {
    let before = token_change(140, 15);
    let after = token_change(100, 10);

    assert_eq!(
        after.change_since(&before),
        SessionExpertRouteSample::default()
    );
}
