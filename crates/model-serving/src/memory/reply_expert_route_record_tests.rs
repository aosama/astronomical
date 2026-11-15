use super::ReplyExpertRouteRecord;

#[test]
fn should_keep_each_token_storage_reads_in_order_and_the_routed_width() {
    let mut route_record = ReplyExpertRouteRecord::default();

    route_record.record_token(362, 22);
    route_record.record_token(384, 0);
    route_record.record_token(354, 30);

    assert_eq!(route_record.routed_expert_count_per_token(), 384);
    assert_eq!(
        route_record.into_storage_read_expert_counts_by_token(),
        vec![22, 0, 30]
    );
}

#[test]
fn should_report_no_tokens_and_no_width_before_the_first_token() {
    let route_record = ReplyExpertRouteRecord::default();

    assert_eq!(route_record.routed_expert_count_per_token(), 0);
    assert!(
        route_record
            .into_storage_read_expert_counts_by_token()
            .is_empty()
    );
}

#[test]
fn should_saturate_counts_too_large_for_the_record() {
    let mut route_record = ReplyExpertRouteRecord::default();

    route_record.record_token(u64::MAX, u64::MAX);

    assert_eq!(route_record.routed_expert_count_per_token(), u16::MAX);
    assert_eq!(
        route_record.into_storage_read_expert_counts_by_token(),
        vec![u16::MAX]
    );
}

#[test]
fn should_record_a_token_of_a_model_that_streams_no_experts_as_zero() {
    let mut route_record = ReplyExpertRouteRecord::default();

    route_record.record_token(0, 0);

    assert_eq!(
        route_record.into_storage_read_expert_counts_by_token(),
        vec![0]
    );
}
