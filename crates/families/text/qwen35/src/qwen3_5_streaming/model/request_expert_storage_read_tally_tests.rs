use super::RequestExpertStorageReadTally;

#[test]
fn should_start_with_zero_observed_storage_reads() {
    assert_eq!(
        RequestExpertStorageReadTally::default().expert_storage_read_byte_count(),
        0
    );
}

#[test]
fn should_accumulate_every_streamed_page_within_one_request() {
    let request_expert_storage_read_tally = RequestExpertStorageReadTally::default();
    request_expert_storage_read_tally.record_streamed_page(4_096);
    request_expert_storage_read_tally.record_streamed_page(8_192);
    assert_eq!(
        request_expert_storage_read_tally.expert_storage_read_byte_count(),
        12_288
    );
}

#[test]
fn should_forget_the_previous_request_reads_when_a_new_request_begins() {
    let request_expert_storage_read_tally = RequestExpertStorageReadTally::default();
    request_expert_storage_read_tally.record_streamed_page(4_096);
    request_expert_storage_read_tally.begin_request();
    assert_eq!(
        request_expert_storage_read_tally.expert_storage_read_byte_count(),
        0
    );
}

#[test]
fn should_saturate_instead_of_wrapping_at_the_maximum_byte_count() {
    let request_expert_storage_read_tally = RequestExpertStorageReadTally::default();
    request_expert_storage_read_tally.record_streamed_page(u64::MAX);
    request_expert_storage_read_tally.record_streamed_page(1);
    assert_eq!(
        request_expert_storage_read_tally.expert_storage_read_byte_count(),
        u64::MAX
    );
}

#[test]
fn should_keep_counting_a_zero_byte_page_as_no_observed_read() {
    let request_expert_storage_read_tally = RequestExpertStorageReadTally::default();
    request_expert_storage_read_tally.record_streamed_page(0);
    assert_eq!(
        request_expert_storage_read_tally.expert_storage_read_byte_count(),
        0
    );
}
