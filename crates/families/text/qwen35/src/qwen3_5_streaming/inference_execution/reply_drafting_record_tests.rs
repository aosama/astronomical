use ipc_protocol::RequestDraftingStatistics;

use super::statistics_without_undelivered_drafts;

fn two_rounds_keeping_six_of_eight() -> RequestDraftingStatistics {
    let mut recorded_statistics = RequestDraftingStatistics::NONE;
    recorded_statistics.record_round(4, 4);
    recorded_statistics.record_round(4, 2);
    recorded_statistics
}

#[test]
fn should_report_every_kept_draft_when_the_reply_delivered_all_it_decided() {
    let reply_statistics =
        statistics_without_undelivered_drafts(two_rounds_keeping_six_of_eight(), 0);

    assert_eq!(reply_statistics.round_count, 2);
    assert_eq!(reply_statistics.drafted_token_count, 8);
    assert_eq!(reply_statistics.accepted_token_count, 6);
}

#[test]
fn should_not_take_back_the_targets_own_choice_left_waiting() {
    // Edge: one waiting token is the target's next choice, not a draft.
    let reply_statistics =
        statistics_without_undelivered_drafts(two_rounds_keeping_six_of_eight(), 1);

    assert_eq!(reply_statistics.accepted_token_count, 6);
}

#[test]
fn should_take_back_kept_drafts_the_reply_ended_before_delivering() {
    let reply_statistics =
        statistics_without_undelivered_drafts(two_rounds_keeping_six_of_eight(), 3);

    assert_eq!(reply_statistics.accepted_token_count, 4);
    assert_eq!(reply_statistics.rejected_token_count(), 4);
}

#[test]
fn should_never_report_fewer_than_no_kept_drafts() {
    // Negative: more waiting tokens than drafts were kept cannot go below zero.
    let reply_statistics =
        statistics_without_undelivered_drafts(two_rounds_keeping_six_of_eight(), 40);

    assert_eq!(reply_statistics.accepted_token_count, 0);
    assert_eq!(reply_statistics.drafted_token_count, 8);
}

#[test]
fn should_report_no_drafting_for_a_reply_that_ran_no_round() {
    let reply_statistics =
        statistics_without_undelivered_drafts(RequestDraftingStatistics::NONE, 0);

    assert!(!reply_statistics.has_drafted());
}
