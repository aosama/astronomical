use super::RequestDraftingStatistics;

#[test]
fn should_count_rounds_guesses_and_kept_words() {
    let mut reply_drafting = RequestDraftingStatistics::default();
    assert!(!reply_drafting.has_drafted());
    reply_drafting.record_round(3, 2);
    reply_drafting.record_round(3, 0);
    assert_eq!(reply_drafting.round_count, 2);
    assert_eq!(reply_drafting.drafted_token_count, 6);
    assert_eq!(reply_drafting.accepted_token_count, 2);
    assert_eq!(reply_drafting.rejected_token_count(), 4);
    assert!(reply_drafting.has_drafted());
}

#[test]
fn should_add_replies_into_session_totals_without_overflowing() {
    let mut session_drafting = RequestDraftingStatistics {
        round_count: u32::MAX,
        drafted_token_count: 10,
        accepted_token_count: 4,
    };
    session_drafting.accumulate(&RequestDraftingStatistics {
        round_count: 1,
        drafted_token_count: 3,
        accepted_token_count: 3,
    });
    assert_eq!(session_drafting.round_count, u32::MAX);
    assert_eq!(session_drafting.drafted_token_count, 13);
    assert_eq!(session_drafting.accepted_token_count, 7);
}

#[test]
fn should_count_unsent_kept_guesses_as_thrown_away() {
    let mut reply_drafting = RequestDraftingStatistics::default();
    reply_drafting.record_round(3, 3);
    reply_drafting.withdraw_unsent_accepted_tokens(2);
    assert_eq!(reply_drafting.accepted_token_count, 1);
    assert_eq!(reply_drafting.rejected_token_count(), 2);
}

#[test]
fn should_never_withdraw_more_than_was_kept() {
    let mut reply_drafting = RequestDraftingStatistics::default();
    reply_drafting.record_round(3, 1);
    reply_drafting.withdraw_unsent_accepted_tokens(5);
    assert_eq!(reply_drafting.accepted_token_count, 0);
    assert_eq!(reply_drafting.rejected_token_count(), 3);
}
