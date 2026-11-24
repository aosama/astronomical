use ipc_protocol::RequestDraftingStatistics;

use super::GenerationFinalization;

#[test]
fn should_carry_the_drafting_of_a_reply_that_drafted() {
    let reply_drafting = RequestDraftingStatistics {
        round_count: 1,
        drafted_token_count: 3,
        accepted_token_count: 2,
    };
    let generation_finalization =
        GenerationFinalization::default().with_drafting_statistics(reply_drafting);
    assert_eq!(
        generation_finalization.drafting_statistics(),
        Some(reply_drafting)
    );
}

#[test]
fn should_report_no_drafting_for_a_reply_that_ran_no_round() {
    let generation_finalization = GenerationFinalization::default()
        .with_drafting_statistics(RequestDraftingStatistics::default());
    assert_eq!(generation_finalization.drafting_statistics(), None);
}

#[test]
fn should_report_no_drafting_by_default() {
    assert_eq!(
        GenerationFinalization::default().drafting_statistics(),
        None
    );
}
