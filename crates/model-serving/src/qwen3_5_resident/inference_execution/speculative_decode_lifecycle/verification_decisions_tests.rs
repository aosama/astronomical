//! Unit tests for the pure greedy verification decisions.

use super::verification_decisions::{
    accepted_context_growth, decide_greedy_verification, emitted_tokens_for_round,
};

#[test]
fn should_accept_the_full_block_when_the_target_agrees_with_every_proposal() {
    let proposals = [7_u32, 8, 9, 10];
    let target_preferred_token_ids = [7_u32, 8, 9, 10];

    let decision = decide_greedy_verification(&proposals, &target_preferred_token_ids);

    assert_eq!(decision.accepted_proposal_count, 4);
    assert!(decision.accepted_every_proposal);
    assert_eq!(accepted_context_growth(&decision), 5);
    assert_eq!(
        emitted_tokens_for_round(6, &proposals, &target_preferred_token_ids, &decision),
        vec![6, 7, 8, 9, 10]
    );
}

#[test]
fn should_stop_at_the_first_disagreement() {
    let proposals = [7_u32, 8, 111, 10];
    let target_preferred_token_ids = [7_u32, 8, 9, 10];

    let decision = decide_greedy_verification(&proposals, &target_preferred_token_ids);

    assert_eq!(decision.accepted_proposal_count, 2);
    assert!(!decision.accepted_every_proposal);
    assert_eq!(accepted_context_growth(&decision), 3);
}

#[test]
fn should_emit_the_target_correction_token_on_rejection() {
    let proposals = [7_u32, 111];
    let target_preferred_token_ids = [7_u32, 9];

    let decision = decide_greedy_verification(&proposals, &target_preferred_token_ids);
    let emitted_tokens =
        emitted_tokens_for_round(6, &proposals, &target_preferred_token_ids, &decision);

    // Bonus, the one accepted proposal, then the target's own preference at
    // the rejected slot — which the caller also uses as the next bonus.
    assert_eq!(emitted_tokens, vec![6, 7, 9]);
}

#[test]
fn should_reject_the_whole_block_when_the_first_proposal_disagrees() {
    let proposals = [111_u32, 8];
    let target_preferred_token_ids = [7_u32, 8];

    let decision = decide_greedy_verification(&proposals, &target_preferred_token_ids);

    assert_eq!(decision.accepted_proposal_count, 0);
    assert!(!decision.accepted_every_proposal);
    assert_eq!(accepted_context_growth(&decision), 1);
    // Only the bonus and the correction token emit.
    assert_eq!(
        emitted_tokens_for_round(6, &proposals, &target_preferred_token_ids, &decision),
        vec![6, 7]
    );
}

#[test]
fn should_treat_a_short_preference_row_as_safe_partial_evidence() {
    // A caller may pass fewer preference rows than proposals (for example an
    // early end-of-sequence); only the comparable prefix may be accepted and
    // the round must not claim full acceptance.
    let proposals = [7_u32, 8, 9];
    let target_preferred_token_ids = [7_u32, 8];

    let decision = decide_greedy_verification(&proposals, &target_preferred_token_ids);

    assert_eq!(decision.accepted_proposal_count, 2);
    assert!(!decision.accepted_every_proposal);
}

#[test]
fn should_accept_an_empty_draft_unconditionally() {
    let decision = decide_greedy_verification(&[], &[7_u32]);

    assert_eq!(decision.accepted_proposal_count, 0);
    assert!(decision.accepted_every_proposal);
    assert_eq!(accepted_context_growth(&decision), 1);
}
