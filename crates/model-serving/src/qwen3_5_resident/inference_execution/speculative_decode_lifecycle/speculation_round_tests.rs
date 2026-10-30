//! Unit tests for the speculative round's token assembly helpers.

use super::speculation_round::{
    assemble_block_token_ids_for_tests, assemble_replay_tokens_for_tests,
    assemble_verification_tokens_for_tests, validate_draft_context_alignment_for_tests,
};
use super::verification_decisions::{
    TokenEqualityVerificationDecision, decide_token_equality_verification,
};

fn full_acceptance_decision(proposal_count: usize) -> TokenEqualityVerificationDecision {
    decide_token_equality_verification(&vec![7_u32; proposal_count], &vec![7_u32; proposal_count])
}

#[test]
fn should_assemble_verification_tokens_as_bonus_plus_every_proposal() {
    let verification_tokens = assemble_verification_tokens_for_tests(6, &[7, 8, 9]);

    assert_eq!(verification_tokens, vec![6, 7, 8, 9]);
}

#[test]
fn should_assemble_block_tokens_as_bonus_plus_mask_slots() {
    let block_token_ids = assemble_block_token_ids_for_tests(6, 248_077, 4);

    assert_eq!(block_token_ids, vec![6, 248_077, 248_077, 248_077, 248_077]);
}

#[test]
fn should_replay_only_the_accepted_prefix_plus_the_bonus() {
    let proposals = [7_u32, 8, 111, 10];
    let decision = decide_token_equality_verification(&proposals, &[7, 8, 9, 10]);

    let replay_tokens = assemble_replay_tokens_for_tests(6, &proposals, &decision);

    // The rejected suffix and the correction token stay out of the replay:
    // the replay exists to rebuild decoder state for committed tokens only.
    assert_eq!(replay_tokens, vec![6, 7, 8]);
}

#[test]
fn should_replay_only_the_bonus_when_nothing_was_accepted() {
    let proposals = [111_u32, 8];
    let decision = decide_token_equality_verification(&proposals, &[7, 8]);

    let replay_tokens = assemble_replay_tokens_for_tests(6, &proposals, &decision);

    assert_eq!(replay_tokens, vec![6]);
}

#[test]
fn should_replay_the_whole_block_on_full_acceptance() {
    let proposals = [7_u32, 8, 9];
    let decision = full_acceptance_decision(proposals.len());

    let replay_tokens = assemble_replay_tokens_for_tests(6, &proposals, &decision);

    assert_eq!(replay_tokens, vec![6, 7, 8, 9]);
}

#[test]
fn should_accept_a_draft_state_standing_exactly_at_the_context_boundary() {
    let alignment = validate_draft_context_alignment_for_tests(12, 12);

    assert!(alignment.is_ok());
}

#[test]
fn should_reject_a_draft_state_offset_from_the_context_boundary() {
    let alignment = validate_draft_context_alignment_for_tests(11, 12);

    // The rejection names the invariant, so a silent offset shift is
    // diagnosable from the error alone.
    assert_eq!(
        alignment.unwrap_err().to_string(),
        "fatal engine error: invalid Qwen3.5 model input: the draft key-value context length must match the target context length before a speculative round"
    );
}

#[test]
fn should_accept_an_empty_context_at_the_zero_boundary() {
    let alignment = validate_draft_context_alignment_for_tests(0, 0);

    assert!(alignment.is_ok());
}
