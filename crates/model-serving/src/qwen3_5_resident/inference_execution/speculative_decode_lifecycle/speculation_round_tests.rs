//! Unit tests for the speculative round's token assembly helpers.

use super::speculation_round::{
    assemble_block_token_ids_for_tests, assemble_replay_tokens_for_tests,
    assemble_verification_tokens_for_tests,
};
use super::verification_decisions::{GreedyVerificationDecision, decide_greedy_verification};

fn full_acceptance_decision(proposal_count: usize) -> GreedyVerificationDecision {
    decide_greedy_verification(&vec![7_u32; proposal_count], &vec![7_u32; proposal_count])
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
    let decision = decide_greedy_verification(&proposals, &[7, 8, 9, 10]);

    let replay_tokens = assemble_replay_tokens_for_tests(6, &proposals, &decision);

    // The rejected suffix and the correction token stay out of the replay:
    // the replay exists to rebuild decoder state for committed tokens only.
    assert_eq!(replay_tokens, vec![6, 7, 8]);
}

#[test]
fn should_replay_only_the_bonus_when_nothing_was_accepted() {
    let proposals = [111_u32, 8];
    let decision = decide_greedy_verification(&proposals, &[7, 8]);

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
