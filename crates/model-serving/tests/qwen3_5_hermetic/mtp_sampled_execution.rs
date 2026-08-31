use astronomical_model_serving::{
    MtpDraftDepth, MtpVerificationDecisionError, qwen3_5_mtp_sampled_acceptance_probability,
    qwen3_5_mtp_sampled_verification_decision,
};

const END_OF_SEQUENCE_TOKEN_ID: u32 = 404;

fn test_depth(depth_token_count: u8) -> MtpDraftDepth {
    MtpDraftDepth::new(depth_token_count).expect("test depth should be valid")
}

#[test]
fn should_accept_the_full_coin_backed_prefix_at_depths_one_through_three() {
    for depth_token_count in 1_u8..=3 {
        let draft_token_ids = (0..depth_token_count)
            .map(|draft_index| 100 + u32::from(draft_index))
            .collect::<Vec<_>>();
        let accepted_coin_flags = vec![true; usize::from(depth_token_count)];
        let decision = qwen3_5_mtp_sampled_verification_decision(
            test_depth(depth_token_count),
            &draft_token_ids,
            &accepted_coin_flags,
            Some(999),
            &[],
        )
        .expect("all-accept coins describe a valid sampled verification window");
        assert_eq!(decision.accepted_count(), depth_token_count);
        assert_eq!(decision.pending_target_token_id(), Some(999));
        assert!(!decision.was_eos_truncated());
        assert!(!decision.is_operational_fallback());
    }
}

#[test]
fn should_reject_at_the_first_false_coin_and_emit_the_residual_token() {
    let decision = qwen3_5_mtp_sampled_verification_decision(
        test_depth(3),
        &[700_u32, 800, 900],
        &[true, false, true],
        Some(42),
        &[],
    )
    .expect("mixed acceptance coins describe a valid sampled window");
    assert_eq!(decision.accepted_count(), 1);
    assert_eq!(decision.pending_target_token_id(), Some(42));
    assert!(!decision.was_eos_truncated());
}

#[test]
fn should_truncate_at_the_first_eos_inside_the_coin_accepted_prefix() {
    let decision = qwen3_5_mtp_sampled_verification_decision(
        test_depth(3),
        &[700_u32, END_OF_SEQUENCE_TOKEN_ID, 900],
        &[true, true, true],
        Some(42),
        &[END_OF_SEQUENCE_TOKEN_ID],
    )
    .expect("an accepted prefix that reaches EOS truncates exactly like the greedy path");
    assert_eq!(decision.accepted_count(), 2);
    assert!(decision.was_eos_truncated());
    assert_eq!(decision.pending_target_token_id(), None);
}

#[test]
fn should_keep_the_residual_eos_token_when_the_rejection_correction_emits_it() {
    let decision = qwen3_5_mtp_sampled_verification_decision(
        test_depth(2),
        &[700_u32, 800],
        &[true, false],
        Some(END_OF_SEQUENCE_TOKEN_ID),
        &[END_OF_SEQUENCE_TOKEN_ID],
    )
    .expect("an EOS emitted by the residual correction still resolves the next token");
    assert_eq!(decision.accepted_count(), 1);
    assert_eq!(
        decision.pending_target_token_id(),
        Some(END_OF_SEQUENCE_TOKEN_ID)
    );
    assert!(!decision.was_eos_truncated());
}

#[test]
fn should_require_a_post_prefix_token_for_every_untruncated_window() {
    let rejected_without_residual = qwen3_5_mtp_sampled_verification_decision(
        test_depth(2),
        &[700_u32, 800],
        &[true, false],
        None,
        &[],
    );
    let fully_accepted_without_bonus =
        qwen3_5_mtp_sampled_verification_decision(test_depth(1), &[700_u32], &[true], None, &[]);
    assert_eq!(
        rejected_without_residual.unwrap_err(),
        MtpVerificationDecisionError::MissingPostPrefixToken
    );
    assert_eq!(
        fully_accepted_without_bonus.unwrap_err(),
        MtpVerificationDecisionError::MissingPostPrefixToken
    );
}

#[test]
fn should_reject_sampled_vectors_that_do_not_match_the_effective_depth() {
    assert_eq!(
        qwen3_5_mtp_sampled_verification_decision(
            test_depth(2),
            &[700_u32],
            &[true, true],
            Some(42),
            &[],
        )
        .unwrap_err(),
        MtpVerificationDecisionError::VectorLengthMismatch
    );
    assert_eq!(
        qwen3_5_mtp_sampled_verification_decision(
            test_depth(2),
            &[700_u32, 800],
            &[true],
            Some(42),
            &[],
        )
        .unwrap_err(),
        MtpVerificationDecisionError::VectorLengthMismatch
    );
}

#[test]
fn should_cap_sampled_acceptance_probability_at_one_and_reject_zero_draft_mass() {
    assert_eq!(
        qwen3_5_mtp_sampled_acceptance_probability(0.3_f32, 0.6_f32),
        0.5_f32
    );
    assert_eq!(
        qwen3_5_mtp_sampled_acceptance_probability(0.9_f32, 0.4_f32),
        1.0_f32
    );
    assert_eq!(
        qwen3_5_mtp_sampled_acceptance_probability(0.5_f32, 0.5_f32),
        1.0_f32
    );
    assert_eq!(
        qwen3_5_mtp_sampled_acceptance_probability(0.0_f32, 0.6_f32),
        0.0_f32
    );
    // A draft token the draft distribution could not have produced is a
    // caller bug; the verifier treats it as an automatic rejection.
    assert_eq!(
        qwen3_5_mtp_sampled_acceptance_probability(0.4_f32, 0.0_f32),
        0.0_f32
    );
}
