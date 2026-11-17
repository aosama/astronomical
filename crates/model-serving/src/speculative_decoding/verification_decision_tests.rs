use super::verification_decision::{
    DraftedSlot, TargetDistribution, VerificationDecision, decide_by_rejection_sampling,
    decide_by_token_equality,
};

/// The first `probabilities.len()` token ids, the shape tests build rows in.
const DENSE_TOKEN_IDS: [u32; 4] = [0, 1, 2, 3];

fn dense_row(probabilities: &[f32]) -> TargetDistribution<'_> {
    TargetDistribution {
        token_ids: &DENSE_TOKEN_IDS[..probabilities.len()],
        probabilities,
    }
}

/// A seeded uniform stream for the statistical test (splitmix64).
struct UniformStream {
    state: u64,
}

impl UniformStream {
    fn next_uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut mixed = self.state;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        mixed ^= mixed >> 31;
        (mixed >> 11) as f64 / (1_u64 << 53) as f64
    }
}

#[test]
fn should_keep_every_draft_that_matches_the_greedy_target_and_add_its_next_token() {
    let decision = decide_by_token_equality(&[5, 6, 7], &[5, 6, 7, 8]);
    assert_eq!(
        decision,
        VerificationDecision {
            kept_draft_count: 3,
            next_token_id: 8
        }
    );
}

#[test]
fn should_stop_at_the_first_greedy_mismatch_and_correct_it() {
    let decision = decide_by_token_equality(&[5, 9, 7], &[5, 6, 7, 8]);
    assert_eq!(
        decision,
        VerificationDecision {
            kept_draft_count: 1,
            next_token_id: 6
        }
    );
}

#[test]
fn should_refuse_a_draft_the_target_gives_no_probability_and_resample_from_the_rest() {
    let target_rows = [dense_row(&[0.0, 1.0, 0.0]), dense_row(&[1.0, 0.0, 0.0])];
    let drafted_slots = [DraftedSlot {
        token_id: 2,
        candidate_ids: &[2],
        candidate_probabilities: &[1.0],
    }];
    let decision = decide_by_rejection_sampling(&drafted_slots, &target_rows, &[0.0, 0.5]);
    assert_eq!(
        decision,
        VerificationDecision {
            kept_draft_count: 0,
            next_token_id: 1
        }
    );
}

#[test]
fn should_always_keep_a_draft_the_target_is_certain_of() {
    let target_rows = [dense_row(&[0.0, 1.0, 0.0]), dense_row(&[0.0, 0.0, 1.0])];
    let drafted_slots = [DraftedSlot {
        token_id: 1,
        candidate_ids: &[1, 2],
        candidate_probabilities: &[0.5, 0.5],
    }];
    let decision = decide_by_rejection_sampling(&drafted_slots, &target_rows, &[0.999_999, 0.3]);
    assert_eq!(
        decision,
        VerificationDecision {
            kept_draft_count: 1,
            next_token_id: 2
        }
    );
}

#[test]
fn should_handle_an_empty_draft_by_sampling_the_target_row() {
    let target_rows = [dense_row(&[0.25, 0.25, 0.5])];
    let decision = decide_by_rejection_sampling(&[], &target_rows, &[0.6]);
    assert_eq!(
        decision,
        VerificationDecision {
            kept_draft_count: 0,
            next_token_id: 2
        }
    );
}

#[test]
fn should_emit_first_tokens_distributed_exactly_as_the_target_over_many_rounds() {
    // The target and the drafter disagree on purpose; whatever the drafter
    // proposes, the first emitted token must follow the target's p.
    let target_distribution: [f32; 4] = [0.1, 0.2, 0.3, 0.4];
    let drafter_candidates: [u32; 3] = [0, 1, 3];
    let drafter_distribution: [f32; 3] = [0.5, 0.3, 0.2];
    let target_rows = [
        dense_row(&target_distribution),
        dense_row(&target_distribution),
    ];
    let mut uniforms = UniformStream { state: 20_261_115 };
    let round_count = 200_000;
    let mut first_token_counts = [0_u32; 4];
    for _ in 0..round_count {
        let draw = uniforms.next_uniform();
        let mut cumulative = 0.0;
        let mut drafted_index = drafter_candidates.len() - 1;
        for (candidate_index, probability) in drafter_distribution.iter().enumerate() {
            cumulative += f64::from(*probability);
            if draw < cumulative {
                drafted_index = candidate_index;
                break;
            }
        }
        let drafted_slots = [DraftedSlot {
            token_id: drafter_candidates[drafted_index],
            candidate_ids: &drafter_candidates,
            candidate_probabilities: &drafter_distribution,
        }];
        let decision = decide_by_rejection_sampling(
            &drafted_slots,
            &target_rows,
            &[uniforms.next_uniform(), uniforms.next_uniform()],
        );
        let first_token = if decision.kept_draft_count == 1 {
            drafted_slots[0].token_id
        } else {
            decision.next_token_id
        };
        first_token_counts[first_token as usize] += 1;
    }
    for (token_index, expected_probability) in target_distribution.iter().enumerate() {
        let observed_probability =
            f64::from(first_token_counts[token_index]) / f64::from(round_count);
        assert!(
            (observed_probability - f64::from(*expected_probability)).abs() < 0.005,
            "token {token_index}: observed {observed_probability}, target {expected_probability}"
        );
    }
}

#[test]
fn should_resample_from_the_target_tokens_the_drafter_left_unused_when_its_draft_is_refused() {
    // The target keeps three scattered tokens; the drafter proposes one the
    // target never samples, so the correction must come from the target's own
    // ids, weighted by what the drafter did not already cover.
    let target_token_ids = [9_000_u32, 42, 248_000];
    let target_probabilities = [0.5_f32, 0.3, 0.2];
    let target_rows = [
        TargetDistribution {
            token_ids: &target_token_ids,
            probabilities: &target_probabilities,
        },
        TargetDistribution {
            token_ids: &target_token_ids,
            probabilities: &target_probabilities,
        },
    ];
    let drafted_slots = [DraftedSlot {
        token_id: 7,
        candidate_ids: &[7, 9_000],
        candidate_probabilities: &[0.6, 0.4],
    }];
    // Leftover weights: 9000 -> 0.5 - 0.4 = 0.1, 42 -> 0.3, 248000 -> 0.2 (total 0.6);
    // a draw of 0.5 lands at 0.3 of 0.6, inside token 42's span (0.1 to 0.4).
    let decision = decide_by_rejection_sampling(&drafted_slots, &target_rows, &[0.0, 0.5]);
    assert_eq!(
        decision,
        VerificationDecision {
            kept_draft_count: 0,
            next_token_id: 42
        }
    );
}
