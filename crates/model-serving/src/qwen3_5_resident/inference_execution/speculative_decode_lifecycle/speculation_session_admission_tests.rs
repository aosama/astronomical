//! Unit tests for speculative session admission.

use super::speculation_session_admission::{
    SpeculationSessionRequestProperties, VerificationTokenChoice,
    should_admit_speculation_for_session, verification_token_choice_for_strategy,
};

fn plain_greedy_request() -> SpeculationSessionRequestProperties {
    SpeculationSessionRequestProperties {
        sampling_strategy: Some(VerificationTokenChoice::HighestLogit),
        structured_generation_is_active: false,
        forced_thinking_transition_is_pending: false,
        visual_embeddings_are_active: false,
        repetition_penalty_is_active: false,
    }
}

const PAIRED_TARGET: (usize, usize, u32) = (40, 2048, 248_320);
const PAIRED_DRAFT: (i32, i32, i32) = (40, 248_320, 16_384);
const PAIRED_TAPS: [i32; 8] = [1, 6, 11, 16, 22, 27, 32, 37];

#[test]
fn should_admit_a_plain_greedy_session() {
    assert!(should_admit_speculation_for_session(&plain_greedy_request()));
}

#[test]
fn should_admit_a_sampled_top_k_top_p_session() {
    // Sampled decoding preserves exactly under token-equality verification:
    // every emitted token is the target's own tempered sample, so a
    // temperature-above-zero session may run draft-and-verify.
    let mut sampled = plain_greedy_request();
    sampled.sampling_strategy = Some(VerificationTokenChoice::TopKTopP {
        temperature_thousandths: 700,
        top_k: 40,
        top_p_thousandths: 950,
    });
    assert!(should_admit_speculation_for_session(&sampled));

    // The sampled strategy does not unlock the conflicting properties.
    let mut sampled_with_penalty = sampled.clone();
    sampled_with_penalty.repetition_penalty_is_active = true;
    assert!(!should_admit_speculation_for_session(&sampled_with_penalty));

    let mut sampled_with_structured_generation = sampled;
    sampled_with_structured_generation.structured_generation_is_active = true;
    assert!(!should_admit_speculation_for_session(
        &sampled_with_structured_generation
    ));
}

#[test]
fn should_reject_sessions_the_token_equality_contract_cannot_preserve() {
    let mut sampled = plain_greedy_request();
    sampled.sampling_strategy = None;
    assert!(!should_admit_speculation_for_session(&sampled));

    let mut structured = plain_greedy_request();
    structured.structured_generation_is_active = true;
    assert!(!should_admit_speculation_for_session(&structured));

    let mut forced = plain_greedy_request();
    forced.forced_thinking_transition_is_pending = true;
    assert!(!should_admit_speculation_for_session(&forced));

    let mut visual = plain_greedy_request();
    visual.visual_embeddings_are_active = true;
    assert!(!should_admit_speculation_for_session(&visual));

    let mut penalized = plain_greedy_request();
    penalized.repetition_penalty_is_active = true;
    assert!(!should_admit_speculation_for_session(&penalized));
}

#[test]
fn should_map_request_strategies_onto_verification_token_choices() {
    use crate::Qwen3_5SamplingStrategy;

    // Greedy maps to per-position argmax.
    assert_eq!(
        verification_token_choice_for_strategy(&Qwen3_5SamplingStrategy::HighestLogit),
        Some(VerificationTokenChoice::HighestLogit)
    );

    // The sampled strategy carries its parameters across exactly, so the
    // verifier reproduces the classic per-token choice bit for bit.
    assert_eq!(
        verification_token_choice_for_strategy(&Qwen3_5SamplingStrategy::TopKTopP {
            temperature_thousandths: 700,
            top_k: 40,
            top_p_thousandths: 950,
            seed: None,
        }),
        Some(VerificationTokenChoice::TopKTopP {
            temperature_thousandths: 700,
            top_k: 40,
            top_p_thousandths: 950,
        })
    );
}
