//! Beside-source tests for the pure context-projection unit of the resident
//! decode-lifecycle feedback injection. The injection step itself needs a
//! loaded engine, so only the pure arithmetic guard is unit-tested here.

use super::{InjectionContextTokenProjection, project_injection_context_token_count};
use crate::InferenceEngineError;

fn projection_tokens(
    next_position_tokens: usize,
    injected_input_token_count: usize,
    remaining_output_tokens: usize,
) -> InjectionContextTokenProjection {
    InjectionContextTokenProjection {
        next_position_tokens,
        injected_input_token_count,
        remaining_output_tokens,
        hard_maximum_position_count: 1_000,
        advertised_maximum_position_count: 800,
    }
}

#[test]
fn should_project_context_tokens_within_the_advertised_context_limit() {
    let projected_context_tokens =
        project_injection_context_token_count(projection_tokens(100, 10, 50))
            .expect("projection within the advertised limit must be admitted");
    assert_eq!(projected_context_tokens, 160);
}

#[test]
fn should_admit_context_projection_exactly_at_the_advertised_context_limit() {
    let projected_context_tokens =
        project_injection_context_token_count(projection_tokens(750, 50, 0))
            .expect("projection exactly at the advertised limit is not an excess");
    assert_eq!(projected_context_tokens, 800);
}

#[test]
fn should_serve_context_projection_between_advertised_and_hard_maximum() {
    // Beyond the advertised limit but within the artifact hard maximum the
    // engine serves anyway (a warn is logged, admission still succeeds).
    let projected_context_tokens =
        project_injection_context_token_count(projection_tokens(890, 10, 0))
            .expect("projection within the hard maximum must still be served");
    assert_eq!(projected_context_tokens, 900);
}

#[test]
fn should_admit_context_projection_exactly_at_the_hard_maximum() {
    let projected_context_tokens =
        project_injection_context_token_count(projection_tokens(990, 10, 0))
            .expect("projection exactly at the hard maximum is not an excess");
    assert_eq!(projected_context_tokens, 1_000);
}

#[test]
fn should_reject_context_projection_beyond_the_hard_maximum() {
    let projection_outcome = project_injection_context_token_count(projection_tokens(1_000, 10, 0));
    match projection_outcome {
        Err(InferenceEngineError::InvalidRequest { reason }) => assert_eq!(
            reason,
            "generation context exceeds the model maximum position count"
        ),
        other => panic!("expected hard-maximum rejection, got {other:?}"),
    }
}

#[test]
fn should_reject_context_projection_on_token_count_overflow() {
    let projection_outcome =
        project_injection_context_token_count(projection_tokens(usize::MAX, 1, 0));
    match projection_outcome {
        Err(InferenceEngineError::InvalidRequest { reason }) => {
            assert_eq!(reason, "generation context token count overflowed")
        }
        other => panic!("expected overflow rejection, got {other:?}"),
    }
}
