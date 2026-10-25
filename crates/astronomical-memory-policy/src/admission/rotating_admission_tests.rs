//! Boundary and negative tests for rotating-cache admission geometry.

use crate::admission::rotating::{
    RotatingAdmissionError, rotating_committed_token_count, rotating_prefill_transient_token_count,
};

#[test]
fn should_reject_a_zero_window_size() {
    assert_eq!(
        rotating_prefill_transient_token_count(0, 8),
        Err(RotatingAdmissionError::ZeroWindowSize)
    );
}

#[test]
fn should_return_the_window_itself_for_a_zero_token_chunk() {
    assert_eq!(rotating_prefill_transient_token_count(4, 0), Ok(4));
}

#[test]
fn should_peak_at_window_plus_chunk_minus_one_tokens() {
    assert_eq!(rotating_prefill_transient_token_count(4, 8), Ok(11));
    assert_eq!(rotating_prefill_transient_token_count(1, 1), Ok(1));
}

#[test]
fn should_admit_a_full_window_committing_one_token_at_the_integer_boundary() {
    // A full u32 window committing a one-token chunk peaks at exactly
    // u32::MAX, which is representable; rejecting it would shrink admission
    // below the geometry the caller actually owns.
    assert_eq!(
        rotating_prefill_transient_token_count(u32::MAX, 1),
        Ok(u32::MAX)
    );
}

#[test]
fn should_reject_when_the_transient_peak_exceeds_the_integer_boundary() {
    assert_eq!(
        rotating_prefill_transient_token_count(u32::MAX, 2),
        Err(RotatingAdmissionError::TransientTokenCountOverflow {
            window_size: u32::MAX,
            prompt_chunk_token_count: 2,
        })
    );
}

#[test]
fn should_commit_no_more_than_the_window_after_a_rotating_update() {
    assert_eq!(rotating_committed_token_count(8, 0), 0);
    assert_eq!(rotating_committed_token_count(8, 8), 8);
    assert_eq!(rotating_committed_token_count(8, u32::MAX), 8);
}
