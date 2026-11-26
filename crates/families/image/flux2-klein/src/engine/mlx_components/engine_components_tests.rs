//! Beside-source coverage for the pure text-conditioning geometry check that decides whether
//! finished conditioning covers exactly one request. Running the text-conditioning boundaries
//! themselves needs the text encoder weights, so they are covered by the model-serving image
//! lifecycle tests and the direct-MLX acceptance lane.

use super::{FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH, conditioning_describes_one_request};

/// The attention-mask shape of one request: a batch of one over the fixed conditioning length.
const ONE_REQUEST_MASK_SHAPE: [i32; 2] = [1, FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH as i32];

#[test]
fn should_accept_conditioning_for_exactly_one_request() {
    assert!(conditioning_describes_one_request(
        1,
        FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH,
        &ONE_REQUEST_MASK_SHAPE
    ));
}

#[test]
fn should_accept_an_attention_mask_whose_only_dimension_is_one() {
    assert!(conditioning_describes_one_request(
        1,
        FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH,
        &[1]
    ));
}

#[test]
fn should_reject_a_batch_of_zero_or_more_than_one() {
    assert!(!conditioning_describes_one_request(
        0,
        FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH,
        &ONE_REQUEST_MASK_SHAPE
    ));
    assert!(!conditioning_describes_one_request(
        2,
        FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH,
        &ONE_REQUEST_MASK_SHAPE
    ));
}

#[test]
fn should_reject_a_sequence_length_one_off_the_fixed_conditioning_length() {
    assert!(!conditioning_describes_one_request(
        1,
        FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH - 1,
        &ONE_REQUEST_MASK_SHAPE
    ));
    assert!(!conditioning_describes_one_request(
        1,
        FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH + 1,
        &ONE_REQUEST_MASK_SHAPE
    ));
}

#[test]
fn should_reject_an_attention_mask_that_does_not_lead_with_one() {
    assert!(!conditioning_describes_one_request(
        1,
        FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH,
        &[2, ONE_REQUEST_MASK_SHAPE[1]]
    ));
    assert!(!conditioning_describes_one_request(
        1,
        FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH,
        &[0, ONE_REQUEST_MASK_SHAPE[1]]
    ));
}

#[test]
fn should_reject_a_zero_dimensional_attention_mask_without_panicking() {
    assert!(!conditioning_describes_one_request(
        1,
        FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH,
        &[]
    ));
}
