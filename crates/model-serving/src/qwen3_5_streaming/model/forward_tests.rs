//! Beside-source coverage for the pure paged-token-count unit of the MoE
//! forward dispatch. The routing and paging units run under the serial engine
//! lane against real arrays.

use super::paged_token_count;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;

#[test]
fn should_read_the_token_count_from_the_second_to_last_dimension() {
    let chunk_shape = [1, 5, 768];
    assert_eq!(
        paged_token_count(&chunk_shape).expect("a chunk shape has tokens"),
        5
    );

    let decode_shape = [1, 1, 768];
    assert_eq!(
        paged_token_count(&decode_shape).expect("a decode shape has tokens"),
        1
    );
}

#[test]
fn should_reject_an_empty_shape_as_invalid_input() {
    let outcome = paged_token_count(&[]);

    assert!(matches!(
        outcome,
        Err(Qwen3_5ExecutionError::InvalidInput { description })
            if description == "paged MoE hidden states must include token and hidden dimensions"
    ));
}

#[test]
fn should_fall_back_to_index_zero_for_a_single_dimension_shape() {
    // A single-dimension shape saturates the second-to-last index at zero,
    // so the only dimension is read as the token count; this documents the
    // actual dispatch behavior rather than an intended one.
    let flat_shape = [768];
    assert_eq!(
        paged_token_count(&flat_shape).expect("a flat shape has one dimension"),
        768
    );
}
