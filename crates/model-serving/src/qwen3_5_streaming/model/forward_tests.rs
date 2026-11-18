//! Beside-source coverage for the pure units of the MoE forward dispatch:
//! the paged token count and the distinct experts of a route. The routing and paging units run under the serial engine
//! lane against real arrays.

use super::{paged_token_count, sorted_unique_ids};
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

#[test]
fn should_list_each_routed_expert_once_in_ascending_order() {
    // Two rows of a draft round that share expert 7 and name 3 twice.
    assert_eq!(sorted_unique_ids(&[9, 7, 3, 3, 7, 200]), vec![3, 7, 9, 200]);
}

#[test]
fn should_keep_a_route_that_is_already_distinct_and_ascending() {
    assert_eq!(sorted_unique_ids(&[0, 1, 255]), vec![0, 1, 255]);
}

#[test]
fn should_list_no_expert_for_an_empty_route() {
    assert!(sorted_unique_ids(&[]).is_empty());
}
