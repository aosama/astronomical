//! Unit tests for output-size estimation; plan construction against written
//! shards lives in `tests/offline/requantize_plan.rs`.

use super::*;
use crate::bit_width::{BitWidth, GroupSize};

#[test]
#[ignore = "pending implementation of estimate_output_bytes"]
fn should_estimate_packed_weight_plus_scales_and_biases() {
    // 1024 elements at 4 bits = 512 bytes; 16 groups of 64, each with a
    // 2-byte scale and a 2-byte bias = 64 bytes.
    let estimate = RequantizeTensorPlan::estimate_output_bytes(
        &[16, 64],
        TensorEncodingDecision::Encode {
            bit_width: BitWidth::new(4).expect("valid"),
            group_size: GroupSize::new(64).expect("valid"),
        },
    );

    assert_eq!(estimate, 512 + 64);
}

#[test]
#[ignore = "pending implementation of estimate_output_bytes"]
fn should_estimate_pass_through_tensors_at_bfloat16() {
    let estimate =
        RequantizeTensorPlan::estimate_output_bytes(&[16, 64], TensorEncodingDecision::PassThrough);

    assert_eq!(estimate, 1024 * 2);
}
