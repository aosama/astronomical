use mlx_bridge::MlxDtype;
use runtime_integration::MlxRuntimeError;

use super::{
    CODEBOOK_DECODE_OPERATION, plan_page_decode_launch, validate_page_operand_types,
    validate_page_shapes,
};

/// The description a page decode error carries, after checking it names the
/// codebook decode operation.
fn decode_error_description(runtime_error: MlxRuntimeError) -> String {
    match runtime_error {
        MlxRuntimeError::RuntimeOperation {
            operation,
            description,
        } => {
            assert_eq!(operation, CODEBOOK_DECODE_OPERATION);
            description
        }
        other_error => panic!("expected a runtime operation error, got {other_error:?}"),
    }
}

#[test]
fn should_accept_every_codebook_bit_width_with_uint32_codes_and_float16_scales() {
    for bit_width in [2, 3, 4] {
        assert!(
            validate_page_operand_types(bit_width, MlxDtype::UInt32, MlxDtype::Float16).is_ok(),
            "bit width {bit_width} should be accepted"
        );
    }
}

#[test]
fn should_reject_a_bit_width_outside_the_codebook_family() {
    for unsupported_bit_width in [0, 1, 5, 8] {
        let validation_error =
            validate_page_operand_types(unsupported_bit_width, MlxDtype::UInt32, MlxDtype::Float16)
                .expect_err("a width outside 2 to 4 must be rejected");
        assert_eq!(
            decode_error_description(validation_error),
            format!("unsupported jangtq2 codebook bit width {unsupported_bit_width}")
        );
    }
}

#[test]
fn should_reject_packed_words_and_scales_of_the_wrong_element_type() {
    let packed_error =
        validate_page_operand_types(2, MlxDtype::Float16, MlxDtype::Float16).unwrap_err();
    let scale_error =
        validate_page_operand_types(2, MlxDtype::UInt32, MlxDtype::BFloat16).unwrap_err();

    assert!(
        decode_error_description(packed_error).starts_with("packed code words must use uint32")
    );
    assert!(decode_error_description(scale_error).starts_with("row scales must use float16"));
}

#[test]
fn should_accept_matching_rank_three_codes_and_rank_two_scales() {
    assert!(validate_page_shapes(&[4, 8, 12], &[4, 8]).is_ok());
    // One expert with one row is the smallest page.
    assert!(validate_page_shapes(&[1, 1, 1], &[1, 1]).is_ok());
}

#[test]
fn should_reject_codes_and_scales_of_the_wrong_rank() {
    let validation_error = validate_page_shapes(&[4, 8], &[4, 8, 12]).unwrap_err();

    assert_eq!(
        decode_error_description(validation_error),
        "packed tensors must have rank three and row scales rank two, got ranks 2 and 3"
    );
}

#[test]
fn should_reject_empty_or_disagreeing_page_dimensions() {
    let empty_expert_axis_error = validate_page_shapes(&[0, 8, 12], &[0, 8]).unwrap_err();
    let disagreeing_rows_error = validate_page_shapes(&[4, 8, 12], &[4, 9]).unwrap_err();

    assert_eq!(
        decode_error_description(empty_expert_axis_error),
        "packed code shape [0, 8, 12] and row scale shape [0, 8] disagree"
    );
    assert_eq!(
        decode_error_description(disagreeing_rows_error),
        "packed code shape [4, 8, 12] and row scale shape [4, 9] disagree"
    );
}

#[test]
fn should_plan_a_two_bit_page_launch_with_one_thread_per_rotation_block() {
    let launch_plan = plan_page_decode_launch(vec![2, 8, 12], 2).expect("a valid two-bit page");

    // 12 words of 32 bits at 2 bits per code decode to 192 inputs per row.
    assert_eq!(launch_plan.input_width_i32, 192);
    assert_eq!(launch_plan.bits_i32, 2);
    assert_eq!(launch_plan.bit_width, 2);
    assert_eq!(launch_plan.packed_shape, vec![2, 8, 12]);
    // 16 rows, each split into 192 / 32 = 6 rotation blocks.
    assert_eq!(launch_plan.thread_count, 96);
}

#[test]
fn should_plan_the_smallest_four_bit_page_launch() {
    let launch_plan = plan_page_decode_launch(vec![1, 1, 4], 4).expect("a valid four-bit page");

    assert_eq!(launch_plan.input_width_i32, 32);
    assert_eq!(launch_plan.thread_count, 1);
}

#[test]
fn should_reject_a_packed_row_that_is_not_a_whole_number_of_code_groups() {
    // Three-bit rows of one word hold 32 / 3 codes, which is not a whole
    // number, so the page is not a valid codebook page.
    let planning_error = plan_page_decode_launch(vec![1, 1, 1], 3).unwrap_err();

    assert_eq!(
        decode_error_description(planning_error),
        "packed source shape is not a valid codebook page"
    );
}

#[test]
fn should_reject_a_page_whose_row_count_overflows_the_mlx_grid() {
    let planning_error = plan_page_decode_launch(vec![i32::MAX, 2, 2], 2).unwrap_err();

    assert_eq!(
        decode_error_description(planning_error),
        "page row count exceeds the MLX grid range"
    );
}
