use super::{
    contraction_sum_operations, convolution_operation, convolution_parameter_constants,
    split_axis_constant, split_count_constant,
};
use crate::program_shape::ProgramShape;

const FINAL_OUTPUT_NAME: &str = "projected";

fn contains_text(encoded: &[u8], expected_text: &str) -> bool {
    for window in encoded.windows(expected_text.len()) {
        if window == expected_text.as_bytes() {
            return true;
        }
    }
    false
}

fn shape() -> ProgramShape {
    ProgramShape::new(8, 4, 2).expect("valid shape")
}

fn part_names(part_count: usize) -> Vec<String> {
    let mut names = Vec::with_capacity(part_count);
    for part_index in 0..part_count {
        names.push(format!("partial_{part_index}"));
    }
    names
}

#[test]
fn should_write_no_sum_for_a_single_contraction_part() {
    let sum_operations =
        contraction_sum_operations(shape(), &part_names(1), "sum", FINAL_OUTPUT_NAME);

    assert!(sum_operations.is_empty());
}

#[test]
fn should_chain_two_parts_directly_into_the_final_output() {
    let sum_operations =
        contraction_sum_operations(shape(), &part_names(2), "sum", FINAL_OUTPUT_NAME);

    assert_eq!(sum_operations.len(), 1);
    assert!(contains_text(&sum_operations[0], "partial_0"));
    assert!(contains_text(&sum_operations[0], "partial_1"));
    assert!(contains_text(&sum_operations[0], FINAL_OUTPUT_NAME));
}

#[test]
fn should_feed_each_running_sum_into_the_next_part() {
    let sum_operations =
        contraction_sum_operations(shape(), &part_names(3), "sum", FINAL_OUTPUT_NAME);

    assert_eq!(sum_operations.len(), 2);
    assert!(contains_text(&sum_operations[0], "sum_1"));
    assert!(contains_text(&sum_operations[1], "sum_1"));
    assert!(contains_text(&sum_operations[1], "partial_2"));
    assert!(contains_text(&sum_operations[1], FINAL_OUTPUT_NAME));
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn should_refuse_a_sum_of_no_parts() {
    contraction_sum_operations(shape(), &[], "sum", FINAL_OUTPUT_NAME);
}

#[test]
fn should_bind_every_constant_the_convolution_needs() {
    let constants = convolution_parameter_constants();
    let convolution = convolution_operation(shape(), "activations", "weight", "projected");

    assert_eq!(constants.len(), 5);
    for constant_name in [
        "conv_strides",
        "conv_pad_type",
        "conv_pad",
        "conv_dilations",
        "conv_groups",
    ] {
        assert!(contains_text(&convolution, constant_name));
    }
    assert!(contains_text(&convolution, "activations"));
    assert!(contains_text(&convolution, "weight"));
}

#[test]
fn should_name_the_split_constants_it_is_given() {
    assert!(contains_text(
        &split_axis_constant("rotation_axis"),
        "rotation_axis"
    ));
    assert!(contains_text(
        &split_count_constant("rotation_count", 4),
        "rotation_count"
    ));
}
