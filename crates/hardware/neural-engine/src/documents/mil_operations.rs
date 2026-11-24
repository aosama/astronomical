//! The ML Program building blocks every model document shares: the program
//! envelope, the convolution that does the projecting, and the small constants
//! around it. Keeping them apart from any one document lets a new model shape
//! reuse them without depending on another shape's encoder.

use crate::documents::model_interface::{ACTIVATIONS_FEATURE_NAME, PROJECTED_FEATURE_NAME};
use crate::program_shape::ProgramShape;
use crate::wire::mil_program::{
    MilDataType, MilOperation, append_map_entry, blob_file_tensor_value, int32_tensor_value,
    named_value_type, string_scalar_value, tensor_value_type,
};
use crate::wire::protobuf::{append_bytes_field, append_string_field, append_varint_field};

// Specification version 9 is iOS 18 / macOS 15, the first with block-wise
// 4-bit weights; its MIL opset is "CoreML8".
pub(crate) const SPECIFICATION_VERSION: u64 = 9;
const OPSET_NAME: &str = "CoreML8";
const MAIN_FUNCTION_NAME: &str = "main";

/// Where the program expects its weights, relative to the model package.
const WEIGHT_BLOB_FILE_NAME: &str = "@model_path/weights/weight.bin";

// A 1 × 1 kernel with unit stride and no padding is exactly a matrix
// multiply over channels; these are the values Apple's converter writes as
// conv defaults, emitted once and shared by every part.
const CONVOLUTION_PARAMETER_CONSTANTS: [&str; 5] = [
    "conv_strides",
    "conv_pad_type",
    "conv_pad",
    "conv_dilations",
    "conv_groups",
];

pub(crate) fn convolution_parameter_constants() -> Vec<Vec<u8>> {
    vec![
        constant_operation(
            "conv_strides",
            &int32_tensor_value(&[1, 1], &[2]),
            MilDataType::Int32,
            &[2],
        ),
        constant_operation(
            "conv_pad_type",
            &string_scalar_value("valid"),
            MilDataType::String,
            &[],
        ),
        constant_operation(
            "conv_pad",
            &int32_tensor_value(&[0, 0, 0, 0], &[4]),
            MilDataType::Int32,
            &[4],
        ),
        constant_operation(
            "conv_dilations",
            &int32_tensor_value(&[1, 1], &[2]),
            MilDataType::Int32,
            &[2],
        ),
        constant_operation(
            "conv_groups",
            &int32_tensor_value(&[1], &[]),
            MilDataType::Int32,
            &[],
        ),
    ]
}

pub(crate) fn constant_operation(
    constant_name: &str,
    encoded_value: &[u8],
    data_type: MilDataType,
    dimensions: &[usize],
) -> Vec<u8> {
    MilOperation::new("const", constant_name)
        .attribute("val", encoded_value)
        .output(constant_name, &tensor_value_type(data_type, dimensions))
        .into_bytes()
}

pub(crate) fn weight_blob_value(
    data_type: MilDataType,
    dimensions: &[usize],
    blob_offset: u64,
) -> Vec<u8> {
    blob_file_tensor_value(data_type, dimensions, WEIGHT_BLOB_FILE_NAME, blob_offset)
}

pub(crate) fn convolution_operation(
    shape: ProgramShape,
    input_name: &str,
    weight_name: &str,
    output_name: &str,
) -> Vec<u8> {
    let mut convolution = MilOperation::new("conv", output_name)
        .input_named("x", input_name)
        .input_named("weight", weight_name);
    for (parameter_name, constant_name) in ["strides", "pad_type", "pad", "dilations", "groups"]
        .iter()
        .zip(CONVOLUTION_PARAMETER_CONSTANTS)
    {
        convolution = convolution.input_named(parameter_name, constant_name);
    }
    convolution
        .output(
            output_name,
            &activation_type(shape.output_channels(), shape.token_count()),
        )
        .into_bytes()
}

/// `partial_0 + partial_1 + …` as a chain of binary adds ending in
/// `final_output_name`; MIL's `add` takes exactly two operands.
pub(crate) fn contraction_sum_operations(
    shape: ProgramShape,
    partial_projection_names: &[String],
    sum_name_prefix: &str,
    final_output_name: &str,
) -> Vec<Vec<u8>> {
    let projected_type = activation_type(shape.output_channels(), shape.token_count());
    let mut operations = Vec::new();
    let mut running_sum_name = partial_projection_names[0].clone();
    for (part_index, partial_projection_name) in partial_projection_names.iter().enumerate().skip(1)
    {
        let sum_name = if part_index + 1 == partial_projection_names.len() {
            final_output_name.to_owned()
        } else {
            format!("{sum_name_prefix}_{part_index}")
        };
        operations.push(
            MilOperation::new("add", &sum_name)
                .input_named("x", &running_sum_name)
                .input_named("y", partial_projection_name)
                .output(&sum_name, &projected_type)
                .into_bytes(),
        );
        running_sum_name = sum_name;
    }
    operations
}

pub(crate) fn activation_type(channel_count: usize, token_count: usize) -> Vec<u8> {
    tensor_value_type(MilDataType::Float16, &[1, channel_count, 1, token_count])
}

pub(crate) fn program(shape: ProgramShape, operations: &[Vec<u8>]) -> Vec<u8> {
    let mut block = Vec::new();
    append_string_field(&mut block, 2, PROJECTED_FEATURE_NAME);
    for operation in operations {
        append_bytes_field(&mut block, 3, operation);
    }
    let mut function = Vec::new();
    append_bytes_field(
        &mut function,
        1,
        &named_value_type(
            ACTIVATIONS_FEATURE_NAME,
            &activation_type(shape.input_channels(), shape.token_count()),
        ),
    );
    append_string_field(&mut function, 2, OPSET_NAME);
    append_map_entry(&mut function, 3, OPSET_NAME, &block);
    let mut program = Vec::new();
    append_varint_field(&mut program, 1, 1);
    append_map_entry(&mut program, 2, MAIN_FUNCTION_NAME, &function);
    program
}

pub(crate) fn split_axis_constant(axis_name: &str) -> Vec<u8> {
    constant_operation(
        axis_name,
        &int32_tensor_value(&[1], &[]),
        MilDataType::Int32,
        &[],
    )
}

pub(crate) fn split_count_constant(count_name: &str, split_count: usize) -> Vec<u8> {
    constant_operation(
        count_name,
        &int32_tensor_value(&[split_count as i32], &[]),
        MilDataType::Int32,
        &[],
    )
}

#[cfg(test)]
#[path = "mil_operations_tests.rs"]
mod tests;
