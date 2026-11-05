//! Writes one whole dense MLP as a single Core ML ML Program:
//! `down(silu(gate(x)) · up(x))`, the form `hidden_act: silu` names.
//!
//! Splitting one projection's output channels between the engines costs a
//! hand-off per projection, and the hand-off journeys measured that taking
//! most of the gain (issue #161). Running the chain on the Neural Engine for a
//! slice of tokens needs one hand-off per MLP instead, because `gate`, `up`, the
//! activation, and `down` stay inside the program.
//!
//! `silu` is written as `gate · sigmoid(gate)`: MIL also has a `silu`
//! operation, but this two-operation form is the one whose Neural Engine
//! placement was proven here, and `silu` has not been tried. Each projection is cut into contraction parts because the Neural
//! Engine runs a 2,048-deep contraction at 18.6 TFLOP/s and a 4,096-deep one at
//! 9.7 (shape sweep, issue #161).

use crate::mil_program_wire::{MilDataType, MilOperation, int32_tensor_value};
use crate::ml_program_projection_document::{
    MlProgramPackageContents, SPECIFICATION_VERSION, activation_type, constant_operation,
    contraction_sum_operations, convolution_operation, convolution_parameter_constants, program,
    weight_blob_value,
};
use crate::ml_program_weight_blob::WeightBlobFile;
use crate::projection_model_interface::{
    ACTIVATIONS_FEATURE_NAME, PROJECTED_FEATURE_NAME, ProjectionDocumentError, model_description,
};
use crate::projection_shape::ProjectionShape;
use crate::protobuf_wire::{append_bytes_field, append_varint_field};
use crate::row_major_block::row_block;

const GATE_AND_UP_NAME: &str = "gate_and_up";
const GATE_NAME: &str = "gate";
const UP_NAME: &str = "up";
const SIGMOID_NAME: &str = "gate_sigmoid";
const SILU_NAME: &str = "gate_silu";
const GATED_NAME: &str = "gated";

/// The fp16 weights of one MLP, row-major `[output][input]`.
#[derive(Clone, Copy, Debug)]
pub struct MlpFp16Weights<'weights> {
    /// `[2 · intermediate][hidden]`: the gate's rows, then the up projection's.
    /// One fused convolution, because both read the same input.
    pub gate_and_up: &'weights [u16],
    /// `[hidden][intermediate]`.
    pub down: &'weights [u16],
}

/// One projection of the chain, cut into contraction parts.
struct ProjectionStage<'stage> {
    stage_name: &'stage str,
    input_name: &'stage str,
    output_name: &'stage str,
    shape: ProjectionShape,
    fp16_weights: &'stage [u16],
    part_count: usize,
}

/// Encodes the chain. `shape` is the whole program's `hidden → hidden` over its
/// tokens; `contraction_part_width` must divide both contraction depths.
pub fn encode_ml_program_mlp(
    shape: ProjectionShape,
    intermediate_channels: usize,
    weights: MlpFp16Weights<'_>,
    contraction_part_width: usize,
) -> Result<MlProgramPackageContents, ProjectionDocumentError> {
    let hidden_channels = shape.input_channels();
    let gate_and_up_shape = validated_stage_shape(
        hidden_channels,
        2 * intermediate_channels,
        shape.token_count(),
        weights.gate_and_up,
        contraction_part_width,
    )?;
    let down_shape = validated_stage_shape(
        intermediate_channels,
        shape.output_channels(),
        shape.token_count(),
        weights.down,
        contraction_part_width,
    )?;
    let mut weight_blob_file = WeightBlobFile::new();
    let mut operations = convolution_parameter_constants();
    operations.extend(projection_stage_operations(
        &ProjectionStage {
            stage_name: GATE_AND_UP_NAME,
            input_name: ACTIVATIONS_FEATURE_NAME,
            output_name: GATE_AND_UP_NAME,
            shape: gate_and_up_shape,
            fp16_weights: weights.gate_and_up,
            part_count: hidden_channels / contraction_part_width,
        },
        &mut weight_blob_file,
    ));
    operations.extend(silu_gating_operations(
        intermediate_channels,
        shape.token_count(),
    ));
    operations.extend(projection_stage_operations(
        &ProjectionStage {
            stage_name: "down",
            input_name: GATED_NAME,
            output_name: PROJECTED_FEATURE_NAME,
            shape: down_shape,
            fp16_weights: weights.down,
            part_count: intermediate_channels / contraction_part_width,
        },
        &mut weight_blob_file,
    ));
    let mut model_document = Vec::new();
    append_varint_field(&mut model_document, 1, SPECIFICATION_VERSION);
    append_bytes_field(&mut model_document, 2, &model_description(shape));
    append_bytes_field(&mut model_document, 502, &program(shape, &operations));
    Ok(MlProgramPackageContents {
        model_document,
        weight_blob_file: weight_blob_file.into_bytes(),
    })
}

fn validated_stage_shape(
    input_channels: usize,
    output_channels: usize,
    token_count: usize,
    fp16_weights: &[u16],
    contraction_part_width: usize,
) -> Result<ProjectionShape, ProjectionDocumentError> {
    if contraction_part_width == 0 || !input_channels.is_multiple_of(contraction_part_width) {
        return Err(ProjectionDocumentError::UnevenContractionPartWidth {
            input_channels,
            contraction_part_width,
        });
    }
    // A zero dimension leaves no weights to place, so it is reported as that.
    let Ok(stage_shape) = ProjectionShape::new(input_channels, output_channels, token_count) else {
        return Err(ProjectionDocumentError::WeightCountMismatch {
            expected: 0,
            actual: fp16_weights.len(),
        });
    };
    if fp16_weights.len() != stage_shape.weight_element_count() {
        return Err(ProjectionDocumentError::WeightCountMismatch {
            expected: stage_shape.weight_element_count(),
            actual: fp16_weights.len(),
        });
    }
    Ok(stage_shape)
}

fn split_axis_constant(axis_name: &str) -> Vec<u8> {
    constant_operation(
        axis_name,
        &int32_tensor_value(&[1], &[]),
        MilDataType::Int32,
        &[],
    )
}

fn split_count_constant(count_name: &str, split_count: usize) -> Vec<u8> {
    constant_operation(
        count_name,
        &int32_tensor_value(&[split_count as i32], &[]),
        MilDataType::Int32,
        &[],
    )
}

fn part_input_names(stage: &ProjectionStage<'_>) -> Vec<String> {
    if stage.part_count == 1 {
        return vec![stage.input_name.to_owned()];
    }
    let mut part_names = Vec::with_capacity(stage.part_count);
    for part_index in 0..stage.part_count {
        part_names.push(format!("{}_input_{part_index}", stage.stage_name));
    }
    part_names
}

fn stage_split_operations(stage: &ProjectionStage<'_>, part_inputs: &[String]) -> Vec<Vec<u8>> {
    let count_name = format!("{}_split_count", stage.stage_name);
    let axis_name = format!("{}_split_axis", stage.stage_name);
    let part_width = stage.shape.input_channels() / stage.part_count;
    let part_type = activation_type(part_width, stage.shape.token_count());
    let mut split_operation = MilOperation::new("split", &format!("{}_split", stage.stage_name))
        .input_named("x", stage.input_name)
        .input_named("num_splits", &count_name)
        .input_named("axis", &axis_name);
    for part_input in part_inputs {
        split_operation = split_operation.output(part_input, &part_type);
    }
    vec![
        split_count_constant(&count_name, stage.part_count),
        split_axis_constant(&axis_name),
        split_operation.into_bytes(),
    ]
}

fn stage_part_weight(
    stage: &ProjectionStage<'_>,
    part_index: usize,
    weight_name: &str,
    weight_blob_file: &mut WeightBlobFile,
) -> Vec<u8> {
    let part_width = stage.shape.input_channels() / stage.part_count;
    let part_weights = row_block(
        stage.fp16_weights,
        stage.shape.input_channels(),
        part_index * part_width,
        part_width,
    );
    let weight_dimensions = [stage.shape.output_channels(), part_width, 1, 1];
    let blob_offset = weight_blob_file.append_fp16(&part_weights);
    constant_operation(
        weight_name,
        &weight_blob_value(MilDataType::Float16, &weight_dimensions, blob_offset),
        MilDataType::Float16,
        &weight_dimensions,
    )
}

fn projection_stage_operations(
    stage: &ProjectionStage<'_>,
    weight_blob_file: &mut WeightBlobFile,
) -> Vec<Vec<u8>> {
    let part_inputs = part_input_names(stage);
    let mut operations = Vec::new();
    if stage.part_count > 1 {
        operations.extend(stage_split_operations(stage, &part_inputs));
    }
    let mut partial_projection_names = Vec::with_capacity(stage.part_count);
    for (part_index, part_input) in part_inputs.iter().enumerate() {
        let weight_name = format!("{}_weight_{part_index}", stage.stage_name);
        operations.push(stage_part_weight(
            stage,
            part_index,
            &weight_name,
            weight_blob_file,
        ));
        let partial_projection_name = if stage.part_count == 1 {
            stage.output_name.to_owned()
        } else {
            format!("{}_partial_{part_index}", stage.stage_name)
        };
        operations.push(convolution_operation(
            stage.shape,
            part_input,
            &weight_name,
            &partial_projection_name,
        ));
        partial_projection_names.push(partial_projection_name);
    }
    operations.extend(contraction_sum_operations(
        stage.shape,
        &partial_projection_names,
        &format!("{}_sum", stage.stage_name),
        stage.output_name,
    ));
    operations
}

/// Splits the fused result, then computes `gate · sigmoid(gate) · up`.
fn silu_gating_operations(intermediate_channels: usize, token_count: usize) -> Vec<Vec<u8>> {
    let intermediate_type = activation_type(intermediate_channels, token_count);
    vec![
        split_count_constant("gate_up_split_count", 2),
        split_axis_constant("gate_up_split_axis"),
        MilOperation::new("split", "gate_up_split")
            .input_named("x", GATE_AND_UP_NAME)
            .input_named("num_splits", "gate_up_split_count")
            .input_named("axis", "gate_up_split_axis")
            .output(GATE_NAME, &intermediate_type)
            .output(UP_NAME, &intermediate_type)
            .into_bytes(),
        MilOperation::new("sigmoid", SIGMOID_NAME)
            .input_named("x", GATE_NAME)
            .output(SIGMOID_NAME, &intermediate_type)
            .into_bytes(),
        MilOperation::new("mul", SILU_NAME)
            .input_named("x", GATE_NAME)
            .input_named("y", SIGMOID_NAME)
            .output(SILU_NAME, &intermediate_type)
            .into_bytes(),
        MilOperation::new("mul", GATED_NAME)
            .input_named("x", SILU_NAME)
            .input_named("y", UP_NAME)
            .output(GATED_NAME, &intermediate_type)
            .into_bytes(),
    ]
}

#[cfg(test)]
#[path = "ml_program_mlp_document_tests.rs"]
mod tests;
