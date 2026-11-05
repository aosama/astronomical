//! Writes one whole dense MLP of a folded ternary checkpoint as a single Core ML
//! ML Program: `down(rotate(silu(gate(x)) · up(x)))`.
//!
//! Folded packs store their weights in a Hadamard-rotated basis, so the
//! activation must be rotated before each projection. The first rotation stays
//! on the GPU, because the GPU already rotates the same activations for its own
//! tokens. The second sits between the activation and `down`, so it has to run
//! inside the program: leaving it on the GPU would cost a second hand-off per
//! layer, which is the cost the token split exists to avoid.
//!
//! The rotation is written as one dense 1 × 1 convolution per Hadamard block,
//! all sharing one weight constant. A fast Walsh-Hadamard transform would take
//! fewer operations, but its radix passes need reshapes between stages, and this
//! convolution is the form already proven on the Neural Engine. It adds about 7
//! percent to the program's work at the 27B shape (17,408 × 1,024 against 267
//! million weights per layer).
//!
//! The rotation runs in fp16 here, where the upstream reference casts to fp32,
//! so this program is an approximation of the checkpoint's arithmetic. The
//! journeys measure how close it comes against the GPU.

use crate::half_precision::f32_to_f16_bits;
use crate::mil_program_wire::{MilDataType, MilOperation, bool_scalar_value};
use crate::ml_program_mlp_document::{
    GATE_AND_UP_NAME, GATED_NAME, ProjectionStage, StageWeightEncoding,
    projection_stage_operations, silu_gating_operations, split_axis_constant, split_count_constant,
    validated_stage_shape,
};
use crate::ml_program_projection_document::{
    MlProgramPackageContents, SPECIFICATION_VERSION, activation_type, constant_operation,
    convolution_operation, convolution_parameter_constants, program, weight_blob_value,
};
use crate::ml_program_weight_blob::WeightBlobFile;
use crate::projection_model_interface::{
    ACTIVATIONS_FEATURE_NAME, PROJECTED_FEATURE_NAME, ProjectionDocumentError, model_description,
};
use crate::projection_shape::ProjectionShape;
use crate::protobuf_wire::{append_bytes_field, append_varint_field};

const SIGNS_NAME: &str = "rotation_signs";
const SIGNED_NAME: &str = "signed_gated";
const ROTATED_NAME: &str = "rotated_gated";
const HADAMARD_WEIGHT_NAME: &str = "hadamard_weight";

/// The shape of one rotated MLP program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RotatedMlpLayout {
    /// `hidden → hidden` over this program's tokens.
    pub shape: ProjectionShape,
    pub intermediate_channels: usize,
    pub contraction_part_width: usize,
    pub rotation_block_size: usize,
    pub weight_encoding: StageWeightEncoding,
}

/// Checkpoint weights, row-major `[output][input]`, in the rotated basis.
#[derive(Clone, Copy, Debug)]
pub struct RotatedMlpWeights<'weights> {
    /// `[2 · intermediate][hidden]`: the gate's rows, then the up projection's.
    pub gate_and_up: &'weights [u16],
    /// `[hidden][intermediate]`.
    pub down: &'weights [u16],
    /// The pack's explicit ±1 signs for the intermediate width, as fp16.
    pub intermediate_signs: &'weights [u16],
}

/// The normalized Sylvester-ordered Walsh-Hadamard matrix, row-major fp16.
/// Entry `(row, column)` is `(−1)^popcount(row & column) / √size`.
#[must_use]
pub fn normalized_hadamard_fp16(block_size: usize) -> Vec<u16> {
    let positive_entry = f32_to_f16_bits(1.0 / (block_size as f32).sqrt());
    let negative_entry = f32_to_f16_bits(-1.0 / (block_size as f32).sqrt());
    let mut matrix = Vec::with_capacity(block_size * block_size);
    for row_index in 0..block_size {
        for column_index in 0..block_size {
            let is_positive = (row_index & column_index).count_ones().is_multiple_of(2);
            matrix.push(if is_positive {
                positive_entry
            } else {
                negative_entry
            });
        }
    }
    matrix
}

pub fn encode_ml_program_rotated_mlp(
    layout: RotatedMlpLayout,
    weights: RotatedMlpWeights<'_>,
) -> Result<MlProgramPackageContents, ProjectionDocumentError> {
    let token_count = layout.shape.token_count();
    let hidden_channels = layout.shape.input_channels();
    let intermediate_channels = layout.intermediate_channels;
    let part_width = layout.contraction_part_width;
    validate_rotation(layout, weights.intermediate_signs.len())?;
    let gate_and_up_shape = validated_stage_shape(
        hidden_channels,
        2 * intermediate_channels,
        token_count,
        weights.gate_and_up,
        part_width,
    )?;
    let down_shape = validated_stage_shape(
        intermediate_channels,
        layout.shape.output_channels(),
        token_count,
        weights.down,
        part_width,
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
            part_count: hidden_channels / part_width,
            weight_encoding: layout.weight_encoding,
        },
        &mut weight_blob_file,
    )?);
    operations.extend(silu_gating_operations(intermediate_channels, token_count));
    operations.extend(rotation_operations(
        layout,
        weights.intermediate_signs,
        &mut weight_blob_file,
    ));
    operations.extend(projection_stage_operations(
        &ProjectionStage {
            stage_name: "down",
            input_name: ROTATED_NAME,
            output_name: PROJECTED_FEATURE_NAME,
            shape: down_shape,
            fp16_weights: weights.down,
            part_count: intermediate_channels / part_width,
            weight_encoding: layout.weight_encoding,
        },
        &mut weight_blob_file,
    )?);
    let mut model_document = Vec::new();
    append_varint_field(&mut model_document, 1, SPECIFICATION_VERSION);
    append_bytes_field(&mut model_document, 2, &model_description(layout.shape));
    append_bytes_field(
        &mut model_document,
        502,
        &program(layout.shape, &operations),
    );
    Ok(MlProgramPackageContents {
        model_document,
        weight_blob_file: weight_blob_file.into_bytes(),
    })
}

fn validate_rotation(
    layout: RotatedMlpLayout,
    sign_count: usize,
) -> Result<(), ProjectionDocumentError> {
    let intermediate_channels = layout.intermediate_channels;
    let block_size = layout.rotation_block_size;
    // The popcount rule only yields an orthogonal matrix at powers of two; any
    // other size would encode and run while silently corrupting the activations.
    if !block_size.is_power_of_two() || !intermediate_channels.is_multiple_of(block_size) {
        return Err(ProjectionDocumentError::UnevenRotationBlocks {
            channel_count: intermediate_channels,
            block_size,
        });
    }
    if sign_count != intermediate_channels {
        return Err(ProjectionDocumentError::WeightCountMismatch {
            expected: intermediate_channels,
            actual: sign_count,
        });
    }
    Ok(())
}

/// Signs, then one convolution per Hadamard block, then the concatenation.
fn rotation_operations(
    layout: RotatedMlpLayout,
    intermediate_signs: &[u16],
    weight_blob_file: &mut WeightBlobFile,
) -> Vec<Vec<u8>> {
    let token_count = layout.shape.token_count();
    let intermediate_channels = layout.intermediate_channels;
    let block_size = layout.rotation_block_size;
    let block_count = intermediate_channels / block_size;
    let intermediate_type = activation_type(intermediate_channels, token_count);
    let block_type = activation_type(block_size, token_count);
    let block_shape = ProjectionShape::new(block_size, block_size, token_count)
        .expect("the block size and token count were validated as non-zero");
    let mut operations = rotation_constant_operations(layout, intermediate_signs, weight_blob_file);
    operations.push(
        MilOperation::new("mul", SIGNED_NAME)
            .input_named("x", GATED_NAME)
            .input_named("y", SIGNS_NAME)
            .output(SIGNED_NAME, &intermediate_type)
            .into_bytes(),
    );
    let mut block_split = MilOperation::new("split", "rotation_split")
        .input_named("x", SIGNED_NAME)
        .input_named("num_splits", "rotation_split_count")
        .input_named("axis", "rotation_split_axis");
    let mut rotated_block_names = Vec::with_capacity(block_count);
    let mut block_convolutions = Vec::with_capacity(block_count);
    for block_index in 0..block_count {
        let block_name = format!("signed_block_{block_index}");
        let rotated_block_name = format!("rotated_block_{block_index}");
        block_split = block_split.output(&block_name, &block_type);
        block_convolutions.push(convolution_operation(
            block_shape,
            &block_name,
            HADAMARD_WEIGHT_NAME,
            &rotated_block_name,
        ));
        rotated_block_names.push(rotated_block_name);
    }
    operations.push(block_split.into_bytes());
    operations.extend(block_convolutions);
    operations.push(
        MilOperation::new("concat", ROTATED_NAME)
            .input_named_list("values", &rotated_block_names)
            .input_named("axis", "rotation_split_axis")
            .input_named("interleave", "rotation_concat_interleave")
            .output(ROTATED_NAME, &intermediate_type)
            .into_bytes(),
    );
    operations
}

fn rotation_constant_operations(
    layout: RotatedMlpLayout,
    intermediate_signs: &[u16],
    weight_blob_file: &mut WeightBlobFile,
) -> Vec<Vec<u8>> {
    let block_size = layout.rotation_block_size;
    let sign_dimensions = [1, layout.intermediate_channels, 1, 1];
    let hadamard_dimensions = [block_size, block_size, 1, 1];
    let signs_offset = weight_blob_file.append_fp16(intermediate_signs);
    let hadamard_offset = weight_blob_file.append_fp16(&normalized_hadamard_fp16(block_size));
    vec![
        constant_operation(
            SIGNS_NAME,
            &weight_blob_value(MilDataType::Float16, &sign_dimensions, signs_offset),
            MilDataType::Float16,
            &sign_dimensions,
        ),
        constant_operation(
            HADAMARD_WEIGHT_NAME,
            &weight_blob_value(MilDataType::Float16, &hadamard_dimensions, hadamard_offset),
            MilDataType::Float16,
            &hadamard_dimensions,
        ),
        split_count_constant(
            "rotation_split_count",
            layout.intermediate_channels / block_size,
        ),
        split_axis_constant("rotation_split_axis"),
        constant_operation(
            "rotation_concat_interleave",
            &bool_scalar_value(false),
            MilDataType::Bool,
            &[],
        ),
    ]
}

#[cfg(test)]
#[path = "ml_program_rotated_mlp_document_tests.rs"]
mod tests;
