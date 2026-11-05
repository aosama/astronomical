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
//! placement was proven here, and `silu` has not been tried. Each projection is
//! cut into contraction parts ([`crate::documents::mlp_stages`]) because the
//! Neural Engine runs a 2,048-deep contraction at 18.6 TFLOP/s and a 4,096-deep
//! one at 9.7 (shape sweep, issue #161).

use crate::documents::mil_operations::{
    SPECIFICATION_VERSION, convolution_parameter_constants, program,
};
use crate::documents::mlp_stages::{
    GATE_AND_UP_NAME, GATED_NAME, ProjectionStage, StageWeightEncoding,
    projection_stage_operations, silu_gating_operations, validated_stage_shape,
};
use crate::documents::model_interface::{
    ACTIVATIONS_FEATURE_NAME, ModelDocumentError, PROJECTED_FEATURE_NAME, model_description,
};
use crate::documents::package_layout::MlProgramPackageContents;
use crate::program_shape::ProgramShape;
use crate::wire::protobuf::{append_bytes_field, append_varint_field};
use crate::wire::weight_blob::WeightBlobFile;

/// The fp16 weights of one MLP, row-major `[output][input]`.
#[derive(Clone, Copy, Debug)]
pub struct MlpFp16Weights<'weights> {
    /// `[2 · intermediate][hidden]`: the gate's rows, then the up projection's.
    /// One fused convolution, because both read the same input.
    pub gate_and_up: &'weights [u16],
    /// `[hidden][intermediate]`.
    pub down: &'weights [u16],
}

/// Encodes the chain. `shape` is the whole program's `hidden → hidden` over its
/// tokens; `contraction_part_width` must divide both contraction depths.
pub fn encode_ml_program_mlp(
    shape: ProgramShape,
    intermediate_channels: usize,
    weights: MlpFp16Weights<'_>,
    contraction_part_width: usize,
) -> Result<MlProgramPackageContents, ModelDocumentError> {
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
            weight_encoding: StageWeightEncoding::Fp16,
        },
        &mut weight_blob_file,
    )?);
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
            weight_encoding: StageWeightEncoding::Fp16,
        },
        &mut weight_blob_file,
    )?);
    let mut model_document = Vec::new();
    append_varint_field(&mut model_document, 1, SPECIFICATION_VERSION);
    append_bytes_field(&mut model_document, 2, &model_description(shape));
    append_bytes_field(&mut model_document, 502, &program(shape, &operations));
    Ok(MlProgramPackageContents {
        model_document,
        weight_blob_file: weight_blob_file.into_bytes(),
    })
}

#[cfg(test)]
#[path = "mlp_tests.rs"]
mod tests;
