//! The real ternary MLP's shape and program loading, shared by its accuracy and
//! memory journeys.

use astronomical_neural_engine::core_ml_projection::{CoreMlComputeUnits, CoreMlProjection};
use astronomical_neural_engine::half_precision::f32_to_f16_bits;
use astronomical_neural_engine::ml_program_mlp_document::StageWeightEncoding;
use astronomical_neural_engine::ml_program_rotated_mlp_document::{
    RotatedMlpLayout, RotatedMlpWeights,
};
use astronomical_neural_engine::projection_shape::ProjectionShape;

use super::ternary_checkpoint::TernaryMlpLayer;

pub(crate) const HIDDEN_CHANNELS: usize = 5120;
pub(crate) const INTERMEDIATE_CHANNELS: usize = 17_408;

// The deepest width dividing both 5,120 and 17,408.
pub(crate) const CONTRACTION_PART_WIDTH: usize = 1024;
pub(crate) const ROTATION_BLOCK_SIZE: usize = 1024;
pub(crate) const ROWS_PER_LOOKUP_TABLE: usize = 16;

// The share serving uses on the 9B, and a multiple of 64 so MLX can read the
// surface directly. The journey prints enough to choose a better balance.
pub(crate) const NEURAL_ENGINE_TOKEN_COUNT: usize = 960;
pub(crate) const LAYER_COUNT: f64 = 64.0;

pub(crate) fn load_program(
    layer: &TernaryMlpLayer,
    weight_encoding: StageWeightEncoding,
    directory: &std::path::Path,
) -> (CoreMlProjection, usize) {
    let mut gate_and_up_weights = layer.gate.fp16_weights();
    gate_and_up_weights.extend(layer.up.fp16_weights());
    let down_weights = layer.down.fp16_weights();
    let mut fp16_signs = Vec::with_capacity(layer.down.signs.len());
    for sign in &layer.down.signs {
        fp16_signs.push(f32_to_f16_bits(*sign));
    }
    let shape = ProjectionShape::new(HIDDEN_CHANNELS, HIDDEN_CHANNELS, NEURAL_ENGINE_TOKEN_COUNT)
        .expect("valid shape");
    let weight_count = gate_and_up_weights.len() + down_weights.len();
    let projection = CoreMlProjection::load_ml_program_rotated_mlp(
        RotatedMlpLayout {
            shape,
            intermediate_channels: INTERMEDIATE_CHANNELS,
            contraction_part_width: CONTRACTION_PART_WIDTH,
            rotation_block_size: ROTATION_BLOCK_SIZE,
            weight_encoding,
        },
        RotatedMlpWeights {
            gate_and_up: &gate_and_up_weights,
            down: &down_weights,
            intermediate_signs: &fp16_signs,
        },
        CoreMlComputeUnits::CpuAndNeuralEngine,
        directory,
    )
    .expect("the rotated MLP program loads");
    (projection, weight_count)
}

pub(crate) fn gate_verdict(has_passed: bool) -> &'static str {
    if has_passed { "PASS" } else { "STOP" }
}
