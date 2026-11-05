//! A projection's inputs and its fp32 reference result.

use astronomical_neural_engine::program_shape::ProgramShape;
use astronomical_neural_engine::weights::half_precision::{
    f16_bits_slice_to_f32, f32_slice_to_f16_bits,
};
use astronomical_neural_engine_research::accelerate_reference::reference_projection;
use astronomical_test_harness::deterministic_values::{
    PROJECTION_WEIGHT_STANDARD_DEVIATION, bell_shaped_values,
};

use crate::support::journey_constants::PREFILL_CHUNK_TOKEN_COUNT;

const MIXER_OUTPUT_INPUT_CHANNELS: usize = 4096;
const MIXER_OUTPUT_OUTPUT_CHANNELS: usize = 2048;
const WEIGHT_SEED: u64 = 0x5eed_0161;
const ACTIVATION_SEED: u64 = 0x5eed_0162;

pub struct ProjectionFixture {
    pub shape: ProgramShape,
    pub fp16_weights: Vec<u16>,
    pub fp16_activations: Vec<u16>,
    /// Accelerate's fp32 product of the fp16-rounded inputs, so the measured
    /// error is the engine's arithmetic alone, not the input narrowing.
    pub reference_projected: Vec<f32>,
}

impl ProjectionFixture {
    pub fn mixer_output_projection() -> Self {
        Self::with_shape(
            MIXER_OUTPUT_INPUT_CHANNELS,
            MIXER_OUTPUT_OUTPUT_CHANNELS,
            PREFILL_CHUNK_TOKEN_COUNT,
        )
    }

    pub fn with_shape(input_channels: usize, output_channels: usize, token_count: usize) -> Self {
        let shape = ProgramShape::new(input_channels, output_channels, token_count)
            .expect("the fixture shape is valid");
        let fp16_weights = f32_slice_to_f16_bits(&bell_shaped_values(
            WEIGHT_SEED,
            shape.weight_element_count(),
            PROJECTION_WEIGHT_STANDARD_DEVIATION,
        ));
        Self::with_fp16_weights(shape, fp16_weights)
    }

    /// Fresh activations and the reference for caller-chosen weights, such as
    /// weights rounded through a quantized format first.
    pub fn with_fp16_weights(shape: ProgramShape, fp16_weights: Vec<u16>) -> Self {
        let fp16_activations = f32_slice_to_f16_bits(&bell_shaped_values(
            ACTIVATION_SEED,
            shape.activation_element_count(),
            1.0,
        ));
        let reference_projected = reference_projection(
            shape,
            &f16_bits_slice_to_f32(&fp16_weights),
            &f16_bits_slice_to_f32(&fp16_activations),
        )
        .expect("the reference projection runs");
        Self {
            shape,
            fp16_weights,
            fp16_activations,
            reference_projected,
        }
    }

    /// The same projection restricted to output channels
    /// `first_output_channel .. first_output_channel + output_channel_count`.
    /// Output channels are independent, so a slice's reference is exactly the
    /// matching rows of the full reference. Only the GPU split journeys use
    /// slices, so the method exists only where they compile.
    pub fn output_channel_slice(
        &self,
        first_output_channel: usize,
        output_channel_count: usize,
    ) -> Self {
        let shape = ProgramShape::new(
            self.shape.input_channels(),
            output_channel_count,
            self.shape.token_count(),
        )
        .expect("the slice shape is valid");
        let weight_row_width = self.shape.input_channels();
        let projected_row_width = self.shape.token_count();
        let slice_end = first_output_channel + output_channel_count;
        Self {
            shape,
            fp16_weights: self.fp16_weights
                [first_output_channel * weight_row_width..slice_end * weight_row_width]
                .to_vec(),
            fp16_activations: self.fp16_activations.clone(),
            reference_projected: self.reference_projected
                [first_output_channel * projected_row_width..slice_end * projected_row_width]
                .to_vec(),
        }
    }
}
