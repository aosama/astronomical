//! Input validation and request-scoped graph preparation for transformer execution.

use astronomical_runtime_integration::{MlxArray, MlxDtype};

use super::Flux2KleinTransformerError;
use super::blocks::{DoubleStreamState, ModulationSet};
use super::execution::{Flux2KleinForwardState, Flux2KleinTransformer, ForwardBlockState};
use super::math::{linear, rope_frequencies, timestep_embedding};

#[derive(Clone, Copy)]
pub struct Flux2KleinTransformerInputs<'a> {
    image_hidden_states: &'a MlxArray,
    text_hidden_states: &'a MlxArray,
    timestep: &'a MlxArray,
    image_position_ids: &'a MlxArray,
    text_position_ids: &'a MlxArray,
}

impl<'a> Flux2KleinTransformerInputs<'a> {
    pub const fn new(
        image_hidden_states: &'a MlxArray,
        text_hidden_states: &'a MlxArray,
        timestep: &'a MlxArray,
        image_position_ids: &'a MlxArray,
        text_position_ids: &'a MlxArray,
    ) -> Self {
        Self {
            image_hidden_states,
            text_hidden_states,
            timestep,
            image_position_ids,
            text_position_ids,
        }
    }
}

pub struct Flux2KleinComponentOracle {
    timestep_embedding: MlxArray,
    image_projection: MlxArray,
    text_projection: MlxArray,
    rope_cosines: MlxArray,
    rope_sines: MlxArray,
}

impl Flux2KleinComponentOracle {
    pub const fn timestep_embedding(&self) -> &MlxArray {
        &self.timestep_embedding
    }
    pub const fn image_projection(&self) -> &MlxArray {
        &self.image_projection
    }
    pub const fn text_projection(&self) -> &MlxArray {
        &self.text_projection
    }
    pub const fn rope_cosines(&self) -> &MlxArray {
        &self.rope_cosines
    }
    pub const fn rope_sines(&self) -> &MlxArray {
        &self.rope_sines
    }
}

impl Flux2KleinTransformer {
    pub fn component_oracle(
        &self,
        inputs: Flux2KleinTransformerInputs<'_>,
    ) -> Result<Flux2KleinComponentOracle, Flux2KleinTransformerError> {
        self.validate_inputs(inputs)?;
        let timestep_embedding =
            self.build_timestep_embedding(inputs.timestep, inputs.image_hidden_states.dtype())?;
        let image_projection = linear(
            &self.runtime,
            inputs.image_hidden_states,
            self.weights.tensor("x_embedder.weight")?,
        )?;
        let text_projection = linear(
            &self.runtime,
            inputs.text_hidden_states,
            self.weights.tensor("context_embedder.weight")?,
        )?;
        let (rope_cosines, rope_sines) =
            self.build_joint_rope(inputs.text_position_ids, inputs.image_position_ids)?;
        Ok(Flux2KleinComponentOracle {
            timestep_embedding,
            image_projection,
            text_projection,
            rope_cosines,
            rope_sines,
        })
    }

    pub fn start_forward(
        &self,
        inputs: Flux2KleinTransformerInputs<'_>,
    ) -> Result<Flux2KleinForwardState, Flux2KleinTransformerError> {
        self.validate_inputs(inputs)?;
        let text_token_count = inputs.text_hidden_states.shape()[1];
        let oracle = self.component_oracle(inputs)?;
        let modulation = self.build_modulation(&oracle.timestep_embedding)?;
        let Flux2KleinComponentOracle {
            timestep_embedding,
            image_projection,
            text_projection,
            rope_cosines,
            rope_sines,
        } = oracle;
        Ok(Flux2KleinForwardState {
            timestep_embedding,
            rope_cosines,
            rope_sines,
            modulation,
            block_state: ForwardBlockState::DoubleStream {
                state: DoubleStreamState {
                    image: image_projection,
                    text: text_projection,
                },
                next_block_index: 0,
            },
            text_token_count,
        })
    }

    fn build_timestep_embedding(
        &self,
        timestep: &MlxArray,
        activation_dtype: MlxDtype,
    ) -> Result<MlxArray, Flux2KleinTransformerError> {
        let sinusoidal = timestep_embedding(
            &self.runtime,
            timestep,
            self.geometry.timestep_embedding_width(),
        )?;
        let sinusoidal = self.runtime.astype(&sinusoidal, activation_dtype)?;
        let first = linear(
            &self.runtime,
            &sinusoidal,
            self.weights
                .tensor("time_guidance_embed.timestep_embedder.linear_1.weight")?,
        )?;
        linear(
            &self.runtime,
            &self.runtime.silu(&first)?,
            self.weights
                .tensor("time_guidance_embed.timestep_embedder.linear_2.weight")?,
        )
    }

    fn build_modulation(
        &self,
        timestep_embedding: &MlxArray,
    ) -> Result<ModulationSet, Flux2KleinTransformerError> {
        let activated = self.runtime.silu(timestep_embedding)?;
        Ok(ModulationSet {
            image_double: linear(
                &self.runtime,
                &activated,
                self.weights
                    .tensor("double_stream_modulation_img.linear.weight")?,
            )?,
            text_double: linear(
                &self.runtime,
                &activated,
                self.weights
                    .tensor("double_stream_modulation_txt.linear.weight")?,
            )?,
            single: linear(
                &self.runtime,
                &activated,
                self.weights
                    .tensor("single_stream_modulation.linear.weight")?,
            )?,
        })
    }

    fn build_joint_rope(
        &self,
        text_ids: &MlxArray,
        image_ids: &MlxArray,
    ) -> Result<(MlxArray, MlxArray), Flux2KleinTransformerError> {
        let (text_cos, text_sin) = rope_frequencies(&self.runtime, text_ids, &self.geometry)?;
        let (image_cos, image_sin) = rope_frequencies(&self.runtime, image_ids, &self.geometry)?;
        Ok((
            self.runtime.concatenate_axis(&[&text_cos, &image_cos], 0)?,
            self.runtime.concatenate_axis(&[&text_sin, &image_sin], 0)?,
        ))
    }

    fn validate_inputs(
        &self,
        inputs: Flux2KleinTransformerInputs<'_>,
    ) -> Result<(), Flux2KleinTransformerError> {
        let image_shape = inputs.image_hidden_states.shape();
        let text_shape = inputs.text_hidden_states.shape();
        if image_shape.len() != 3
            || text_shape.len() != 3
            || image_shape[0] <= 0
            || image_shape[1] <= 0
            || text_shape[1] <= 0
            || image_shape[0] != text_shape[0]
            || image_shape[2] != self.geometry.input_width() as i32
            || text_shape[2] != self.geometry.conditioning_width() as i32
            || inputs.timestep.shape() != [image_shape[0]]
            || inputs.image_position_ids.shape() != [image_shape[1], 4]
            || inputs.text_position_ids.shape() != [text_shape[1], 4]
            || inputs.image_hidden_states.dtype() != MlxDtype::BFloat16
            || inputs.text_hidden_states.dtype() != MlxDtype::BFloat16
        {
            return Err(Flux2KleinTransformerError::InvalidInput {
                description: "batch, token, width, position, or BF16 activation contract is invalid",
            });
        }
        Ok(())
    }
}
