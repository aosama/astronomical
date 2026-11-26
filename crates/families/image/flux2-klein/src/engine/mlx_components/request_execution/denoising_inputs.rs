//! The request state one denoising step reads, borrowed field by field from the component owner
//! so the owner can still update its other fields while the step runs.

use super::*;
use crate::transformer::Flux2KleinTransformerOutput;

/// Shared borrows of the transformer and the request arrays that a denoising step reads.
pub(super) struct DenoisingInputs<'owner> {
    transformer: &'owner Flux2KleinTransformer,
    latents: &'owner MlxArray,
    conditioning: &'owner Flux2KleinTextConditioning,
    image_position_ids: &'owner MlxArray,
    text_position_ids: &'owner MlxArray,
}

impl<'owner> DenoisingInputs<'owner> {
    /// Binds every input a step needs, failing with the first one that is missing, in the order
    /// the arguments are listed.
    pub(super) fn bind(
        transformer: Option<&'owner Flux2KleinTransformer>,
        latents: Option<&'owner MlxArray>,
        conditioning: Option<&'owner Flux2KleinTextConditioning>,
        image_position_ids: Option<&'owner MlxArray>,
        text_position_ids: Option<&'owner MlxArray>,
    ) -> Result<Self, String> {
        Ok(Self {
            transformer: transformer.ok_or_else(|| "transformer is unavailable".to_owned())?,
            latents: latents.ok_or_else(|| "latent state is unavailable".to_owned())?,
            conditioning: conditioning
                .ok_or_else(|| "text conditioning is unavailable".to_owned())?,
            image_position_ids: image_position_ids
                .ok_or_else(|| "image position IDs are unavailable".to_owned())?,
            text_position_ids: text_position_ids
                .ok_or_else(|| "text position IDs are unavailable".to_owned())?,
        })
    }

    /// Starts the forward pass of one denoising step. The step index is recorded after the
    /// timestep array exists and before the transformer sees any input.
    pub(super) fn start_forward_state(
        &self,
        flow_step: &Flux2KleinFlowStep,
        step_index: usize,
        forward_step_index: &mut Option<usize>,
    ) -> Result<Flux2KleinForwardState, String> {
        let timestep = self
            .transformer
            .runtime()
            .array_from_f32(&[flow_step.sigma() as f32], &[1])
            .map_err(|error| error.to_string())?;
        *forward_step_index = Some(step_index);
        self.transformer
            .start_forward(Flux2KleinTransformerInputs::new(
                self.latents,
                self.conditioning.hidden_states(),
                &timestep,
                self.image_position_ids,
                self.text_position_ids,
            ))
            .map_err(|error| error.to_string())
    }

    /// Advances the forward pass by one block group, so cancellation can land between groups.
    pub(super) fn advance_block_group(
        &self,
        forward_state: Flux2KleinForwardState,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Flux2KleinForwardAdvance, String> {
        let mut ignore_group_event = |_| {};
        self.transformer
            .advance_one_block_group_with_performance_attribution(
                forward_state,
                TRANSFORMER_BLOCKS_PER_CANCELLATION_GROUP,
                &mut ignore_group_event,
                performance_attribution,
            )
            .map_err(|error| error.to_string())
    }

    /// Builds the Euler update of the latents from the finished forward pass and waits for it.
    pub(super) fn euler_updated_latents(
        &self,
        output: &Flux2KleinTransformerOutput,
        flow_step: &Flux2KleinFlowStep,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, String> {
        let updated_latents = performance_attribution.measure_operation(
            PerformanceOperation::ImageSchedulerUpdateGraphConstruction,
            |_| {
                flux2_klein_euler_update(
                    self.transformer.runtime(),
                    self.latents,
                    output.sample(),
                    flow_step.delta_sigma() as f32,
                )
                .map_err(|error| error.to_string())
            },
        )?;
        performance_attribution.measure_operation(
            PerformanceOperation::ImageSchedulerUpdateSynchronizationWait,
            |_| {
                self.transformer
                    .runtime()
                    .evaluate_arrays(&[&updated_latents])
                    .map_err(|error| error.to_string())
            },
        )?;
        Ok(updated_latents)
    }
}

#[cfg(test)]
#[path = "denoising_inputs_tests.rs"]
mod denoising_inputs_tests;
