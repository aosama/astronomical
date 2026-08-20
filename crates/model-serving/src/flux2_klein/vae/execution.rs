//! Request-scoped VAE advancement bounds complete-decoder graph construction.

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use crate::{PerformanceAttribution, PerformanceOperation};

use super::convolution::as_i32;
use super::decoder::BATCH_NORM_EPSILON;
use super::{
    FLUX2_KLEIN_PACKED_LATENT_CHANNEL_COUNT, Flux2KleinPackedLatentLayout, Flux2KleinVaeDecodeMode,
    Flux2KleinVaeDecoder, Flux2KleinVaeError,
};

#[derive(Debug)]
pub(in crate::flux2_klein) struct Flux2KleinVaeDecodeState {
    complete: CompleteDecodeState,
}

#[derive(Debug)]
pub(in crate::flux2_klein) enum Flux2KleinVaeDecodeAdvance {
    Decoding(Flux2KleinVaeDecodeState),
    PixelsReady(MlxArray),
}

#[derive(Debug)]
struct CompleteDecodeState {
    hidden_states: MlxArray,
    next_stage: CompleteDecodeStage,
}

#[derive(Clone, Copy, Debug)]
enum CompleteDecodeStage {
    Input,
    MiddleBeforeAttention,
    MiddleAttention,
    MiddleAfterAttention,
    UpBlock(usize),
    Output,
}

impl Flux2KleinVaeDecoder {
    pub(in crate::flux2_klein) fn start_decode_packed_latents_with_performance_attribution(
        &self,
        runtime: &MlxRuntime,
        packed_latents: &MlxArray,
        layout: Flux2KleinPackedLatentLayout,
        mode: Flux2KleinVaeDecodeMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Flux2KleinVaeDecodeState, Flux2KleinVaeError> {
        layout.validate_packed_shape(&packed_latents.shape())?;
        if matches!(mode, Flux2KleinVaeDecodeMode::Tiled(_)) {
            return Err(Flux2KleinVaeError::tiling_geometry(
                "independent VAE tiles are unavailable because they change global GroupNorm and middle-attention arithmetic; complete VAE decoding is required",
            ));
        }
        let unpatchified_latents = performance_attribution.measure_operation(
            PerformanceOperation::ImageVaeCompleteDecodeGraphConstruction,
            |_| self.restore_spatial_latents(runtime, packed_latents, layout),
        )?;
        performance_attribution.measure_operation(
            PerformanceOperation::ImageVaeDecodeSynchronizationWait,
            |_| unpatchified_latents.evaluate(),
        )?;
        Ok(Flux2KleinVaeDecodeState {
            complete: CompleteDecodeState {
                hidden_states: unpatchified_latents,
                next_stage: CompleteDecodeStage::Input,
            },
        })
    }

    pub(in crate::flux2_klein) fn advance_decode_with_performance_attribution(
        &self,
        runtime: &MlxRuntime,
        state: Flux2KleinVaeDecodeState,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Flux2KleinVaeDecodeAdvance, Flux2KleinVaeError> {
        match self.advance_complete_stage(
            runtime,
            state.complete,
            PerformanceOperation::ImageVaeCompleteDecodeGraphConstruction,
            PerformanceOperation::ImageVaeDecodeSynchronizationWait,
            performance_attribution,
        )? {
            CompleteDecodeAdvance::Decoding(next_state) => Ok(
                Flux2KleinVaeDecodeAdvance::Decoding(Flux2KleinVaeDecodeState {
                    complete: next_state,
                }),
            ),
            CompleteDecodeAdvance::PixelsReady(decoded_pixels) => {
                Ok(Flux2KleinVaeDecodeAdvance::PixelsReady(decoded_pixels))
            }
        }
    }

    fn restore_spatial_latents(
        &self,
        runtime: &MlxRuntime,
        packed_latents: &MlxArray,
        layout: Flux2KleinPackedLatentLayout,
    ) -> Result<MlxArray, Flux2KleinVaeError> {
        let packed_spatial_shape = to_i32_shape(layout.packed_spatial_shape())?;
        let packed_spatial = runtime.reshape(packed_latents, &packed_spatial_shape)?;
        let epsilon = runtime.full(
            &[as_i32(
                FLUX2_KLEIN_PACKED_LATENT_CHANNEL_COUNT,
                "BatchNorm channels",
            )?],
            BATCH_NORM_EPSILON,
            self.running_variance.dtype(),
        )?;
        let variance_with_epsilon = runtime.add(&self.running_variance, &epsilon)?;
        let standard_deviation = runtime.sqrt(&variance_with_epsilon)?;
        let scaled = runtime.multiply(&packed_spatial, &standard_deviation)?;
        let restored = runtime.add(&scaled, &self.running_mean)?;
        let shape = packed_spatial_shape;
        let patched = runtime.reshape(&restored, &[shape[0], shape[1], shape[2], 32, 2, 2])?;
        let spatial_order = runtime.transpose_axes(&patched, &[0, 1, 4, 2, 5, 3])?;
        Ok(runtime.reshape(&spatial_order, &to_i32_shape(layout.unpatchified_shape())?)?)
    }

    fn advance_complete_stage(
        &self,
        runtime: &MlxRuntime,
        state: CompleteDecodeState,
        graph_operation: PerformanceOperation,
        synchronization_operation: PerformanceOperation,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<CompleteDecodeAdvance, Flux2KleinVaeError> {
        let next_stage = state.next_stage;
        let hidden_states =
            performance_attribution.measure_operation(graph_operation, |_| match next_stage {
                CompleteDecodeStage::Input => {
                    let post_quant = self
                        .post_quant_conv
                        .forward(runtime, &state.hidden_states)?;
                    self.conv_in.forward(runtime, &post_quant)
                }
                CompleteDecodeStage::MiddleBeforeAttention => self
                    .middle_resnet_before_attention
                    .forward(runtime, &state.hidden_states),
                CompleteDecodeStage::MiddleAttention => {
                    self.middle_attention.forward(runtime, &state.hidden_states)
                }
                CompleteDecodeStage::MiddleAfterAttention => self
                    .middle_resnet_after_attention
                    .forward(runtime, &state.hidden_states),
                CompleteDecodeStage::UpBlock(block_index) => {
                    self.up_blocks[block_index].forward(runtime, &state.hidden_states)
                }
                CompleteDecodeStage::Output => {
                    let normalized = self.output_norm.forward(runtime, &state.hidden_states)?;
                    let activated = runtime.silu(&normalized)?;
                    let decoded = self.conv_out.forward(runtime, &activated)?;
                    Ok(runtime.clip(&decoded, -1.0, 1.0)?)
                }
            })?;
        performance_attribution
            .measure_operation(synchronization_operation, |_| hidden_states.evaluate())?;
        let following_stage = match next_stage {
            CompleteDecodeStage::Input => Some(CompleteDecodeStage::MiddleBeforeAttention),
            CompleteDecodeStage::MiddleBeforeAttention => {
                Some(CompleteDecodeStage::MiddleAttention)
            }
            CompleteDecodeStage::MiddleAttention => Some(CompleteDecodeStage::MiddleAfterAttention),
            CompleteDecodeStage::MiddleAfterAttention => Some(CompleteDecodeStage::UpBlock(0)),
            CompleteDecodeStage::UpBlock(block_index) if block_index + 1 < self.up_blocks.len() => {
                Some(CompleteDecodeStage::UpBlock(block_index + 1))
            }
            CompleteDecodeStage::UpBlock(_) => Some(CompleteDecodeStage::Output),
            CompleteDecodeStage::Output => None,
        };
        Ok(match following_stage {
            Some(next_stage) => CompleteDecodeAdvance::Decoding(CompleteDecodeState {
                hidden_states,
                next_stage,
            }),
            None => CompleteDecodeAdvance::PixelsReady(hidden_states),
        })
    }
}

enum CompleteDecodeAdvance {
    Decoding(CompleteDecodeState),
    PixelsReady(MlxArray),
}

fn to_i32_shape<const DIMENSIONS: usize>(
    shape: [usize; DIMENSIONS],
) -> Result<[i32; DIMENSIONS], Flux2KleinVaeError> {
    let mut converted = [0_i32; DIMENSIONS];
    for (dimension_index, dimension) in shape.into_iter().enumerate() {
        converted[dimension_index] = as_i32(dimension, "latent dimension")?;
    }
    Ok(converted)
}
