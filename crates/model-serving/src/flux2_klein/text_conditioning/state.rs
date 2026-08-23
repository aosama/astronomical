//! Request-scoped Qwen layer state that yields after one bounded layer group.

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime};

use crate::{PerformanceAttribution, PerformanceOperation};

use super::batch::{FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH, Flux2KleinPreparedTextBatch};
use super::error::Flux2KleinTextConditioningError;
use super::layer::{
    build_causal_padding_mask, forward_decoder_layer_attention, forward_decoder_layer_feed_forward,
};
use super::weights::{EXECUTED_LAYER_COUNT, Flux2KleinTextWeights, HIDDEN_WIDTH};

const CONDITIONING_WIDTH: i32 = 7_680;
const HIDDEN_STATE_TAPS: [usize; 3] = [9, 18, 27];

pub(crate) struct Flux2KleinTextConditioningState {
    weights: Flux2KleinTextWeights,
    hidden_states: MlxArray,
    attention_mask: MlxArray,
    combined_attention_mask: MlxArray,
    captured_hidden_states: Vec<MlxArray>,
    batch_size: usize,
    signed_batch_size: i32,
    signed_sequence_length: i32,
    next_layer_index: usize,
}

pub(crate) enum Flux2KleinTextConditioningAdvance {
    LayerGroupCompleted(Flux2KleinTextConditioningState),
    ConditioningCompleted(Flux2KleinTextConditioning),
}

pub(crate) struct Flux2KleinTextConditioning {
    hidden_states: MlxArray,
    attention_mask: MlxArray,
    batch_size: usize,
}

impl Flux2KleinTextConditioningState {
    pub(super) fn initialize(
        runtime: &MlxRuntime,
        prepared_batch: Flux2KleinPreparedTextBatch,
        mut weights: Flux2KleinTextWeights,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, Flux2KleinTextConditioningError> {
        let signed_batch_size = i32::try_from(prepared_batch.batch_size())
            .map_err(|_| Flux2KleinTextConditioningError::BatchGeometryOverflow)?;
        let signed_sequence_length = i32::try_from(prepared_batch.sequence_length())
            .map_err(|_| Flux2KleinTextConditioningError::BatchGeometryOverflow)?;
        let token_ids = runtime.array_from_u32(
            prepared_batch.token_ids(),
            &[signed_batch_size, signed_sequence_length],
        )?;
        let attention_mask_integers = runtime.array_from_u32(
            prepared_batch.attention_mask(),
            &[signed_batch_size, signed_sequence_length],
        )?;
        let attention_mask = runtime.astype(&attention_mask_integers, MlxDtype::Bool)?;
        let combined_attention_mask = performance_attribution.measure_operation(
            PerformanceOperation::ImageTextComponentLoading,
            |_| {
                build_causal_padding_mask(
                    runtime,
                    prepared_batch.attention_mask(),
                    signed_batch_size,
                    signed_sequence_length,
                )
            },
        )?;
        let embedding = weights
            .embedding
            .take()
            .ok_or(Flux2KleinTextConditioningError::WeightsUnavailable)?;
        let hidden_states = runtime.take_axis(&embedding, &token_ids, 0)?;
        performance_attribution
            .measure_operation(PerformanceOperation::ImageTextComponentLoading, |_| {
                runtime.evaluate_arrays(&[&hidden_states])
            })?;
        drop(embedding);
        if weights.is_streamed() {
            performance_attribution
                .measure_operation(PerformanceOperation::MlxAllocatorCacheCleanup, |_| {
                    runtime.clear_allocator_cache()
                })?;
        }
        Ok(Self {
            weights,
            hidden_states,
            attention_mask,
            combined_attention_mask,
            captured_hidden_states: Vec::with_capacity(HIDDEN_STATE_TAPS.len()),
            batch_size: prepared_batch.batch_size(),
            signed_batch_size,
            signed_sequence_length,
            next_layer_index: 0,
        })
    }

    pub(crate) fn advance_layer_group(
        mut self,
        runtime: &MlxRuntime,
        maximum_layer_count: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Flux2KleinTextConditioningAdvance, Flux2KleinTextConditioningError> {
        if maximum_layer_count == 0 {
            return Err(Flux2KleinTextConditioningError::EmptyLayerGroup);
        }
        let layer_group_end = self
            .next_layer_index
            .saturating_add(maximum_layer_count)
            .min(EXECUTED_LAYER_COUNT);
        while self.next_layer_index < layer_group_end {
            self.advance_one_layer(runtime, performance_attribution)?;
        }
        if self.next_layer_index < EXECUTED_LAYER_COUNT {
            return Ok(Flux2KleinTextConditioningAdvance::LayerGroupCompleted(self));
        }
        self.finish(runtime, performance_attribution)
            .map(Flux2KleinTextConditioningAdvance::ConditioningCompleted)
    }

    fn advance_one_layer(
        &mut self,
        runtime: &MlxRuntime,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Flux2KleinTextConditioningError> {
        let layer_weights =
            self.weights
                .take_layer(runtime, self.next_layer_index, performance_attribution)?;
        let next_hidden_states = performance_attribution.measure_operation(
            PerformanceOperation::ImageQwenLayerGraphConstruction,
            |_| {
                let attention_output = forward_decoder_layer_attention(
                    runtime,
                    &self.hidden_states,
                    &self.combined_attention_mask,
                    self.signed_batch_size,
                    self.signed_sequence_length,
                    &layer_weights,
                )?;
                forward_decoder_layer_feed_forward(runtime, &attention_output, &layer_weights)
            },
        )?;
        performance_attribution.measure_operation(
            PerformanceOperation::ImageQwenLayerSynchronizationWait,
            |_| runtime.evaluate_arrays(&[&next_hidden_states]),
        )?;
        self.next_layer_index += 1;
        self.hidden_states = next_hidden_states;
        if HIDDEN_STATE_TAPS.contains(&self.next_layer_index) {
            self.captured_hidden_states
                .push(self.hidden_states.retain()?);
        }
        let is_streamed = self.weights.is_streamed();
        self.weights.release_layer(layer_weights);
        if is_streamed {
            performance_attribution
                .measure_operation(PerformanceOperation::MlxAllocatorCacheCleanup, |_| {
                    runtime.clear_allocator_cache()
                })?;
        }
        Ok(())
    }

    fn finish(
        mut self,
        runtime: &MlxRuntime,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Flux2KleinTextConditioning, Flux2KleinTextConditioningError> {
        self.weights.release_descriptor_source()?;
        let captured_references = self.captured_hidden_states.iter().collect::<Vec<_>>();
        let conditioning_hidden_states = runtime.concatenate_axis(&captured_references, 2)?;
        performance_attribution
            .measure_operation(PerformanceOperation::ImageTextComponentLoading, |_| {
                runtime.evaluate_arrays(&[&conditioning_hidden_states, &self.attention_mask])
            })?;
        validate_conditioning_output(
            &conditioning_hidden_states,
            self.signed_batch_size,
            self.signed_sequence_length,
        )?;
        Ok(Flux2KleinTextConditioning {
            hidden_states: conditioning_hidden_states,
            attention_mask: self.attention_mask,
            batch_size: self.batch_size,
        })
    }
}

impl Flux2KleinTextConditioning {
    pub(crate) const fn hidden_states(&self) -> &MlxArray {
        &self.hidden_states
    }
    pub(crate) const fn attention_mask(&self) -> &MlxArray {
        &self.attention_mask
    }
    pub(crate) const fn batch_size(&self) -> usize {
        self.batch_size
    }
    pub(crate) const fn sequence_length(&self) -> usize {
        FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH
    }
}

fn validate_conditioning_output(
    hidden_states: &MlxArray,
    batch_size: i32,
    sequence_length: i32,
) -> Result<(), Flux2KleinTextConditioningError> {
    if hidden_states.dtype() != MlxDtype::BFloat16
        || hidden_states.shape() != [batch_size, sequence_length, CONDITIONING_WIDTH]
    {
        return Err(Flux2KleinTextConditioningError::InvalidTensor {
            tensor_name: "captured_hidden_states".to_owned(),
            description: "output must be BF16 with shape [batch, 512, 7680]",
        });
    }
    if CONDITIONING_WIDTH != HIDDEN_WIDTH * 3 {
        return Err(Flux2KleinTextConditioningError::InvalidTensor {
            tensor_name: "captured_hidden_states".to_owned(),
            description: "tap concatenation width disagrees with the official profile",
        });
    }
    Ok(())
}
