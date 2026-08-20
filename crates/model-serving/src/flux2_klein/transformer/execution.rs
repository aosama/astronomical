//! Engine-facing bounded transformer execution and component loading.

use std::time::Instant;

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use crate::{PerformanceAttribution, PerformanceOperation};

use super::blocks::{
    DoubleStreamState, ModulationSet, double_stream_block, single_stream_block, split_modulation,
};
use super::math::{fp32_layer_norm, linear};
use super::{
    Flux2KleinTransformerError, Flux2KleinTransformerGeometry, Flux2KleinTransformerInputs,
    Flux2KleinTransformerWeights,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Flux2KleinBlockKind {
    DoubleStream,
    SingleStream,
}

#[derive(Clone, Copy, Debug)]
pub enum Flux2KleinBlockGroupEvent {
    Started {
        kind: Flux2KleinBlockKind,
        first_block_index: usize,
        end_block_index: usize,
        started_at: Instant,
    },
    Completed {
        kind: Flux2KleinBlockKind,
        first_block_index: usize,
        end_block_index: usize,
        started_at: Instant,
        ended_at: Instant,
    },
}

pub struct Flux2KleinTransformerOutput {
    sample: MlxArray,
}
impl Flux2KleinTransformerOutput {
    pub const fn sample(&self) -> &MlxArray {
        &self.sample
    }
}

/// Request-scoped arrays retained between independently cancellable block groups.
pub struct Flux2KleinForwardState {
    pub(super) timestep_embedding: MlxArray,
    pub(super) rope_cosines: MlxArray,
    pub(super) rope_sines: MlxArray,
    pub(super) modulation: ModulationSet,
    pub(super) block_state: ForwardBlockState,
    pub(super) text_token_count: i32,
}

pub(super) enum ForwardBlockState {
    DoubleStream {
        state: DoubleStreamState,
        next_block_index: usize,
    },
    SingleStream {
        joint_states: MlxArray,
        next_block_index: usize,
    },
}

/// One bounded transformer advance, which either retains state or yields the final sample.
pub enum Flux2KleinForwardAdvance {
    BlockGroupCompleted(Flux2KleinForwardState),
    ForwardCompleted(Flux2KleinTransformerOutput),
}

impl Flux2KleinForwardAdvance {
    #[must_use]
    pub fn into_forward_state(self) -> Option<Flux2KleinForwardState> {
        match self {
            Self::BlockGroupCompleted(forward_state) => Some(forward_state),
            Self::ForwardCompleted(_) => None,
        }
    }

    #[must_use]
    pub fn into_output(self) -> Option<Flux2KleinTransformerOutput> {
        match self {
            Self::BlockGroupCompleted(_) => None,
            Self::ForwardCompleted(output) => Some(output),
        }
    }
}

#[derive(Debug)]
pub struct Flux2KleinTransformer {
    pub(super) runtime: MlxRuntime,
    pub(super) geometry: Flux2KleinTransformerGeometry,
    pub(super) weights: Flux2KleinTransformerWeights,
}

impl Flux2KleinTransformer {
    pub fn new(
        runtime: MlxRuntime,
        geometry: Flux2KleinTransformerGeometry,
        weights: Flux2KleinTransformerWeights,
    ) -> Result<Self, Flux2KleinTransformerError> {
        if weights.tensor_count() != geometry.expected_weight_shapes().count() {
            return Err(Flux2KleinTransformerError::InvalidInput {
                description: "bound tensor count disagrees with transformer geometry",
            });
        }
        weights.materialize_owned(&runtime)?;
        Ok(Self {
            runtime,
            geometry,
            weights,
        })
    }

    pub(crate) fn load_with_geometry_and_performance_attribution(
        runtime: MlxRuntime,
        weights_file: crate::ValidatedWeightsFile,
        geometry: Flux2KleinTransformerGeometry,
        retained_block_indices: &[usize],
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, Flux2KleinTransformerError> {
        let weights = performance_attribution.measure_operation(
            PerformanceOperation::ImageTransformerComponentMapping,
            |_| {
                Flux2KleinTransformerWeights::load_with_residency(
                    &runtime,
                    weights_file,
                    &geometry,
                    retained_block_indices,
                )
            },
        )?;
        performance_attribution.measure_operation(
            PerformanceOperation::ImageTransformerComponentLoading,
            |_| Self::new(runtime, geometry, weights),
        )
    }

    pub const fn runtime(&self) -> &MlxRuntime {
        &self.runtime
    }
    pub const fn geometry(&self) -> &Flux2KleinTransformerGeometry {
        &self.geometry
    }
    pub const fn weights(&self) -> &Flux2KleinTransformerWeights {
        &self.weights
    }

    /// Returns runtime ownership after the evaluated denoising state no longer depends on weights.
    pub(crate) fn into_runtime(self) -> MlxRuntime {
        self.runtime
    }

    pub fn forward_in_block_groups(
        &self,
        inputs: Flux2KleinTransformerInputs<'_>,
        maximum_blocks_per_group: usize,
        is_cancelled: &mut dyn FnMut() -> bool,
        record_event: &mut dyn FnMut(Flux2KleinBlockGroupEvent),
    ) -> Result<Flux2KleinTransformerOutput, Flux2KleinTransformerError> {
        let mut performance_attribution = PerformanceAttribution::disabled();
        self.forward_in_block_groups_with_performance_attribution(
            inputs,
            maximum_blocks_per_group,
            is_cancelled,
            record_event,
            &mut performance_attribution,
        )
    }

    pub(crate) fn forward_in_block_groups_with_performance_attribution(
        &self,
        inputs: Flux2KleinTransformerInputs<'_>,
        maximum_blocks_per_group: usize,
        is_cancelled: &mut dyn FnMut() -> bool,
        record_event: &mut dyn FnMut(Flux2KleinBlockGroupEvent),
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Flux2KleinTransformerOutput, Flux2KleinTransformerError> {
        let mut forward_state = self.start_forward(inputs)?;
        loop {
            if is_cancelled() {
                return Err(Flux2KleinTransformerError::Cancelled);
            }
            match self.advance_one_block_group_with_performance_attribution(
                forward_state,
                maximum_blocks_per_group,
                record_event,
                performance_attribution,
            )? {
                Flux2KleinForwardAdvance::BlockGroupCompleted(state) => forward_state = state,
                Flux2KleinForwardAdvance::ForwardCompleted(output) => return Ok(output),
            }
        }
    }

    pub fn advance_one_block_group(
        &self,
        forward_state: Flux2KleinForwardState,
        maximum_blocks_per_group: usize,
        record_event: &mut dyn FnMut(Flux2KleinBlockGroupEvent),
    ) -> Result<Flux2KleinForwardAdvance, Flux2KleinTransformerError> {
        let mut performance_attribution = PerformanceAttribution::disabled();
        self.advance_one_block_group_with_performance_attribution(
            forward_state,
            maximum_blocks_per_group,
            record_event,
            &mut performance_attribution,
        )
    }

    pub(crate) fn advance_one_block_group_with_performance_attribution(
        &self,
        mut forward_state: Flux2KleinForwardState,
        maximum_blocks_per_group: usize,
        record_event: &mut dyn FnMut(Flux2KleinBlockGroupEvent),
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Flux2KleinForwardAdvance, Flux2KleinTransformerError> {
        if maximum_blocks_per_group == 0 {
            return Err(Flux2KleinTransformerError::ZeroBlockGroupSize);
        }
        forward_state.block_state = match forward_state.block_state {
            ForwardBlockState::DoubleStream {
                mut state,
                next_block_index,
            } => {
                let end_block_index = next_block_index
                    .saturating_add(maximum_blocks_per_group)
                    .min(self.geometry.double_stream_block_count());
                let started_at = Instant::now();
                record_event(Flux2KleinBlockGroupEvent::Started {
                    kind: Flux2KleinBlockKind::DoubleStream,
                    first_block_index: next_block_index,
                    end_block_index,
                    started_at,
                });
                let mut group_streamed_weights = false;
                for block_index in next_block_index..end_block_index {
                    let block_weights = self.weights.bind_block(&self.runtime, block_index)?;
                    group_streamed_weights |= block_weights.is_operation_local();
                    state = performance_attribution.measure_operation(
                        PerformanceOperation::ImageTransformerBlockGroupGraphConstruction,
                        |_| {
                            double_stream_block(
                                &self.runtime,
                                &self.geometry,
                                &block_weights,
                                block_index,
                                state,
                                &forward_state.modulation,
                                &forward_state.rope_cosines,
                                &forward_state.rope_sines,
                            )
                        },
                    )?;
                    if block_weights.is_operation_local() {
                        performance_attribution.measure_operation(
                            PerformanceOperation::ImageTransformerBlockGroupSynchronizationWait,
                            |_| self.runtime.evaluate_arrays(&[&state.image, &state.text]),
                        )?;
                    }
                }
                if !group_streamed_weights {
                    performance_attribution.measure_operation(
                        PerformanceOperation::ImageTransformerBlockGroupSynchronizationWait,
                        |_| self.runtime.evaluate_arrays(&[&state.image, &state.text]),
                    )?;
                }
                if group_streamed_weights {
                    self.runtime.clear_allocator_cache()?;
                }
                record_event(Flux2KleinBlockGroupEvent::Completed {
                    kind: Flux2KleinBlockKind::DoubleStream,
                    first_block_index: next_block_index,
                    end_block_index,
                    started_at,
                    ended_at: Instant::now(),
                });
                if end_block_index == self.geometry.double_stream_block_count() {
                    ForwardBlockState::SingleStream {
                        joint_states: self
                            .runtime
                            .concatenate_axis(&[&state.text, &state.image], 1)?,
                        next_block_index: 0,
                    }
                } else {
                    ForwardBlockState::DoubleStream {
                        state,
                        next_block_index: end_block_index,
                    }
                }
            }
            ForwardBlockState::SingleStream {
                mut joint_states,
                next_block_index,
            } => {
                let end_block_index = next_block_index
                    .saturating_add(maximum_blocks_per_group)
                    .min(self.geometry.single_stream_block_count());
                let started_at = Instant::now();
                record_event(Flux2KleinBlockGroupEvent::Started {
                    kind: Flux2KleinBlockKind::SingleStream,
                    first_block_index: next_block_index,
                    end_block_index,
                    started_at,
                });
                let mut group_streamed_weights = false;
                for block_index in next_block_index..end_block_index {
                    let combined_block_index =
                        self.geometry.double_stream_block_count() + block_index;
                    let block_weights = self
                        .weights
                        .bind_block(&self.runtime, combined_block_index)?;
                    group_streamed_weights |= block_weights.is_operation_local();
                    joint_states = performance_attribution.measure_operation(
                        PerformanceOperation::ImageTransformerBlockGroupGraphConstruction,
                        |_| {
                            single_stream_block(
                                &self.runtime,
                                &self.geometry,
                                &block_weights,
                                block_index,
                                &joint_states,
                                &forward_state.modulation,
                                &forward_state.rope_cosines,
                                &forward_state.rope_sines,
                            )
                        },
                    )?;
                    if block_weights.is_operation_local() {
                        performance_attribution.measure_operation(
                            PerformanceOperation::ImageTransformerBlockGroupSynchronizationWait,
                            |_| self.runtime.evaluate_arrays(&[&joint_states]),
                        )?;
                    }
                }
                if !group_streamed_weights {
                    performance_attribution.measure_operation(
                        PerformanceOperation::ImageTransformerBlockGroupSynchronizationWait,
                        |_| self.runtime.evaluate_arrays(&[&joint_states]),
                    )?;
                }
                if group_streamed_weights {
                    self.runtime.clear_allocator_cache()?;
                }
                record_event(Flux2KleinBlockGroupEvent::Completed {
                    kind: Flux2KleinBlockKind::SingleStream,
                    first_block_index: next_block_index,
                    end_block_index,
                    started_at,
                    ended_at: Instant::now(),
                });
                if end_block_index < self.geometry.single_stream_block_count() {
                    forward_state.block_state = ForwardBlockState::SingleStream {
                        joint_states,
                        next_block_index: end_block_index,
                    };
                    return Ok(Flux2KleinForwardAdvance::BlockGroupCompleted(forward_state));
                }
                return self
                    .complete_forward(
                        forward_state.timestep_embedding,
                        forward_state.text_token_count,
                        joint_states,
                    )
                    .map(Flux2KleinForwardAdvance::ForwardCompleted);
            }
        };
        Ok(Flux2KleinForwardAdvance::BlockGroupCompleted(forward_state))
    }

    fn complete_forward(
        &self,
        timestep_embedding: MlxArray,
        text_token_count: i32,
        joint_states: MlxArray,
    ) -> Result<Flux2KleinTransformerOutput, Flux2KleinTransformerError> {
        let image_states = self.runtime.slice(
            &joint_states,
            &[0, text_token_count, 0],
            &[
                joint_states.shape()[0],
                joint_states.shape()[1],
                self.geometry.hidden_width() as i32,
            ],
            &[1, 1, 1],
        )?;
        let normalized = fp32_layer_norm(
            &self.runtime,
            &image_states,
            self.geometry.normalization_epsilon(),
        )?;
        let final_modulation = linear(
            &self.runtime,
            &self.runtime.silu(&timestep_embedding)?,
            self.weights.tensor("norm_out.linear.weight")?,
        )?;
        let final_parts = split_modulation(
            &self.runtime,
            &final_modulation,
            self.geometry.hidden_width(),
            2,
        )?;
        let one_plus_scale = self.runtime.add(
            &self.runtime.full(&[], 1.0, final_parts[0].dtype())?,
            &final_parts[0],
        )?;
        let adaptive_normalized = self.runtime.add(
            &self.runtime.multiply(&normalized, &one_plus_scale)?,
            &final_parts[1],
        )?;
        let sample = linear(
            &self.runtime,
            &adaptive_normalized,
            self.weights.tensor("proj_out.weight")?,
        )?;
        Ok(Flux2KleinTransformerOutput { sample })
    }
}
