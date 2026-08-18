//! Per-layer Laguna decoder cache: append-only full attention or rotating sliding.

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use crate::attention::rotating_prefill_transient_token_count;
use crate::decoder_cache::{
    FullAttentionKeyValueState, FullAttentionKeyValueStateAllocationCheckpoint,
    RotatingKeyValueState, RotatingKeyValueStateAllocationCheckpoint,
};
use crate::laguna::normalization::{
    LagunaCacheDescriptor, LagunaExecutionDtype, LagunaTargetContract,
};
use crate::performance_attribution::PerformanceAttribution;

use super::error::LagunaExecutionError;

#[path = "decoder_cache_bridge.rs"]
mod decoder_cache_bridge;

/// One request's Laguna decoder cache, one entry per layer descriptor.
pub struct LagunaDecoderState {
    layers: Vec<LagunaLayerCacheState>,
}

enum LagunaLayerCacheState {
    AppendOnly(FullAttentionKeyValueState),
    Rotating(RotatingKeyValueState),
}

/// Retained decoder ownership needed to roll back one failed Laguna allocation attempt.
pub struct LagunaDecoderStateAllocationCheckpoint {
    layers: Vec<LagunaLayerCacheAllocationCheckpoint>,
}

/// Exact persistent and temporary byte owners required by one decoder forward.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LagunaDecoderForwardMemoryProjection {
    persistent_growth_bytes: usize,
    sliding_temporary_workspace_bytes: usize,
}

impl LagunaDecoderForwardMemoryProjection {
    #[must_use]
    pub const fn persistent_growth_bytes(self) -> usize {
        self.persistent_growth_bytes
    }

    #[must_use]
    pub const fn sliding_temporary_workspace_bytes(self) -> usize {
        self.sliding_temporary_workspace_bytes
    }
}

enum LagunaLayerCacheAllocationCheckpoint {
    AppendOnly(FullAttentionKeyValueStateAllocationCheckpoint),
    Rotating(RotatingKeyValueStateAllocationCheckpoint),
}

impl LagunaDecoderState {
    /// Allocates empty cache state from the ordered layer descriptors.
    pub fn empty(contract: &LagunaTargetContract) -> Result<Self, LagunaExecutionError> {
        let mut layers = Vec::with_capacity(contract.layers().len());
        for layer_descriptor in contract.layers() {
            layers.push(match *layer_descriptor.attention().cache() {
                LagunaCacheDescriptor::AppendOnly => LagunaLayerCacheState::AppendOnly(
                    FullAttentionKeyValueState::empty_with_growth_tokens(256)?,
                ),
                LagunaCacheDescriptor::Rotating { window_size } => LagunaLayerCacheState::Rotating(
                    RotatingKeyValueState::empty(i32::try_from(window_size).map_err(|_| {
                        LagunaExecutionError::invalid_geometry(
                            "rotating window exceeds the i32 range",
                        )
                    })?)?,
                ),
            });
        }
        Ok(Self { layers })
    }

    pub(super) fn update_and_fetch(
        &mut self,
        runtime: &MlxRuntime,
        layer_index: usize,
        keys: &MlxArray,
        values: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(MlxArray, MlxArray, i32), LagunaExecutionError> {
        let layer_state = self.layers.get_mut(layer_index).ok_or_else(|| {
            LagunaExecutionError::invalid_geometry("decoder state is missing a layer")
        })?;
        match layer_state {
            LagunaLayerCacheState::AppendOnly(state) => {
                let offset_tokens = state.offset_tokens();
                let (active_keys, active_values) =
                    state.update_and_fetch(runtime, keys, values, offset_tokens)?;
                Ok((active_keys, active_values, offset_tokens))
            }
            LagunaLayerCacheState::Rotating(state) => {
                let offset_tokens = state.absolute_position();
                let (active_keys, active_values) =
                    state.update_and_fetch(runtime, keys, values, performance_attribution)?;
                Ok((active_keys, active_values, offset_tokens))
            }
        }
    }

    /// Returns the absolute token count stored by one layer.
    #[must_use]
    pub fn absolute_position(&self, layer_index: usize) -> Option<i32> {
        match self.layers.get(layer_index)? {
            LagunaLayerCacheState::AppendOnly(state) => Some(state.offset_tokens()),
            LagunaLayerCacheState::Rotating(state) => Some(state.absolute_position()),
        }
    }

    /// Returns the next physical write slot for a rotating layer.
    #[must_use]
    pub fn ring_write_index(&self, layer_index: usize) -> Option<i32> {
        match self.layers.get(layer_index)? {
            LagunaLayerCacheState::AppendOnly(_) => None,
            LagunaLayerCacheState::Rotating(state) => Some(state.ring_write_index()),
        }
    }

    /// Returns logical payload bytes owned by every live attention cache slab.
    #[must_use]
    pub fn payload_byte_count(&self) -> u64 {
        self.layers
            .iter()
            .map(|layer_state| match layer_state {
                LagunaLayerCacheState::AppendOnly(state) => state.payload_byte_count(),
                LagunaLayerCacheState::Rotating(state) => state.payload_byte_count(),
            })
            .fold(0, u64::saturating_add)
    }

    /// Projects exact physical persistent growth for one concrete forward.
    pub fn projected_persistent_growth_bytes(
        &self,
        contract: &LagunaTargetContract,
        forward_token_count: usize,
    ) -> Result<usize, LagunaExecutionError> {
        Ok(self
            .projected_forward_memory(contract, forward_token_count)?
            .persistent_growth_bytes())
    }

    /// Projects descriptor-derived persistent growth and rotating chronological workspace.
    pub fn projected_forward_memory(
        &self,
        contract: &LagunaTargetContract,
        forward_token_count: usize,
    ) -> Result<LagunaDecoderForwardMemoryProjection, LagunaExecutionError> {
        if self.layers.len() != contract.layers().len() {
            return Err(LagunaExecutionError::invalid_geometry(
                "Laguna decoder state and contract layer counts differ",
            ));
        }
        let scalar_byte_count = match contract.model().execution_dtype() {
            LagunaExecutionDtype::Float16 | LagunaExecutionDtype::Bfloat16 => 2_usize,
            LagunaExecutionDtype::Float32 => 4,
        };
        self.layers.iter().zip(contract.layers()).try_fold(
            LagunaDecoderForwardMemoryProjection {
                persistent_growth_bytes: 0,
                sliding_temporary_workspace_bytes: 0,
            },
            |projection, (layer_state, layer_descriptor)| {
                let attention_descriptor = layer_descriptor.attention();
                let key_value_bytes_per_token =
                    usize::try_from(attention_descriptor.key_value_head_count())
                        .ok()
                        .and_then(|head_count| {
                            head_count.checked_mul(
                                usize::try_from(attention_descriptor.head_dimension()).ok()?,
                            )
                        })
                        .and_then(|elements| elements.checked_mul(scalar_byte_count))
                        .and_then(|one_tensor_bytes| one_tensor_bytes.checked_mul(2))
                        .ok_or_else(|| {
                            LagunaExecutionError::invalid_geometry(
                                "Laguna key/value bytes per token overflowed",
                            )
                        })?;
                let growth_token_count = match layer_state {
                    LagunaLayerCacheState::AppendOnly(state) => {
                        state.projected_capacity_growth_tokens(forward_token_count)?
                    }
                    LagunaLayerCacheState::Rotating(state) => {
                        let current_committed_token_count =
                            usize::try_from(state.committed_token_count()).unwrap_or(usize::MAX);
                        let window_token_count =
                            usize::try_from(state.window_size()).unwrap_or(usize::MAX);
                        current_committed_token_count
                            .checked_add(forward_token_count)
                            .map(|next_token_count| next_token_count.min(window_token_count))
                            .unwrap_or(window_token_count)
                            .saturating_sub(current_committed_token_count)
                    }
                };
                let sliding_temporary_token_count = match layer_state {
                    LagunaLayerCacheState::AppendOnly(_) => 0,
                    LagunaLayerCacheState::Rotating(state) if forward_token_count > 1 => {
                        let window_token_count =
                            u32::try_from(state.window_size()).map_err(|_| {
                                LagunaExecutionError::invalid_geometry(
                                    "Laguna rotating window exceeds the u32 range",
                                )
                            })?;
                        let chunk_token_count =
                            u32::try_from(forward_token_count).map_err(|_| {
                                LagunaExecutionError::invalid_geometry(
                                    "Laguna forward token count exceeds the u32 range",
                                )
                            })?;
                        usize::try_from(
                            rotating_prefill_transient_token_count(
                                window_token_count,
                                chunk_token_count,
                            )
                            .map_err(|_| {
                                LagunaExecutionError::invalid_geometry(
                                    "Laguna rotating prefill transient geometry is invalid",
                                )
                            })?,
                        )
                        .map_err(|_| {
                            LagunaExecutionError::invalid_geometry(
                                "Laguna rotating prefill transient exceeds the usize range",
                            )
                        })?
                    }
                    LagunaLayerCacheState::Rotating(_) => 0,
                };
                let layer_growth_bytes = growth_token_count
                    .checked_mul(key_value_bytes_per_token)
                    .ok_or_else(|| {
                        LagunaExecutionError::invalid_geometry(
                            "Laguna projected decoder growth overflowed",
                        )
                    })?;
                let layer_sliding_workspace_bytes = sliding_temporary_token_count
                    .checked_mul(key_value_bytes_per_token)
                    .ok_or_else(|| {
                        LagunaExecutionError::invalid_geometry(
                            "Laguna sliding temporary workspace overflowed",
                        )
                    })?;
                Ok(LagunaDecoderForwardMemoryProjection {
                    persistent_growth_bytes: projection
                        .persistent_growth_bytes
                        .checked_add(layer_growth_bytes)
                        .ok_or_else(|| {
                            LagunaExecutionError::invalid_geometry(
                                "Laguna total projected decoder growth overflowed",
                            )
                        })?,
                    sliding_temporary_workspace_bytes: projection
                        .sliding_temporary_workspace_bytes
                        .checked_add(layer_sliding_workspace_bytes)
                        .ok_or_else(|| {
                            LagunaExecutionError::invalid_geometry(
                                "Laguna total sliding temporary workspace overflowed",
                            )
                        })?,
                })
            },
        )
    }

    /// Persistent growth uses the complete context; temporary rotating workspace
    /// uses only the largest forward that can execute at once. Charging the full
    /// prompt as temporary workspace demotes a fitting resident model.
    pub fn projected_context_admission_memory(
        &self,
        contract: &LagunaTargetContract,
        context_growth_token_count: usize,
        maximum_forward_token_count: usize,
    ) -> Result<LagunaDecoderForwardMemoryProjection, LagunaExecutionError> {
        let context_growth_projection =
            self.projected_forward_memory(contract, context_growth_token_count)?;
        let executable_forward_projection = self.projected_forward_memory(
            contract,
            maximum_forward_token_count.min(context_growth_token_count),
        )?;
        Ok(LagunaDecoderForwardMemoryProjection {
            persistent_growth_bytes: context_growth_projection.persistent_growth_bytes,
            sliding_temporary_workspace_bytes: executable_forward_projection
                .sliding_temporary_workspace_bytes,
        })
    }

    /// Returns the committed token count stored by one layer.
    #[must_use]
    pub fn committed_token_count(&self, layer_index: usize) -> Option<i32> {
        match self.layers.get(layer_index)? {
            LagunaLayerCacheState::AppendOnly(state) => Some(state.offset_tokens()),
            LagunaLayerCacheState::Rotating(state) => Some(state.committed_token_count()),
        }
    }

    /// Retains every mutable cache owner before a forward that may hit the MLX ceiling.
    pub fn allocation_checkpoint(
        &self,
    ) -> Result<LagunaDecoderStateAllocationCheckpoint, LagunaExecutionError> {
        let layers = self
            .layers
            .iter()
            .map(|layer_state| match layer_state {
                LagunaLayerCacheState::AppendOnly(state) => state
                    .allocation_checkpoint()
                    .map(LagunaLayerCacheAllocationCheckpoint::AppendOnly),
                LagunaLayerCacheState::Rotating(state) => state
                    .allocation_checkpoint()
                    .map(LagunaLayerCacheAllocationCheckpoint::Rotating),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(LagunaDecoderStateAllocationCheckpoint { layers })
    }

    /// Restores all cache owners and positions before retrying the unchanged forward.
    pub fn restore_allocation_checkpoint(
        &mut self,
        allocation_checkpoint: LagunaDecoderStateAllocationCheckpoint,
    ) -> Result<(), LagunaExecutionError> {
        if self.layers.len() != allocation_checkpoint.layers.len() {
            return Err(LagunaExecutionError::invalid_geometry(
                "a Laguna decoder allocation checkpoint has the wrong layer count",
            ));
        }
        for (layer_state, layer_checkpoint) in
            self.layers.iter_mut().zip(allocation_checkpoint.layers)
        {
            match (layer_state, layer_checkpoint) {
                (
                    LagunaLayerCacheState::AppendOnly(state),
                    LagunaLayerCacheAllocationCheckpoint::AppendOnly(checkpoint),
                ) => state.restore_allocation_checkpoint(checkpoint)?,
                (
                    LagunaLayerCacheState::Rotating(state),
                    LagunaLayerCacheAllocationCheckpoint::Rotating(checkpoint),
                ) => state.restore_allocation_checkpoint(checkpoint)?,
                _ => {
                    return Err(LagunaExecutionError::invalid_geometry(
                        "a Laguna decoder allocation checkpoint changed cache kind",
                    ));
                }
            }
        }
        Ok(())
    }
}
