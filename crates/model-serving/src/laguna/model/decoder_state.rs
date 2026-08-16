//! Per-layer Laguna decoder cache: append-only full attention or rotating sliding.

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use crate::decoder_cache::{FullAttentionKeyValueState, RotatingKeyValueState};
use crate::laguna::normalization::{LagunaCacheDescriptor, LagunaTargetContract};
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

    /// Returns the committed token count stored by one layer.
    #[must_use]
    pub fn committed_token_count(&self, layer_index: usize) -> Option<i32> {
        match self.layers.get(layer_index)? {
            LagunaLayerCacheState::AppendOnly(state) => Some(state.offset_tokens()),
            LagunaLayerCacheState::Rotating(state) => Some(state.committed_token_count()),
        }
    }
}
