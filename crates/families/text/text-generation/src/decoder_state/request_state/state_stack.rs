use crate::astro_decoder_state_cache::{
    AstroDecoderStateCacheLayerLayout, AstroDecoderStateCacheLayout, AstroDecoderStateCacheState,
    AstroDecoderStateCacheStateAllocationCheckpoint, AstroDecoderStateCacheStateCheckpoint,
    AstroDecoderStateCacheTensorLayout, ConvolutionState, FullAttentionKeyValueState,
    GatedDeltaRecurrentState,
};
use runtime_integration::MlxRuntimeError;

use crate::decoder_state::family_decoder_state::FamilyDecoderState;

use super::state_stack_layout::{
    astro_decoder_state_cache_layout_projection_error, request_decoder_layer_state_from_layout,
    request_decoder_state_error,
};

/// One decoder layer's in-memory state. Linear-attention layers carry a
/// convolution rolling buffer and a gated-delta recurrent state; full-attention
/// layers carry a single KV state owner.
/// One in-memory state entry per decoder layer. Owns every
/// KV tensor and recurrent state for one request; the engine threads a
/// `&mut RequestDecoderStateStack` through every forward pass.
pub struct RequestDecoderStateStack {
    layers: Vec<AstroDecoderStateCacheState>,
    /// A family's extra per-request state, installed by its first forward.
    family_state: Option<Box<dyn FamilyDecoderState>>,
}

/// Retained checkpoint of a request's complete decoder-state stack.
pub struct RequestDecoderStateStackCheckpoint {
    layers: Vec<AstroDecoderStateCacheStateCheckpoint>,
    family_state: Option<Box<dyn FamilyDecoderState>>,
}

/// Retained physical owner checkpoint of a request's complete decoder-state stack.
pub struct RequestDecoderStateStackAllocationCheckpoint {
    layers: Vec<AstroDecoderStateCacheStateAllocationCheckpoint>,
    family_state: Option<Box<dyn FamilyDecoderState>>,
}

impl RequestDecoderStateStack {
    /// Creates the live state owners from the validated, model-owned astro decoder state cache layout.
    pub fn empty_from_astro_decoder_state_cache_layout(
        astro_decoder_state_cache_layout: &AstroDecoderStateCacheLayout,
    ) -> Result<Self, MlxRuntimeError> {
        Self::empty_from_astro_decoder_state_cache_layout_with_growth_override(
            astro_decoder_state_cache_layout,
            None,
        )
    }

    /// Creates live state from the model-owned layout with an explicit attention slab granularity.
    pub fn empty_from_astro_decoder_state_cache_layout_with_full_attention_kv_state_growth_tokens(
        astro_decoder_state_cache_layout: &AstroDecoderStateCacheLayout,
        full_attention_kv_state_growth_tokens: i32,
    ) -> Result<Self, MlxRuntimeError> {
        if full_attention_kv_state_growth_tokens <= 0 {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: "create request decoder state",
                description: "full-attention KV-state growth tokens must be positive".to_owned(),
            });
        }
        Self::empty_from_astro_decoder_state_cache_layout_with_growth_override(
            astro_decoder_state_cache_layout,
            Some(full_attention_kv_state_growth_tokens),
        )
    }

    fn empty_from_astro_decoder_state_cache_layout_with_growth_override(
        astro_decoder_state_cache_layout: &AstroDecoderStateCacheLayout,
        full_attention_kv_state_growth_tokens_override: Option<i32>,
    ) -> Result<Self, MlxRuntimeError> {
        let mut layers = Vec::with_capacity(astro_decoder_state_cache_layout.layer_count());
        for decoder_layer_index in 0..astro_decoder_state_cache_layout.layer_count() {
            let layer_layout = astro_decoder_state_cache_layout
                .layer(decoder_layer_index)
                .ok_or_else(|| {
                    request_decoder_state_error(
                        "astro-decoder-state-cache layout lost a declared decoder layer",
                    )
                })?;
            layers.push(request_decoder_layer_state_from_layout(
                layer_layout,
                full_attention_kv_state_growth_tokens_override,
            )?);
        }
        Ok(Self {
            layers,
            family_state: None,
        })
    }

    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    #[must_use]
    pub fn layer(&self, layer_index: usize) -> Option<&AstroDecoderStateCacheState> {
        self.layers.get(layer_index)
    }

    #[must_use]
    pub fn layer_mut(&mut self, layer_index: usize) -> Option<&mut AstroDecoderStateCacheState> {
        self.layers.get_mut(layer_index)
    }

    /// Every decoder layer's state in decoder-layer order, for callers that
    /// visit all layers of one family without an index lookup per layer.
    #[must_use]
    pub fn layers(&self) -> &[AstroDecoderStateCacheState] {
        &self.layers
    }

    /// Mutable decoder-layer states in decoder-layer order, for seating a
    /// restored prompt prefix layer by layer.
    #[must_use]
    pub fn layers_mut(&mut self) -> &mut [AstroDecoderStateCacheState] {
        &mut self.layers
    }

    #[must_use]
    /// Returns logical payload bytes across all request-owned decoder states.
    pub fn payload_byte_count(&self) -> u64 {
        let family_state_bytes = self
            .family_state
            .as_ref()
            .map_or(0, |family_state| family_state.payload_byte_count());
        self.layers
            .iter()
            .map(AstroDecoderStateCacheState::payload_byte_count)
            .fold(family_state_bytes, u64::saturating_add)
    }

    /// The family's extra state of type `FamilyState`, when installed.
    #[must_use]
    pub fn family_state_mut<FamilyState: FamilyDecoderState>(
        &mut self,
    ) -> Option<&mut FamilyState> {
        self.family_state
            .as_mut()
            .and_then(|family_state| family_state.as_any_mut().downcast_mut::<FamilyState>())
    }

    /// The family's extra state of type `FamilyState`, when installed.
    #[must_use]
    pub fn family_state<FamilyState: FamilyDecoderState>(&self) -> Option<&FamilyState> {
        self.family_state
            .as_ref()
            .and_then(|family_state| family_state.as_any().downcast_ref::<FamilyState>())
    }

    /// Arrays the family state needs evaluated with each forward.
    pub fn append_family_state_evaluation_arrays<'state>(
        &'state self,
        arrays: &mut Vec<&'state mlx_bridge::MlxArray>,
    ) {
        if let Some(family_state) = &self.family_state {
            family_state.append_evaluation_arrays(arrays);
        }
    }

    /// One layer's state and the family state of type `FamilyState`
    /// together, for a layer forward that reads both.
    pub fn layer_and_family_state_mut<FamilyState: FamilyDecoderState>(
        &mut self,
        layer_index: usize,
    ) -> Option<(&mut AstroDecoderStateCacheState, Option<&mut FamilyState>)> {
        let family_state = self
            .family_state
            .as_mut()
            .and_then(|family_state| family_state.as_any_mut().downcast_mut::<FamilyState>());
        self.layers
            .get_mut(layer_index)
            .map(|layer_state| (layer_state, family_state))
    }

    /// Installs a family's extra state, replacing any earlier one.
    pub fn install_family_state(&mut self, family_state: Box<dyn FamilyDecoderState>) {
        self.family_state = Some(family_state);
    }

    /// A copy of the family's extra state, or `None` when the family keeps
    /// none; `restore_family_state` puts it back.
    pub fn family_state_snapshot(
        &self,
    ) -> Result<Option<Box<dyn FamilyDecoderState>>, MlxRuntimeError> {
        self.family_state
            .as_ref()
            .map(|family_state| family_state.snapshot())
            .transpose()
    }

    /// Puts back a family state taken by `family_state_snapshot`.
    pub fn restore_family_state(&mut self, family_state: Option<Box<dyn FamilyDecoderState>>) {
        self.family_state = family_state;
    }

    /// Retains all decoder-layer state owners for verified-prefix rollback.
    pub fn checkpoint(&self) -> Result<RequestDecoderStateStackCheckpoint, MlxRuntimeError> {
        Ok(RequestDecoderStateStackCheckpoint {
            layers: self.layer_checkpoints()?,
            family_state: self.family_state_snapshot()?,
        })
    }

    fn layer_checkpoints(
        &self,
    ) -> Result<Vec<AstroDecoderStateCacheStateCheckpoint>, MlxRuntimeError> {
        let mut layer_checkpoints = Vec::with_capacity(self.layers.len());
        for decoder_layer_state in &self.layers {
            layer_checkpoints.push(decoder_layer_state.checkpoint()?);
        }
        Ok(layer_checkpoints)
    }

    /// Restores every decoder-layer state owner from a retained checkpoint.
    pub fn restore_checkpoint(
        &mut self,
        checkpoint: RequestDecoderStateStackCheckpoint,
    ) -> Result<(), MlxRuntimeError> {
        if checkpoint.layers.len() != self.layers.len() {
            return Err(request_decoder_state_error(
                "decoder-state checkpoint layer count does not match the live stack",
            ));
        }
        let checkpoint_families_match = self.layers.iter().zip(&checkpoint.layers).all(
            |(decoder_layer_state, layer_checkpoint)| {
                matches!(
                    (decoder_layer_state, layer_checkpoint),
                    (
                        AstroDecoderStateCacheState::AppendOnlyAttention { .. },
                        AstroDecoderStateCacheStateCheckpoint::AppendOnlyAttention { .. }
                    ) | (
                        AstroDecoderStateCacheState::Composite { .. },
                        AstroDecoderStateCacheStateCheckpoint::Composite { .. }
                    )
                )
            },
        );
        if !checkpoint_families_match {
            return Err(request_decoder_state_error(
                "decoder-state checkpoint layer families do not match the live stack",
            ));
        }
        for (decoder_layer_state, layer_checkpoint) in self.layers.iter_mut().zip(checkpoint.layers)
        {
            decoder_layer_state.restore_checkpoint(layer_checkpoint)?;
        }
        self.family_state = checkpoint.family_state;
        Ok(())
    }

    /// Retains every physical decoder-state owner for a retryable prompt attempt.
    pub fn allocation_checkpoint(
        &self,
    ) -> Result<RequestDecoderStateStackAllocationCheckpoint, MlxRuntimeError> {
        Ok(RequestDecoderStateStackAllocationCheckpoint {
            layers: self.layer_allocation_checkpoints()?,
            family_state: self.family_state_snapshot()?,
        })
    }

    fn layer_allocation_checkpoints(
        &self,
    ) -> Result<Vec<AstroDecoderStateCacheStateAllocationCheckpoint>, MlxRuntimeError> {
        let mut layer_allocation_checkpoints = Vec::with_capacity(self.layers.len());
        for decoder_layer_state in &self.layers {
            layer_allocation_checkpoints.push(decoder_layer_state.allocation_checkpoint()?);
        }
        Ok(layer_allocation_checkpoints)
    }

    /// Restores every physical decoder-state owner after a failed prompt attempt.
    pub fn restore_allocation_checkpoint(
        &mut self,
        allocation_checkpoint: RequestDecoderStateStackAllocationCheckpoint,
    ) -> Result<(), MlxRuntimeError> {
        if allocation_checkpoint.layers.len() != self.layers.len() {
            return Err(request_decoder_state_error(
                "decoder-state allocation checkpoint layer count does not match the live stack",
            ));
        }
        let checkpoint_families_match = self.layers.iter().zip(&allocation_checkpoint.layers).all(
            |(decoder_layer_state, decoder_layer_allocation_checkpoint)| {
                matches!(
                    (decoder_layer_state, decoder_layer_allocation_checkpoint),
                    (
                        AstroDecoderStateCacheState::AppendOnlyAttention { .. },
                        AstroDecoderStateCacheStateAllocationCheckpoint::AppendOnlyAttention { .. }
                    ) | (
                        AstroDecoderStateCacheState::Composite { .. },
                        AstroDecoderStateCacheStateAllocationCheckpoint::Composite { .. }
                    )
                )
            },
        );
        if !checkpoint_families_match {
            return Err(request_decoder_state_error(
                "decoder-state allocation checkpoint layer families do not match the live stack",
            ));
        }
        for (decoder_layer_state, decoder_layer_allocation_checkpoint) in
            self.layers.iter_mut().zip(allocation_checkpoint.layers)
        {
            decoder_layer_state
                .restore_allocation_checkpoint(decoder_layer_allocation_checkpoint)?;
        }
        self.family_state = allocation_checkpoint.family_state;
        Ok(())
    }

    /// Projects every request-owned persistent allocation for one forthcoming forward.
    ///
    /// The validated layout supplies tensor geometry and dtypes. Full-attention
    /// state contributes only newly rounded slab capacity; fixed convolution and
    /// recurrent state contributes only on first use.
    pub fn projected_persistent_state_growth_bytes(
        &self,
        astro_decoder_state_cache_layout: &AstroDecoderStateCacheLayout,
        update_token_count: usize,
    ) -> Result<usize, MlxRuntimeError> {
        if self.layers.len() != astro_decoder_state_cache_layout.layer_count() {
            return Err(request_decoder_state_error(
                "decoder-state layer count does not match the validated astro-decoder-state-cache layout",
            ));
        }
        let mut persistent_growth_bytes = 0_usize;
        for (decoder_layer_index, decoder_layer_state) in self.layers.iter().enumerate() {
            let astro_decoder_state_cache_layer_layout = astro_decoder_state_cache_layout
                .layer(decoder_layer_index)
                .ok_or_else(|| {
                    request_decoder_state_error(
                        "validated astro-decoder-state-cache layout lost a declared decoder layer",
                    )
                })?;
            let layer_growth_bytes = match (
                decoder_layer_state,
                astro_decoder_state_cache_layer_layout,
            ) {
                (
                    AstroDecoderStateCacheState::AppendOnlyAttention { attention },
                    AstroDecoderStateCacheLayerLayout::AppendOnlyAttention { keys, values, .. },
                ) => full_attention_growth_bytes(attention, keys, values, update_token_count)?,
                (
                    AstroDecoderStateCacheState::Composite {
                        convolution,
                        recurrent,
                    },
                    AstroDecoderStateCacheLayerLayout::Composite { components },
                ) => composite_first_use_bytes(convolution, recurrent, components)?,
                _ => {
                    return Err(request_decoder_state_error(
                        "decoder state family does not match the validated astro-decoder-state-cache layout",
                    ));
                }
            };
            persistent_growth_bytes = persistent_growth_bytes
                .checked_add(layer_growth_bytes)
                .ok_or_else(|| {
                    request_decoder_state_error(
                        "total persistent decoder state growth bytes overflowed",
                    )
                })?;
        }
        Ok(persistent_growth_bytes)
    }
}

/// The key and value bytes of the slab capacity this update adds. Only
/// whole growth slabs are allocated, so a short update inside the current
/// slab adds nothing.
fn full_attention_growth_bytes(
    attention: &FullAttentionKeyValueState,
    keys: &AstroDecoderStateCacheTensorLayout,
    values: &AstroDecoderStateCacheTensorLayout,
    update_token_count: usize,
) -> Result<usize, MlxRuntimeError> {
    let capacity_growth_tokens = attention.projected_capacity_growth_tokens(update_token_count)?;
    let bytes_per_token = keys
        .sequence_payload_byte_count_per_token()
        .map_err(astro_decoder_state_cache_layout_projection_error)?
        .checked_add(
            values
                .sequence_payload_byte_count_per_token()
                .map_err(astro_decoder_state_cache_layout_projection_error)?,
        )
        .ok_or_else(|| {
            request_decoder_state_error("full-attention key/value bytes per token overflowed")
        })?;
    capacity_growth_tokens
        .checked_mul(bytes_per_token)
        .ok_or_else(|| {
            request_decoder_state_error("full-attention persistent state growth bytes overflowed")
        })
}

/// Convolution and recurrent state are fixed-size and allocated together on
/// a linear-attention layer's first forward, then never grow.
fn composite_first_use_bytes(
    convolution: &ConvolutionState,
    recurrent: &GatedDeltaRecurrentState,
    components: &[AstroDecoderStateCacheLayerLayout],
) -> Result<usize, MlxRuntimeError> {
    let [
        AstroDecoderStateCacheLayerLayout::RecurrentTensor {
            tensor: convolution_layout,
        },
        AstroDecoderStateCacheLayerLayout::RecurrentTensor {
            tensor: recurrent_layout,
        },
    ] = components
    else {
        return Err(request_decoder_state_error(
            "composite decoder state layout must contain convolution and recurrent tensors",
        ));
    };
    match (convolution.is_unallocated(), recurrent.is_unallocated()) {
        (true, true) => convolution_layout
            .fixed_payload_byte_count()
            .map_err(astro_decoder_state_cache_layout_projection_error)?
            .checked_add(
                recurrent_layout
                    .fixed_payload_byte_count()
                    .map_err(astro_decoder_state_cache_layout_projection_error)?,
            )
            .ok_or_else(|| {
                request_decoder_state_error("composite persistent state growth bytes overflowed")
            }),
        (false, false) => Ok(0),
        _ => Err(request_decoder_state_error(
            "composite decoder state is partially allocated",
        )),
    }
}

#[cfg(test)]
#[path = "state_stack_tests.rs"]
mod state_stack_tests;
