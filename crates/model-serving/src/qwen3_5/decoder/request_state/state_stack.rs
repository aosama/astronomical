use crate::decoder_cache::{
    DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCacheState,
    DecoderCacheStateAllocationCheckpoint, DecoderCacheStateCheckpoint,
};
use crate::qwen3_5::Qwen3_5Config;
use astronomical_runtime_integration::MlxRuntimeError;

use super::state_stack_layout::{
    decoder_cache_layout_projection_error, request_decoder_layer_state_from_layout,
    request_decoder_state_error, request_decoder_state_error_from_string,
};
use crate::qwen3_5::decoder::Qwen3_5PersistentPromptCacheBoundaryCheckpoint;

/// One decoder layer's in-memory state. Linear-attention layers carry a
/// convolution rolling buffer and a gated-delta recurrent state; full-attention
/// layers carry a single KV state owner.
/// One in-memory state entry per Qwen3.5 decoder layer. Owns every
/// KV tensor and recurrent state for one request; the engine threads a
/// `&mut RequestDecoderStateStack` through every forward pass.
pub struct RequestDecoderStateStack {
    layers: Vec<DecoderCacheState>,
}

/// Retained checkpoint of a request's complete decoder-state stack.
pub struct RequestDecoderStateStackCheckpoint {
    layers: Vec<DecoderCacheStateCheckpoint>,
}

/// Retained physical owner checkpoint of a request's complete decoder-state stack.
pub struct RequestDecoderStateStackAllocationCheckpoint {
    layers: Vec<DecoderCacheStateAllocationCheckpoint>,
}

impl RequestDecoderStateStack {
    /// Creates Qwen's live state owners from the validated, model-owned cache layout.
    pub fn empty_from_decoder_cache_layout(
        decoder_cache_layout: &DecoderCacheLayout,
    ) -> Result<Self, MlxRuntimeError> {
        Self::empty_from_decoder_cache_layout_with_growth_override(decoder_cache_layout, None)
    }

    /// Creates live state from the model-owned layout with an explicit attention slab granularity.
    pub fn empty_from_decoder_cache_layout_with_full_attention_kv_state_growth_tokens(
        decoder_cache_layout: &DecoderCacheLayout,
        full_attention_kv_state_growth_tokens: i32,
    ) -> Result<Self, MlxRuntimeError> {
        if full_attention_kv_state_growth_tokens <= 0 {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: "create Qwen3.5 request decoder state",
                description: "full-attention KV-state growth tokens must be positive".to_owned(),
            });
        }
        Self::empty_from_decoder_cache_layout_with_growth_override(
            decoder_cache_layout,
            Some(full_attention_kv_state_growth_tokens),
        )
    }

    fn empty_from_decoder_cache_layout_with_growth_override(
        decoder_cache_layout: &DecoderCacheLayout,
        full_attention_kv_state_growth_tokens_override: Option<i32>,
    ) -> Result<Self, MlxRuntimeError> {
        let mut layers = Vec::with_capacity(decoder_cache_layout.layer_count());
        for decoder_layer_index in 0..decoder_cache_layout.layer_count() {
            let layer_layout =
                decoder_cache_layout
                    .layer(decoder_layer_index)
                    .ok_or_else(|| {
                        request_decoder_state_error(
                            "decoder-cache layout lost a declared decoder layer",
                        )
                    })?;
            layers.push(request_decoder_layer_state_from_layout(
                layer_layout,
                full_attention_kv_state_growth_tokens_override,
            )?);
        }
        Ok(Self { layers })
    }

    /// Creates empty decoder-layer state entries using an explicit
    /// full-attention KV capacity-growth granularity.
    pub fn empty_from_config_with_full_attention_kv_state_growth_tokens(
        qwen3_5_config: &Qwen3_5Config,
        full_attention_kv_state_growth_tokens: i32,
    ) -> Result<Self, MlxRuntimeError> {
        if full_attention_kv_state_growth_tokens <= 0 {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: "create Qwen3.5 request decoder state",
                description: "full-attention KV-state growth tokens must be positive".to_owned(),
            });
        }
        // This config-only constructor is retained for synthetic tests that have
        // no bound weights to inspect. Production model loading derives each
        // layer's exact dtype from the live affine graph before creating state.
        let configured_activation_cache_dtype =
            crate::decoder_cache::DecoderCacheTensorDtype::BFloat16;
        let decoder_layer_cache_dtypes = (0..qwen3_5_config.layer_count() as usize)
            .map(|decoder_layer_index| {
                if qwen3_5_config.decoder_layer_is_full_attention(decoder_layer_index) {
                    crate::qwen3_5::decoder::Qwen3_5DecoderLayerCacheDtypes::FullAttention {
                        keys: configured_activation_cache_dtype,
                        values: configured_activation_cache_dtype,
                    }
                } else {
                    crate::qwen3_5::decoder::Qwen3_5DecoderLayerCacheDtypes::LinearAttention {
                        convolution: configured_activation_cache_dtype,
                    }
                }
            })
            .collect::<Vec<_>>();
        let decoder_cache_layout =
            crate::qwen3_5::decoder::cache_layout::qwen3_5_decoder_cache_layout(
                qwen3_5_config,
                usize::try_from(full_attention_kv_state_growth_tokens).map_err(|_| {
                    request_decoder_state_error(
                        "full-attention key/value growth tokens exceed the usize range",
                    )
                })?,
                &decoder_layer_cache_dtypes,
            )
            .map_err(|decoder_cache_layout_error| {
                request_decoder_state_error_from_string(format!(
                    "validated Qwen decoder-cache layout is invalid: {decoder_cache_layout_error}"
                ))
            })?;
        Self::empty_from_decoder_cache_layout_with_full_attention_kv_state_growth_tokens(
            &decoder_cache_layout,
            full_attention_kv_state_growth_tokens,
        )
    }

    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    #[must_use]
    pub fn layer(&self, layer_index: usize) -> Option<&DecoderCacheState> {
        self.layers.get(layer_index)
    }

    #[must_use]
    pub fn layer_mut(&mut self, layer_index: usize) -> Option<&mut DecoderCacheState> {
        self.layers.get_mut(layer_index)
    }

    #[must_use]
    /// Returns logical payload bytes across all request-owned decoder states.
    pub fn payload_byte_count(&self) -> u64 {
        self.layers
            .iter()
            .map(DecoderCacheState::payload_byte_count)
            .fold(0, u64::saturating_add)
    }

    /// Retains all decoder-layer state owners for verified-prefix rollback.
    pub fn checkpoint(&self) -> Result<RequestDecoderStateStackCheckpoint, MlxRuntimeError> {
        let mut layer_checkpoints = Vec::with_capacity(self.layers.len());
        for decoder_layer_state in &self.layers {
            layer_checkpoints.push(decoder_layer_state.checkpoint()?);
        }
        Ok(RequestDecoderStateStackCheckpoint {
            layers: layer_checkpoints,
        })
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
                        DecoderCacheState::AppendOnlyAttention { .. },
                        DecoderCacheStateCheckpoint::AppendOnlyAttention { .. }
                    ) | (
                        DecoderCacheState::Composite { .. },
                        DecoderCacheStateCheckpoint::Composite { .. }
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
        Ok(())
    }

    /// Restores any validated verifier-prefix boundary retained by fixed-depth MTP.
    #[doc(hidden)]
    pub fn restore_verified_prefix(
        &mut self,
        verified_prefix_position_tokens: u32,
        mut verified_prefix_boundary_checkpoint: Qwen3_5PersistentPromptCacheBoundaryCheckpoint,
    ) -> Result<(), MlxRuntimeError> {
        if verified_prefix_boundary_checkpoint.completed_prefill_chunck_tokens == 0 {
            return Err(request_decoder_state_error(
                "verification boundary must retain at least one verifier row",
            ));
        }
        let verified_prefix_attention_offset_tokens =
            i32::try_from(verified_prefix_position_tokens).map_err(|_| {
                request_decoder_state_error("verified-prefix position exceeds the Int32 range")
            })?;
        let expected_recurrent_snapshot_tensor_count = self
            .layers
            .iter()
            .filter(|decoder_layer_state| {
                matches!(decoder_layer_state, DecoderCacheState::Composite { .. })
            })
            .count()
            .checked_mul(2)
            .ok_or_else(|| {
                request_decoder_state_error("verified-prefix recurrent tensor count overflowed")
            })?;
        if verified_prefix_boundary_checkpoint
            .recurrent_snapshot_tensors
            .len()
            != expected_recurrent_snapshot_tensor_count
        {
            return Err(request_decoder_state_error(
                "verified-prefix boundary tensor count does not match the decoder state",
            ));
        }

        for (decoder_layer_index, decoder_layer_state) in self.layers.iter().enumerate() {
            match decoder_layer_state {
                DecoderCacheState::AppendOnlyAttention { attention } => {
                    if verified_prefix_attention_offset_tokens > attention.offset_tokens() {
                        return Err(request_decoder_state_error(
                            "verified-prefix position exceeds live attention state",
                        ));
                    }
                }
                DecoderCacheState::Composite { .. } => {
                    let convolution_tensor_name =
                        format!("layer_{decoder_layer_index}_linear.convolution");
                    let recurrent_tensor_name =
                        format!("layer_{decoder_layer_index}_linear.gated_delta_recurrent");
                    if !verified_prefix_boundary_checkpoint
                        .recurrent_snapshot_tensors
                        .contains_key(&convolution_tensor_name)
                        || !verified_prefix_boundary_checkpoint
                            .recurrent_snapshot_tensors
                            .contains_key(&recurrent_tensor_name)
                    {
                        return Err(request_decoder_state_error(
                            "verified-prefix boundary is missing recurrent state",
                        ));
                    }
                }
            }
        }

        for (decoder_layer_index, decoder_layer_state) in self.layers.iter_mut().enumerate() {
            match decoder_layer_state {
                DecoderCacheState::AppendOnlyAttention { attention } => {
                    attention.truncate_to_offset(verified_prefix_attention_offset_tokens)?;
                }
                DecoderCacheState::Composite {
                    convolution,
                    recurrent,
                } => {
                    let convolution_tensor_name =
                        format!("layer_{decoder_layer_index}_linear.convolution");
                    let recurrent_tensor_name =
                        format!("layer_{decoder_layer_index}_linear.gated_delta_recurrent");
                    let retained_convolution_state = verified_prefix_boundary_checkpoint
                        .recurrent_snapshot_tensors
                        .remove(&convolution_tensor_name)
                        .ok_or_else(|| {
                            request_decoder_state_error(
                                "verified-prefix boundary lost convolution state",
                            )
                        })?;
                    let retained_recurrent_state = verified_prefix_boundary_checkpoint
                        .recurrent_snapshot_tensors
                        .remove(&recurrent_tensor_name)
                        .ok_or_else(|| {
                            request_decoder_state_error(
                                "verified-prefix boundary lost gated-delta recurrent state",
                            )
                        })?;
                    convolution.restore_from_snapshot(retained_convolution_state);
                    recurrent.restore_from_snapshot(retained_recurrent_state);
                }
            }
        }
        Ok(())
    }

    /// Retains every physical decoder-state owner for a retryable prompt attempt.
    pub fn allocation_checkpoint(
        &self,
    ) -> Result<RequestDecoderStateStackAllocationCheckpoint, MlxRuntimeError> {
        let mut layer_allocation_checkpoints = Vec::with_capacity(self.layers.len());
        for decoder_layer_state in &self.layers {
            layer_allocation_checkpoints.push(decoder_layer_state.allocation_checkpoint()?);
        }
        Ok(RequestDecoderStateStackAllocationCheckpoint {
            layers: layer_allocation_checkpoints,
        })
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
                        DecoderCacheState::AppendOnlyAttention { .. },
                        DecoderCacheStateAllocationCheckpoint::AppendOnlyAttention { .. }
                    ) | (
                        DecoderCacheState::Composite { .. },
                        DecoderCacheStateAllocationCheckpoint::Composite { .. }
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
        Ok(())
    }

    /// Projects every request-owned persistent allocation for one forthcoming forward.
    ///
    /// The validated layout supplies tensor geometry and dtypes. Full-attention
    /// state contributes only newly rounded slab capacity; fixed convolution and
    /// recurrent state contributes only on first use.
    pub fn projected_persistent_state_growth_bytes(
        &self,
        decoder_cache_layout: &DecoderCacheLayout,
        update_token_count: usize,
    ) -> Result<usize, MlxRuntimeError> {
        if self.layers.len() != decoder_cache_layout.layer_count() {
            return Err(request_decoder_state_error(
                "decoder-state layer count does not match the validated decoder-cache layout",
            ));
        }
        let mut persistent_growth_bytes = 0_usize;
        for (decoder_layer_index, decoder_layer_state) in self.layers.iter().enumerate() {
            let decoder_cache_layer_layout = decoder_cache_layout
                .layer(decoder_layer_index)
                .ok_or_else(|| {
                    request_decoder_state_error(
                        "validated decoder-cache layout lost a declared decoder layer",
                    )
                })?;
            let layer_growth_bytes = match (decoder_layer_state, decoder_cache_layer_layout) {
                (
                    DecoderCacheState::AppendOnlyAttention { attention },
                    DecoderCacheLayerLayout::AppendOnlyAttention { keys, values, .. },
                ) => {
                    let capacity_growth_tokens =
                        attention.projected_capacity_growth_tokens(update_token_count)?;
                    let bytes_per_token = keys
                        .sequence_payload_byte_count_per_token()
                        .map_err(decoder_cache_layout_projection_error)?
                        .checked_add(
                            values
                                .sequence_payload_byte_count_per_token()
                                .map_err(decoder_cache_layout_projection_error)?,
                        )
                        .ok_or_else(|| {
                            request_decoder_state_error(
                                "full-attention key/value bytes per token overflowed",
                            )
                        })?;
                    capacity_growth_tokens
                        .checked_mul(bytes_per_token)
                        .ok_or_else(|| {
                            request_decoder_state_error(
                                "full-attention persistent state growth bytes overflowed",
                            )
                        })?
                }
                (
                    DecoderCacheState::Composite {
                        convolution,
                        recurrent,
                    },
                    DecoderCacheLayerLayout::Composite { components },
                ) => {
                    let [
                        DecoderCacheLayerLayout::RecurrentTensor {
                            tensor: convolution_layout,
                        },
                        DecoderCacheLayerLayout::RecurrentTensor {
                            tensor: recurrent_layout,
                        },
                    ] = components.as_slice()
                    else {
                        return Err(request_decoder_state_error(
                            "composite decoder state layout must contain convolution and recurrent tensors",
                        ));
                    };
                    match (convolution.is_unallocated(), recurrent.is_unallocated()) {
                        (true, true) => convolution_layout
                            .fixed_payload_byte_count()
                            .map_err(decoder_cache_layout_projection_error)?
                            .checked_add(
                                recurrent_layout
                                    .fixed_payload_byte_count()
                                    .map_err(decoder_cache_layout_projection_error)?,
                            )
                            .ok_or_else(|| {
                                request_decoder_state_error(
                                    "composite persistent state growth bytes overflowed",
                                )
                            })?,
                        (false, false) => 0,
                        _ => {
                            return Err(request_decoder_state_error(
                                "composite decoder state is partially allocated",
                            ));
                        }
                    }
                }
                _ => {
                    return Err(request_decoder_state_error(
                        "decoder state family does not match the validated decoder-cache layout",
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
