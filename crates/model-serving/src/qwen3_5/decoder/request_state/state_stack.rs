use crate::decoder_cache::{
    ConvolutionState, DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS, DecoderCacheLayerLayout,
    DecoderCacheLayout, DecoderCacheState, DecoderCacheStateAllocationCheckpoint,
    DecoderCacheStateCheckpoint, FullAttentionKeyValueState, GatedDeltaRecurrentState,
};
use crate::qwen3_5::Qwen3_5Config;
use astronomical_runtime_integration::MlxRuntimeError;

/// One decoder layer's in-memory state. Linear-attention layers carry a
/// convolution rolling buffer and a gated-delta recurrent state; full-attention
/// layers carry a single KV state owner.
/// One in-memory state entry per certified Qwen3.5 decoder layer. Owns every
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

    /// Creates empty decoder-layer state entries from the validated Qwen3.5
    /// config layer schedule. No MLX arrays are allocated until the first
    /// forward pass touches a layer.
    #[must_use]
    pub fn empty_from_config(qwen3_5_config: &Qwen3_5Config) -> Self {
        Self::empty_from_config_with_validated_full_attention_kv_state_growth_tokens(
            qwen3_5_config,
            DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS,
        )
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
        let decoder_cache_layout =
            crate::qwen3_5::decoder::cache_layout::qwen3_5_decoder_cache_layout(
                qwen3_5_config,
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

    fn empty_from_config_with_validated_full_attention_kv_state_growth_tokens(
        qwen3_5_config: &Qwen3_5Config,
        full_attention_kv_state_growth_tokens: i32,
    ) -> Self {
        let linear_convolution_kernel_dimension =
            qwen3_5_config.linear_convolution_kernel_dimension() as i32;
        let linear_convolution_dimension = qwen3_5_config.linear_convolution_state_dimension();
        let linear_value_head_count = qwen3_5_config.linear_value_head_count() as i32;
        let linear_value_head_dimension = qwen3_5_config.linear_value_head_dimension() as i32;
        let linear_key_head_dimension = qwen3_5_config.linear_key_head_dimension() as i32;
        let layers = (0..qwen3_5_config.layer_count() as usize)
            .map(|decoder_layer_index| {
                if qwen3_5_config.decoder_layer_is_full_attention(decoder_layer_index) {
                    DecoderCacheState::AppendOnlyAttention {
                        attention: FullAttentionKeyValueState::empty_with_validated_growth_tokens(
                            full_attention_kv_state_growth_tokens,
                        ),
                    }
                } else {
                    DecoderCacheState::Composite {
                        convolution: ConvolutionState::empty_with_shape(
                            linear_convolution_kernel_dimension,
                            linear_convolution_dimension,
                        ),
                        recurrent: GatedDeltaRecurrentState::empty_with_shape(
                            linear_value_head_count,
                            linear_value_head_dimension,
                            linear_key_head_dimension,
                        ),
                    }
                }
            })
            .collect();
        Self { layers }
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

    /// Retains all decoder-layer state owners for MTP rollback.
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

fn decoder_cache_layout_projection_error(
    decoder_cache_layout_error: crate::decoder_cache::DecoderCacheLayoutError,
) -> MlxRuntimeError {
    request_decoder_state_error_from_string(format!(
        "validated decoder-cache tensor payload geometry is invalid: {decoder_cache_layout_error}"
    ))
}

fn request_decoder_layer_state_from_layout(
    decoder_cache_layer_layout: &DecoderCacheLayerLayout,
    full_attention_kv_state_growth_tokens_override: Option<i32>,
) -> Result<DecoderCacheState, MlxRuntimeError> {
    match decoder_cache_layer_layout {
        DecoderCacheLayerLayout::AppendOnlyAttention {
            capacity_growth_tokens,
            ..
        } => {
            let full_attention_kv_state_growth_tokens =
                full_attention_kv_state_growth_tokens_override.map_or_else(
                    || {
                        i32::try_from(*capacity_growth_tokens).map_err(|_| {
                            request_decoder_state_error(
                                "append-only attention growth exceeds the i32 range",
                            )
                        })
                    },
                    Ok,
                )?;
            if full_attention_kv_state_growth_tokens <= 0 {
                return Err(request_decoder_state_error(
                    "append-only attention growth must be positive",
                ));
            }
            Ok(DecoderCacheState::AppendOnlyAttention {
                attention: FullAttentionKeyValueState::empty_with_validated_growth_tokens(
                    full_attention_kv_state_growth_tokens,
                ),
            })
        }
        DecoderCacheLayerLayout::Composite { components } => {
            let [
                DecoderCacheLayerLayout::RecurrentTensor {
                    tensor: convolution_tensor,
                },
                DecoderCacheLayerLayout::RecurrentTensor {
                    tensor: recurrent_tensor,
                },
            ] = components.as_slice()
            else {
                return Err(request_decoder_state_error(
                    "Qwen linear attention requires convolution and recurrent tensor components",
                ));
            };
            if convolution_tensor.qualified_role_name()
                != crate::qwen3_5::decoder::cache_layout::QWEN_CONVOLUTION_TENSOR_ROLE
                || recurrent_tensor.qualified_role_name()
                    != crate::qwen3_5::decoder::cache_layout::QWEN_RECURRENCE_TENSOR_ROLE
            {
                return Err(request_decoder_state_error(
                    "Qwen linear attention tensor roles do not match the model contract",
                ));
            }
            let [1, convolution_history_tokens, convolution_dimension] =
                convolution_tensor.dimensions()
            else {
                return Err(request_decoder_state_error(
                    "Qwen convolution state must have rank three with batch size one",
                ));
            };
            let [
                1,
                recurrent_value_head_count,
                recurrent_value_head_dimension,
                recurrent_key_head_dimension,
            ] = recurrent_tensor.dimensions()
            else {
                return Err(request_decoder_state_error(
                    "Qwen recurrent state must have rank four with batch size one",
                ));
            };
            let convolution_kernel_dimension =
                convolution_history_tokens.checked_add(1).ok_or_else(|| {
                    request_decoder_state_error("Qwen convolution history length overflowed")
                })?;
            Ok(DecoderCacheState::Composite {
                convolution: ConvolutionState::empty_with_shape(
                    i32::try_from(convolution_kernel_dimension).map_err(|_| {
                        request_decoder_state_error("Qwen convolution kernel dimension exceeds i32")
                    })?,
                    i32::try_from(*convolution_dimension).map_err(|_| {
                        request_decoder_state_error("Qwen convolution dimension exceeds i32")
                    })?,
                ),
                recurrent: GatedDeltaRecurrentState::empty_with_shape(
                    i32::try_from(*recurrent_value_head_count).map_err(|_| {
                        request_decoder_state_error("Qwen recurrent value head count exceeds i32")
                    })?,
                    i32::try_from(*recurrent_value_head_dimension).map_err(|_| {
                        request_decoder_state_error(
                            "Qwen recurrent value head dimension exceeds i32",
                        )
                    })?,
                    i32::try_from(*recurrent_key_head_dimension).map_err(|_| {
                        request_decoder_state_error("Qwen recurrent key head dimension exceeds i32")
                    })?,
                ),
            })
        }
        DecoderCacheLayerLayout::RecurrentTensor { .. } => Err(request_decoder_state_error(
            "Qwen decoder layers cannot contain a standalone recurrent tensor",
        )),
    }
}

fn request_decoder_state_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: "create Qwen3.5 request decoder state",
        description: description.to_owned(),
    }
}

fn request_decoder_state_error_from_string(description: String) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: "create Qwen3.5 request decoder state",
        description,
    }
}
