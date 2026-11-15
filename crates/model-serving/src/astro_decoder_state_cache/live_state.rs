use astronomical_runtime_integration::MlxRuntimeError;

use super::{
    ConvolutionState, ConvolutionStateCheckpoint, FullAttentionKeyValueState,
    FullAttentionKeyValueStateAllocationCheckpoint, GatedDeltaRecurrentState,
    GatedDeltaRecurrentStateCheckpoint,
};

/// One architecture-neutral in-memory astro decoder state cache state family.
pub enum AstroDecoderStateCacheState {
    /// Append-only attention keys and values.
    AppendOnlyAttention {
        attention: FullAttentionKeyValueState,
    },
    /// A hybrid layer combining rolling convolution and recurrent state.
    Composite {
        convolution: ConvolutionState,
        recurrent: GatedDeltaRecurrentState,
    },
}

/// Logical rollback point for one architecture-neutral astro decoder state cache state entry.
pub enum AstroDecoderStateCacheStateCheckpoint {
    AppendOnlyAttention {
        offset_tokens: i32,
    },
    Composite {
        convolution: ConvolutionStateCheckpoint,
        recurrent: GatedDeltaRecurrentStateCheckpoint,
    },
}

/// Physical owner checkpoint for retrying one astro decoder state cache update.
pub enum AstroDecoderStateCacheStateAllocationCheckpoint {
    AppendOnlyAttention {
        attention: FullAttentionKeyValueStateAllocationCheckpoint,
    },
    Composite {
        convolution: ConvolutionStateCheckpoint,
        recurrent: GatedDeltaRecurrentStateCheckpoint,
    },
}

impl AstroDecoderStateCacheState {
    /// Returns whether paired state tensors are both absent or both present.
    #[must_use]
    pub fn tensors_are_allocated_consistently(&self) -> bool {
        match self {
            Self::AppendOnlyAttention { attention } => {
                attention.keys_state().is_some() == attention.values_state().is_some()
            }
            Self::Composite {
                convolution,
                recurrent,
            } => convolution.is_unallocated() == recurrent.is_unallocated(),
        }
    }

    /// The rolling convolution state of a hybrid layer: a lazy copy that
    /// branches from the layer's input, so the layer output never reaches it
    /// and evaluating it is the only way to release that input.
    #[must_use]
    pub fn convolution_state_array(&self) -> Option<&mlx_bridge::MlxArray> {
        match self {
            Self::AppendOnlyAttention { .. } => None,
            Self::Composite { convolution, .. } => convolution.state(),
        }
    }

    /// Returns logical payload bytes owned by this state family.
    #[must_use]
    pub fn payload_byte_count(&self) -> u64 {
        match self {
            Self::AppendOnlyAttention { attention } => attention.payload_byte_count(),
            Self::Composite {
                convolution,
                recurrent,
            } => convolution
                .payload_byte_count()
                .saturating_add(recurrent.payload_byte_count()),
        }
    }

    /// Captures this layer's current logical state for transactional rollback.
    pub fn checkpoint(&self) -> Result<AstroDecoderStateCacheStateCheckpoint, MlxRuntimeError> {
        match self {
            Self::AppendOnlyAttention { attention } => {
                Ok(AstroDecoderStateCacheStateCheckpoint::AppendOnlyAttention {
                    offset_tokens: attention.offset_tokens(),
                })
            }
            Self::Composite {
                convolution,
                recurrent,
            } => Ok(AstroDecoderStateCacheStateCheckpoint::Composite {
                convolution: convolution.checkpoint()?,
                recurrent: recurrent.checkpoint()?,
            }),
        }
    }

    /// Retains physical state owners for an operation that may need a full retry.
    pub fn allocation_checkpoint(
        &self,
    ) -> Result<AstroDecoderStateCacheStateAllocationCheckpoint, MlxRuntimeError> {
        match self {
            Self::AppendOnlyAttention { attention } => Ok(
                AstroDecoderStateCacheStateAllocationCheckpoint::AppendOnlyAttention {
                    attention: attention.allocation_checkpoint()?,
                },
            ),
            Self::Composite {
                convolution,
                recurrent,
            } => Ok(AstroDecoderStateCacheStateAllocationCheckpoint::Composite {
                convolution: convolution.checkpoint()?,
                recurrent: recurrent.checkpoint()?,
            }),
        }
    }

    /// Restores this layer to a transactional rollback point.
    pub fn restore_checkpoint(
        &mut self,
        checkpoint: AstroDecoderStateCacheStateCheckpoint,
    ) -> Result<(), MlxRuntimeError> {
        match (self, checkpoint) {
            (
                Self::AppendOnlyAttention { attention },
                AstroDecoderStateCacheStateCheckpoint::AppendOnlyAttention { offset_tokens },
            ) => attention.truncate_to_offset(offset_tokens),
            (
                Self::Composite {
                    convolution,
                    recurrent,
                },
                AstroDecoderStateCacheStateCheckpoint::Composite {
                    convolution: convolution_checkpoint,
                    recurrent: recurrent_checkpoint,
                },
            ) => {
                convolution.restore_checkpoint(convolution_checkpoint);
                recurrent.restore_checkpoint(recurrent_checkpoint);
                Ok(())
            }
            _ => Err(MlxRuntimeError::RuntimeOperation {
                operation: "restore astro-decoder-state-cache state checkpoint",
                description: "checkpoint layer type does not match the decoder state".to_owned(),
            }),
        }
    }

    /// Replaces every state owner with the retained pre-operation owner.
    pub fn restore_allocation_checkpoint(
        &mut self,
        allocation_checkpoint: AstroDecoderStateCacheStateAllocationCheckpoint,
    ) -> Result<(), MlxRuntimeError> {
        match (self, allocation_checkpoint) {
            (
                Self::AppendOnlyAttention { attention },
                AstroDecoderStateCacheStateAllocationCheckpoint::AppendOnlyAttention {
                    attention: attention_allocation_checkpoint,
                },
            ) => attention.restore_allocation_checkpoint(attention_allocation_checkpoint),
            (
                Self::Composite {
                    convolution,
                    recurrent,
                },
                AstroDecoderStateCacheStateAllocationCheckpoint::Composite {
                    convolution: convolution_checkpoint,
                    recurrent: recurrent_checkpoint,
                },
            ) => {
                convolution.restore_checkpoint(convolution_checkpoint);
                recurrent.restore_checkpoint(recurrent_checkpoint);
                Ok(())
            }
            _ => Err(MlxRuntimeError::RuntimeOperation {
                operation: "restore astro-decoder-state-cache allocation checkpoint",
                description: "checkpoint layer type does not match the decoder state".to_owned(),
            }),
        }
    }
}
