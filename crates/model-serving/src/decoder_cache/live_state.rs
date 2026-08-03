use astronomical_runtime_integration::MlxRuntimeError;

use super::{
    ConvolutionState, ConvolutionStateCheckpoint, FullAttentionKeyValueState,
    FullAttentionKeyValueStateAllocationCheckpoint, GatedDeltaRecurrentState,
    GatedDeltaRecurrentStateCheckpoint,
};

/// One architecture-neutral in-memory decoder-cache state family.
pub enum DecoderCacheState {
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

/// Logical rollback point for one architecture-neutral decoder-cache state entry.
pub enum DecoderCacheStateCheckpoint {
    AppendOnlyAttention {
        offset_tokens: i32,
    },
    Composite {
        convolution: ConvolutionStateCheckpoint,
        recurrent: GatedDeltaRecurrentStateCheckpoint,
    },
}

/// Physical owner checkpoint for retrying one decoder-cache update.
pub enum DecoderCacheStateAllocationCheckpoint {
    AppendOnlyAttention {
        attention: FullAttentionKeyValueStateAllocationCheckpoint,
    },
    Composite {
        convolution: ConvolutionStateCheckpoint,
        recurrent: GatedDeltaRecurrentStateCheckpoint,
    },
}

impl DecoderCacheState {
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

    /// Captures this layer's current logical state for MTP rollback.
    pub fn checkpoint(&self) -> Result<DecoderCacheStateCheckpoint, MlxRuntimeError> {
        match self {
            Self::AppendOnlyAttention { attention } => {
                Ok(DecoderCacheStateCheckpoint::AppendOnlyAttention {
                    offset_tokens: attention.offset_tokens(),
                })
            }
            Self::Composite {
                convolution,
                recurrent,
            } => Ok(DecoderCacheStateCheckpoint::Composite {
                convolution: convolution.checkpoint()?,
                recurrent: recurrent.checkpoint()?,
            }),
        }
    }

    /// Retains physical state owners for an operation that may need a full retry.
    pub fn allocation_checkpoint(
        &self,
    ) -> Result<DecoderCacheStateAllocationCheckpoint, MlxRuntimeError> {
        match self {
            Self::AppendOnlyAttention { attention } => {
                Ok(DecoderCacheStateAllocationCheckpoint::AppendOnlyAttention {
                    attention: attention.allocation_checkpoint()?,
                })
            }
            Self::Composite {
                convolution,
                recurrent,
            } => Ok(DecoderCacheStateAllocationCheckpoint::Composite {
                convolution: convolution.checkpoint()?,
                recurrent: recurrent.checkpoint()?,
            }),
        }
    }

    /// Restores this layer to an MTP rollback point.
    pub fn restore_checkpoint(
        &mut self,
        checkpoint: DecoderCacheStateCheckpoint,
    ) -> Result<(), MlxRuntimeError> {
        match (self, checkpoint) {
            (
                Self::AppendOnlyAttention { attention },
                DecoderCacheStateCheckpoint::AppendOnlyAttention { offset_tokens },
            ) => attention.truncate_to_offset(offset_tokens),
            (
                Self::Composite {
                    convolution,
                    recurrent,
                },
                DecoderCacheStateCheckpoint::Composite {
                    convolution: convolution_checkpoint,
                    recurrent: recurrent_checkpoint,
                },
            ) => {
                convolution.restore_checkpoint(convolution_checkpoint);
                recurrent.restore_checkpoint(recurrent_checkpoint);
                Ok(())
            }
            _ => Err(MlxRuntimeError::RuntimeOperation {
                operation: "restore decoder-cache state checkpoint",
                description: "checkpoint layer type does not match the decoder state".to_owned(),
            }),
        }
    }

    /// Replaces every state owner with the retained pre-operation owner.
    pub fn restore_allocation_checkpoint(
        &mut self,
        allocation_checkpoint: DecoderCacheStateAllocationCheckpoint,
    ) -> Result<(), MlxRuntimeError> {
        match (self, allocation_checkpoint) {
            (
                Self::AppendOnlyAttention { attention },
                DecoderCacheStateAllocationCheckpoint::AppendOnlyAttention {
                    attention: attention_allocation_checkpoint,
                },
            ) => attention.restore_allocation_checkpoint(attention_allocation_checkpoint),
            (
                Self::Composite {
                    convolution,
                    recurrent,
                },
                DecoderCacheStateAllocationCheckpoint::Composite {
                    convolution: convolution_checkpoint,
                    recurrent: recurrent_checkpoint,
                },
            ) => {
                convolution.restore_checkpoint(convolution_checkpoint);
                recurrent.restore_checkpoint(recurrent_checkpoint);
                Ok(())
            }
            _ => Err(MlxRuntimeError::RuntimeOperation {
                operation: "restore decoder-cache allocation checkpoint",
                description: "checkpoint layer type does not match the decoder state".to_owned(),
            }),
        }
    }
}
