use super::{DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCacheLayoutError};

impl DecoderCacheLayout {
    /// Returns whether this model persists append-only sequence state.
    #[must_use]
    pub const fn has_sequence_state(&self) -> bool {
        self.sequence_tensor_count() > 0
    }

    /// Returns whether this model persists complete-boundary state.
    #[must_use]
    pub const fn has_boundary_state(&self) -> bool {
        self.boundary_tensor_count() > 0
    }

    /// Returns the total exact sequence-state payload bytes added by one token.
    pub fn sequence_state_payload_byte_count_per_token(
        &self,
    ) -> Result<usize, DecoderCacheLayoutError> {
        self.sequence_tensor_layouts().iter().try_fold(
            0_usize,
            |sequence_payload_bytes_per_token, persisted_tensor_layout| {
                sequence_payload_bytes_per_token
                    .checked_add(
                        persisted_tensor_layout
                            .tensor_layout()
                            .sequence_payload_byte_count_per_token()?,
                    )
                    .ok_or(DecoderCacheLayoutError::SequenceStatePayloadByteCountPerTokenOverflow)
            },
        )
    }

    /// Returns the largest exact payload owned by one sequence tensor at the requested length.
    pub fn maximum_sequence_tensor_payload_byte_count(
        &self,
        sequence_token_count: usize,
    ) -> Result<usize, DecoderCacheLayoutError> {
        self.sequence_tensor_layouts().iter().try_fold(
            0_usize,
            |maximum_tensor_payload_bytes, persisted_tensor_layout| {
                let tensor_payload_bytes = persisted_tensor_layout
                    .tensor_layout()
                    .sequence_payload_byte_count_per_token()?
                    .checked_mul(sequence_token_count)
                    .ok_or(DecoderCacheLayoutError::SequenceTensorPayloadByteCountOverflow)?;
                Ok(maximum_tensor_payload_bytes.max(tensor_payload_bytes))
            },
        )
    }

    /// Returns the largest source payload live during one incremental cache restore step.
    ///
    /// Sequence blocks and the boundary snapshot load at separate times, so admission needs the
    /// larger source rather than the sum of both.
    pub fn incremental_restore_source_workspace_byte_count(
        &self,
        sequence_block_token_count: usize,
    ) -> Result<usize, DecoderCacheLayoutError> {
        let sequence_source_pair_bytes = self
            .maximum_sequence_tensor_payload_byte_count(sequence_block_token_count)?
            .checked_mul(2)
            .ok_or(DecoderCacheLayoutError::SequenceTensorPayloadByteCountOverflow)?;
        let maximum_boundary_tensor_bytes = self.boundary_tensor_layouts().iter().try_fold(
            0_usize,
            |maximum_tensor_bytes, persisted_tensor_layout| {
                persisted_tensor_layout
                    .tensor_layout()
                    .fixed_payload_byte_count()
                    .map(|tensor_bytes| maximum_tensor_bytes.max(tensor_bytes))
            },
        )?;
        let paired_boundary_tensor_bytes = maximum_boundary_tensor_bytes
            .checked_mul(2)
            .and_then(|paired_tensor_bytes| {
                std::mem::size_of::<f32>()
                    .checked_mul(2)
                    .and_then(|boundary_metadata_bytes| {
                        paired_tensor_bytes.checked_add(boundary_metadata_bytes)
                    })
            })
            .ok_or(DecoderCacheLayoutError::BoundarySnapshotPayloadByteCountOverflow)?;
        Ok(sequence_source_pair_bytes.max(paired_boundary_tensor_bytes))
    }

    /// Returns payload bytes for one complete boundary snapshot.
    pub fn boundary_snapshot_payload_byte_count(&self) -> Result<usize, DecoderCacheLayoutError> {
        self.boundary_tensor_layouts().iter().try_fold(
            0_usize,
            |boundary_payload_bytes, persisted_tensor_layout| {
                boundary_payload_bytes
                    .checked_add(
                        persisted_tensor_layout
                            .tensor_layout()
                            .fixed_payload_byte_count()?,
                    )
                    .ok_or(DecoderCacheLayoutError::BoundarySnapshotPayloadByteCountOverflow)
            },
        )
    }

    /// Returns payload bytes for one complete persistent model-state capture.
    pub fn persistent_prompt_cache_block_payload_byte_count(
        &self,
        block_token_count: usize,
    ) -> Result<usize, DecoderCacheLayoutError> {
        let sequence_payload_bytes = self
            .sequence_state_payload_byte_count_per_token()?
            .checked_mul(block_token_count)
            .ok_or(DecoderCacheLayoutError::PersistentPromptCacheBlockPayloadByteCountOverflow)?;
        sequence_payload_bytes
            .checked_add(self.boundary_snapshot_payload_byte_count()?)
            .ok_or(DecoderCacheLayoutError::PersistentPromptCacheBlockPayloadByteCountOverflow)
    }

    /// Returns the natural token alignment shared by every append-only state component.
    pub fn persistence_alignment_token_count(&self) -> Result<usize, DecoderCacheLayoutError> {
        let mut persistence_alignment_token_count = 1_usize;
        for decoder_layer_index in 0..self.layer_count() {
            if let Some(decoder_layer_layout) = self.layer(decoder_layer_index) {
                persistence_alignment_token_count = checked_layer_persistence_alignment(
                    decoder_layer_layout,
                    persistence_alignment_token_count,
                )?;
            }
        }
        Ok(persistence_alignment_token_count)
    }
}

fn checked_layer_persistence_alignment(
    decoder_layer_layout: &DecoderCacheLayerLayout,
    current_alignment_token_count: usize,
) -> Result<usize, DecoderCacheLayoutError> {
    match decoder_layer_layout {
        DecoderCacheLayerLayout::AppendOnlyAttention {
            capacity_growth_tokens,
            ..
        } => checked_least_common_multiple(current_alignment_token_count, *capacity_growth_tokens),
        DecoderCacheLayerLayout::RotatingWindowAttention { .. } => {
            Ok(current_alignment_token_count)
        }
        DecoderCacheLayerLayout::RecurrentTensor { .. } => Ok(current_alignment_token_count),
        DecoderCacheLayerLayout::Composite { components } => components.iter().try_fold(
            current_alignment_token_count,
            |component_alignment_token_count, component_layout| {
                checked_layer_persistence_alignment(
                    component_layout,
                    component_alignment_token_count,
                )
            },
        ),
    }
}

fn checked_least_common_multiple(
    first_token_count: usize,
    second_token_count: usize,
) -> Result<usize, DecoderCacheLayoutError> {
    let greatest_common_divisor = greatest_common_divisor(first_token_count, second_token_count);
    first_token_count
        .checked_div(greatest_common_divisor)
        .and_then(|reduced_first_token_count| {
            reduced_first_token_count.checked_mul(second_token_count)
        })
        .ok_or(DecoderCacheLayoutError::PersistenceAlignmentTokenCountOverflow)
}

fn greatest_common_divisor(mut first_token_count: usize, mut second_token_count: usize) -> usize {
    while second_token_count != 0 {
        let remainder_token_count = first_token_count % second_token_count;
        first_token_count = second_token_count;
        second_token_count = remainder_token_count;
    }
    first_token_count
}
