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
