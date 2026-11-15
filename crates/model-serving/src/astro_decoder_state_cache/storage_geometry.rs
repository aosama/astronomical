use super::{
    AstroDecoderStateCacheLayerLayout, AstroDecoderStateCacheLayout,
    AstroDecoderStateCacheLayoutError,
};

impl AstroDecoderStateCacheLayout {
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
    ) -> Result<usize, AstroDecoderStateCacheLayoutError> {
        self.sequence_tensor_layouts().iter().try_fold(
            0_usize,
            |sequence_payload_bytes_per_token, persisted_tensor_layout| {
                sequence_payload_bytes_per_token
                    .checked_add(
                        persisted_tensor_layout
                            .tensor_layout()
                            .sequence_payload_byte_count_per_token()?,
                    )
                    .ok_or(AstroDecoderStateCacheLayoutError::SequenceStatePayloadByteCountPerTokenOverflow)
            },
        )
    }

    /// Returns the largest exact payload owned by one sequence tensor at the requested length.
    pub fn maximum_sequence_tensor_payload_byte_count(
        &self,
        sequence_token_count: usize,
    ) -> Result<usize, AstroDecoderStateCacheLayoutError> {
        self.sequence_tensor_layouts().iter().try_fold(
            0_usize,
            |maximum_tensor_payload_bytes, persisted_tensor_layout| {
                let tensor_payload_bytes = persisted_tensor_layout
                    .tensor_layout()
                    .sequence_payload_byte_count_per_token()?
                    .checked_mul(sequence_token_count)
                    .ok_or(
                        AstroDecoderStateCacheLayoutError::SequenceTensorPayloadByteCountOverflow,
                    )?;
                Ok(maximum_tensor_payload_bytes.max(tensor_payload_bytes))
            },
        )
    }

    /// Returns the largest source payload live during one incremental astro decoder state cache restore step.
    ///
    /// A restore loads one sequence block (every sequence tensor at block length) and,
    /// separately, the complete boundary snapshot (every boundary tensor). Because the two
    /// load at separate times, admission needs the larger source rather than the sum of both.
    pub fn incremental_restore_source_workspace_byte_count(
        &self,
        sequence_block_token_count: usize,
    ) -> Result<usize, AstroDecoderStateCacheLayoutError> {
        let sequence_source_bytes = self
            .sequence_state_payload_byte_count_per_token()?
            .checked_mul(sequence_block_token_count)
            .ok_or(AstroDecoderStateCacheLayoutError::SequenceTensorPayloadByteCountOverflow)?;
        let boundary_source_bytes = self.boundary_snapshot_payload_byte_count()?;
        Ok(sequence_source_bytes.max(boundary_source_bytes))
    }

    /// Returns payload bytes for one complete boundary snapshot.
    pub fn boundary_snapshot_payload_byte_count(
        &self,
    ) -> Result<usize, AstroDecoderStateCacheLayoutError> {
        self.boundary_tensor_layouts().iter().try_fold(
            0_usize,
            |boundary_payload_bytes, persisted_tensor_layout| {
                boundary_payload_bytes
                    .checked_add(
                        persisted_tensor_layout
                            .tensor_layout()
                            .fixed_payload_byte_count()?,
                    )
                    .ok_or(
                        AstroDecoderStateCacheLayoutError::BoundarySnapshotPayloadByteCountOverflow,
                    )
            },
        )
    }

    /// Returns the natural token alignment shared by every append-only state component.
    pub fn persistence_alignment_token_count(
        &self,
    ) -> Result<usize, AstroDecoderStateCacheLayoutError> {
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
    decoder_layer_layout: &AstroDecoderStateCacheLayerLayout,
    current_alignment_token_count: usize,
) -> Result<usize, AstroDecoderStateCacheLayoutError> {
    match decoder_layer_layout {
        AstroDecoderStateCacheLayerLayout::AppendOnlyAttention {
            capacity_growth_tokens,
            ..
        } => checked_least_common_multiple(current_alignment_token_count, *capacity_growth_tokens),
        AstroDecoderStateCacheLayerLayout::RotatingWindowAttention { .. } => {
            Ok(current_alignment_token_count)
        }
        AstroDecoderStateCacheLayerLayout::RecurrentTensor { .. } => {
            Ok(current_alignment_token_count)
        }
        AstroDecoderStateCacheLayerLayout::Composite { components } => components.iter().try_fold(
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
) -> Result<usize, AstroDecoderStateCacheLayoutError> {
    let greatest_common_divisor = greatest_common_divisor(first_token_count, second_token_count);
    first_token_count
        .checked_div(greatest_common_divisor)
        .and_then(|reduced_first_token_count| {
            reduced_first_token_count.checked_mul(second_token_count)
        })
        .ok_or(AstroDecoderStateCacheLayoutError::PersistenceAlignmentTokenCountOverflow)
}

fn greatest_common_divisor(mut first_token_count: usize, mut second_token_count: usize) -> usize {
    while second_token_count != 0 {
        let remainder_token_count = first_token_count % second_token_count;
        first_token_count = second_token_count;
        second_token_count = remainder_token_count;
    }
    first_token_count
}
