//! Immutable storage and memory geometry for one model revision.
//!
//! Resolution happens once from validated decoder layout and live machine/user
//! budgets. Every later lookup, write, quota decision, and memory admission uses
//! this same contract so no component invents its own block size or tensor shape.

use sha2::{Digest, Sha256};

use crate::{DecoderCacheLayout, DecoderCachePersistedTensorLayout};

use super::block_format::PERSISTENT_PROMPT_CACHE_FORMAT_VERSION;
use super::model_contract_error::PersistentPromptCacheModelContractError;
use super::model_contract_storage_geometry::{
    exact_state_file_bytes, maximum_block_manifest_file_bytes,
};

/// Architecture-neutral decoder-state storage contract derived from validated model metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentPromptCacheModelContract {
    model_id: String,
    model_revision: String,
    decoder_cache_layout: DecoderCacheLayout,
    maximum_context_token_count: usize,
    effective_mlx_memory_ceiling_bytes: u64,
    block_token_count: usize,
    common_prefix_checkpoint_stride_blocks: u32,
    sequence_state_payload_bytes_per_token: usize,
    sequence_state_payload_bytes_per_block: usize,
    boundary_state_payload_bytes: usize,
    capture_payload_bytes: usize,
    sequence_state_file_bytes: u64,
    boundary_state_file_bytes: u64,
    maximum_block_manifest_file_bytes: u64,
    maximum_committed_block_bytes: u64,
    direct_publication_workspace_bytes: usize,
    storage_contract_fingerprint: [u8; 32],
}

impl PersistentPromptCacheModelContract {
    /// Resolves one deterministic exact-state storage policy for a loaded model and its budgets.
    pub fn resolve(
        model_id: String,
        model_revision: String,
        decoder_cache_layout: DecoderCacheLayout,
        maximum_context_token_count: usize,
        effective_mlx_memory_ceiling_bytes: u64,
        global_ssd_quota_bytes: u64,
        configured_block_token_count: Option<usize>,
        common_prefix_checkpoint_stride_blocks: u32,
    ) -> Result<Self, PersistentPromptCacheModelContractError> {
        if model_id.is_empty() {
            return Err(PersistentPromptCacheModelContractError::EmptyModelId);
        }
        if model_revision.is_empty() {
            return Err(PersistentPromptCacheModelContractError::EmptyModelRevision);
        }
        if maximum_context_token_count == 0 {
            return Err(PersistentPromptCacheModelContractError::ZeroMaximumContextTokenCount);
        }
        if common_prefix_checkpoint_stride_blocks == 0 {
            return Err(
                PersistentPromptCacheModelContractError::ZeroCommonPrefixCheckpointStrideBlocks,
            );
        }
        if !decoder_cache_layout.has_sequence_state() && !decoder_cache_layout.has_boundary_state()
        {
            return Err(PersistentPromptCacheModelContractError::NoPersistentState);
        }

        let sequence_state_payload_bytes_per_token =
            decoder_cache_layout.sequence_state_payload_byte_count_per_token()?;
        let boundary_state_payload_bytes =
            decoder_cache_layout.boundary_snapshot_payload_byte_count()?;
        let persistence_alignment_token_count =
            decoder_cache_layout.persistence_alignment_token_count()?;
        // The block length belongs to the immutable artifact contract, not a deployment-wide
        // tuning constant. Matching append-only allocation growth avoids capture boundaries that
        // force incompatible state reshaping, while the quota-derived size prevents a laptop
        // with less SSD capacity from accepting a state block it can never retain.
        let block_token_count_is_user_configured = configured_block_token_count.is_some();
        let mut block_token_count = match configured_block_token_count {
            Some(configured_block_token_count)
                if configured_block_token_count > 0
                    && configured_block_token_count <= maximum_context_token_count
                    && configured_block_token_count
                        .is_multiple_of(persistence_alignment_token_count) =>
            {
                configured_block_token_count
            }
            Some(_) => {
                return Err(
                    PersistentPromptCacheModelContractError::InvalidConfiguredBlockTokenCount {
                        required_alignment_tokens: persistence_alignment_token_count,
                        maximum_context_tokens: maximum_context_token_count,
                    },
                );
            }
            None => derive_block_token_count(
                maximum_context_token_count,
                persistence_alignment_token_count,
                sequence_state_payload_bytes_per_token,
                boundary_state_payload_bytes,
                global_ssd_quota_bytes,
            )?,
        };
        let maximum_block_manifest_file_bytes =
            maximum_block_manifest_file_bytes(maximum_context_token_count)?;
        // Automatic sizing may need a larger aligned block to amortize repeated
        // manifests and recurrent snapshots across a maximum-length chain. An
        // explicit user block length is different: it is part of the requested
        // storage topology and must either fit exactly or fail clearly. Silently
        // changing it would make status, fingerprints, and observed boundaries
        // disagree with configuration.
        let (sequence_state_file_bytes, boundary_state_file_bytes, maximum_committed_block_bytes) = loop {
            let sequence_state_file_bytes = exact_state_file_bytes(
                block_token_count,
                decoder_cache_layout.sequence_tensor_layouts(),
            )?;
            let boundary_state_file_bytes = exact_state_file_bytes(
                block_token_count,
                decoder_cache_layout.boundary_tensor_layouts(),
            )?;
            let maximum_committed_block_bytes = sequence_state_file_bytes
                .checked_add(boundary_state_file_bytes)
                .and_then(|state_file_bytes| {
                    state_file_bytes.checked_add(maximum_block_manifest_file_bytes)
                })
                .ok_or(PersistentPromptCacheModelContractError::CapturePayloadByteCountOverflow)?;
            let maximum_committed_block_count =
                checked_ceiling_division(maximum_context_token_count, block_token_count)?;
            let maximum_chain_bytes = maximum_committed_block_bytes
                .checked_mul(u64::try_from(maximum_committed_block_count).unwrap_or(u64::MAX))
                .ok_or(PersistentPromptCacheModelContractError::CapturePayloadByteCountOverflow)?;
            if maximum_chain_bytes <= global_ssd_quota_bytes {
                break (
                    sequence_state_file_bytes,
                    boundary_state_file_bytes,
                    maximum_committed_block_bytes,
                );
            }
            if block_token_count_is_user_configured {
                return Err(
                    PersistentPromptCacheModelContractError::ConfiguredBlockChainExceedsSsdQuota {
                        configured_block_tokens: block_token_count,
                        maximum_chain_bytes,
                        global_ssd_quota_bytes,
                    },
                );
            }
            let next_block_token_count = block_token_count
                .checked_add(persistence_alignment_token_count)
                .map(|next_block_token_count| {
                    next_block_token_count.min(maximum_context_token_count)
                })
                .ok_or(PersistentPromptCacheModelContractError::CapturePayloadByteCountOverflow)?;
            if next_block_token_count == block_token_count {
                return Err(
                    PersistentPromptCacheModelContractError::BlockFilesExceedSsdQuota {
                        block_file_bytes: maximum_committed_block_bytes,
                        global_ssd_quota_bytes,
                    },
                );
            }
            block_token_count = next_block_token_count;
        };
        let sequence_state_payload_bytes_per_block = sequence_state_payload_bytes_per_token
            .checked_mul(block_token_count)
            .ok_or(
                PersistentPromptCacheModelContractError::SequenceStateBlockPayloadByteCountOverflow,
            )?;
        let capture_payload_bytes = sequence_state_payload_bytes_per_block
            .checked_add(boundary_state_payload_bytes)
            .ok_or(PersistentPromptCacheModelContractError::CapturePayloadByteCountOverflow)?;
        // Native safetensors publication evaluates and copies one tensor at a
        // time. Peak *additional* workspace is therefore the largest individual
        // tensor, not the sum of captured decoder state already owned by request.
        let direct_publication_workspace_bytes = decoder_cache_layout
            .maximum_sequence_tensor_payload_byte_count(block_token_count)?
            .max(
                decoder_cache_layout
                    .boundary_tensor_layouts()
                    .iter()
                    .map(|persisted_tensor_layout| {
                        persisted_tensor_layout
                            .tensor_layout()
                            .fixed_payload_byte_count()
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .max()
                    .unwrap_or(0),
            );
        // Captured arrays are existing decoder-state ownership. Direct publication adds only
        // the largest one-at-a-time contiguous tensor materialization proven by the MLX writer.
        let direct_publication_workspace_bytes_u64 =
            u64::try_from(direct_publication_workspace_bytes).unwrap_or(u64::MAX);
        if direct_publication_workspace_bytes_u64 > effective_mlx_memory_ceiling_bytes {
            return Err(
                PersistentPromptCacheModelContractError::CaptureExceedsMlxMemoryCeiling {
                    capture_memory_bytes: direct_publication_workspace_bytes_u64,
                    effective_mlx_memory_ceiling_bytes,
                },
            );
        }
        if maximum_committed_block_bytes > global_ssd_quota_bytes {
            return Err(
                PersistentPromptCacheModelContractError::BlockFilesExceedSsdQuota {
                    block_file_bytes: maximum_committed_block_bytes,
                    global_ssd_quota_bytes,
                },
            );
        }
        // The fingerprint is compatibility identity, not merely model identity.
        // Any layout, dtype, block size, retained-checkpoint stride, format,
        // model, or revision change must prevent old files from being joined to
        // a chain whose capture and retention topology differs.
        let storage_contract_fingerprint = storage_contract_fingerprint(
            &model_id,
            &model_revision,
            &decoder_cache_layout,
            block_token_count,
            common_prefix_checkpoint_stride_blocks,
        );

        Ok(Self {
            model_id,
            model_revision,
            decoder_cache_layout,
            maximum_context_token_count,
            effective_mlx_memory_ceiling_bytes,
            block_token_count,
            common_prefix_checkpoint_stride_blocks,
            sequence_state_payload_bytes_per_token,
            sequence_state_payload_bytes_per_block,
            boundary_state_payload_bytes,
            capture_payload_bytes,
            sequence_state_file_bytes,
            boundary_state_file_bytes,
            maximum_block_manifest_file_bytes,
            maximum_committed_block_bytes,
            direct_publication_workspace_bytes,
            storage_contract_fingerprint,
        })
    }

    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    #[must_use]
    pub fn model_revision(&self) -> &str {
        &self.model_revision
    }

    #[must_use]
    pub const fn decoder_cache_layout(&self) -> &DecoderCacheLayout {
        &self.decoder_cache_layout
    }

    #[must_use]
    pub const fn maximum_context_token_count(&self) -> usize {
        self.maximum_context_token_count
    }

    #[must_use]
    pub const fn block_token_count(&self) -> usize {
        self.block_token_count
    }

    #[must_use]
    pub const fn common_prefix_checkpoint_stride_blocks(&self) -> u32 {
        self.common_prefix_checkpoint_stride_blocks
    }

    #[must_use]
    pub const fn has_sequence_state(&self) -> bool {
        self.decoder_cache_layout.has_sequence_state()
    }

    #[must_use]
    pub const fn has_boundary_state(&self) -> bool {
        self.decoder_cache_layout.has_boundary_state()
    }

    #[must_use]
    pub const fn sequence_state_payload_bytes_per_token(&self) -> usize {
        self.sequence_state_payload_bytes_per_token
    }

    #[must_use]
    pub const fn sequence_state_payload_bytes_per_block(&self) -> usize {
        self.sequence_state_payload_bytes_per_block
    }

    #[must_use]
    pub const fn boundary_state_payload_bytes(&self) -> usize {
        self.boundary_state_payload_bytes
    }

    #[must_use]
    pub const fn capture_payload_bytes(&self) -> usize {
        self.capture_payload_bytes
    }

    #[must_use]
    pub const fn sequence_state_file_bytes(&self) -> u64 {
        self.sequence_state_file_bytes
    }

    #[must_use]
    pub const fn boundary_state_file_bytes(&self) -> u64 {
        self.boundary_state_file_bytes
    }

    #[must_use]
    pub const fn maximum_block_manifest_file_bytes(&self) -> u64 {
        self.maximum_block_manifest_file_bytes
    }

    #[must_use]
    pub const fn maximum_committed_block_bytes(&self) -> u64 {
        self.maximum_committed_block_bytes
    }

    #[must_use]
    pub const fn direct_publication_workspace_bytes(&self) -> usize {
        self.direct_publication_workspace_bytes
    }

    #[must_use]
    pub const fn effective_mlx_memory_ceiling_bytes(&self) -> u64 {
        self.effective_mlx_memory_ceiling_bytes
    }

    #[must_use]
    pub const fn storage_contract_fingerprint(&self) -> [u8; 32] {
        self.storage_contract_fingerprint
    }

    #[must_use]
    pub fn storage_contract_fingerprint_hex(&self) -> String {
        hex_encode(self.storage_contract_fingerprint)
    }
}

fn derive_block_token_count(
    maximum_context_token_count: usize,
    persistence_alignment_token_count: usize,
    sequence_state_payload_bytes_per_token: usize,
    boundary_state_payload_bytes: usize,
    global_ssd_quota_bytes: u64,
) -> Result<usize, PersistentPromptCacheModelContractError> {
    // Recurrent state is copied at every restorable boundary. With append-only state, choose a
    // block large enough to amortize that fixed capture; for recurrent-only models, distribute
    // snapshots across the context according to the available global SSD quota instead.
    let unaligned_block_token_count = if sequence_state_payload_bytes_per_token > 0 {
        if boundary_state_payload_bytes == 0 {
            persistence_alignment_token_count
        } else {
            checked_ceiling_division(
                boundary_state_payload_bytes,
                sequence_state_payload_bytes_per_token,
            )?
        }
    } else {
        if boundary_state_payload_bytes == 0 {
            return Err(PersistentPromptCacheModelContractError::NoPersistentState);
        }
        let maximum_boundary_snapshot_count = global_ssd_quota_bytes
            .checked_div(u64::try_from(boundary_state_payload_bytes).unwrap_or(u64::MAX))
            .unwrap_or(0);
        if maximum_boundary_snapshot_count == 0 {
            return Err(
                PersistentPromptCacheModelContractError::BoundarySnapshotExceedsSsdQuota {
                    boundary_snapshot_bytes: u64::try_from(boundary_state_payload_bytes)
                        .unwrap_or(u64::MAX),
                    global_ssd_quota_bytes,
                },
            );
        }
        checked_ceiling_division(
            maximum_context_token_count,
            usize::try_from(maximum_boundary_snapshot_count).unwrap_or(usize::MAX),
        )?
    };
    // Alignment is applied after the quota calculation so every full block can be restored into
    // the tensor capacities described by the layout. Context remains a hard upper bound.
    let aligned_block_token_count = checked_align_up(
        unaligned_block_token_count.max(1),
        persistence_alignment_token_count,
    )?;
    Ok(aligned_block_token_count.min(maximum_context_token_count))
}

fn checked_ceiling_division(
    dividend: usize,
    divisor: usize,
) -> Result<usize, PersistentPromptCacheModelContractError> {
    if divisor == 0 {
        return Err(PersistentPromptCacheModelContractError::ZeroStorageGeometryDivisor);
    }
    (dividend / divisor)
        .checked_add(usize::from(dividend % divisor != 0))
        .ok_or(PersistentPromptCacheModelContractError::BlockTokenCountOverflow)
}

fn checked_align_up(
    token_count: usize,
    alignment_token_count: usize,
) -> Result<usize, PersistentPromptCacheModelContractError> {
    let alignment_remainder = token_count % alignment_token_count;
    if alignment_remainder == 0 {
        return Ok(token_count);
    }
    token_count
        .checked_add(alignment_token_count - alignment_remainder)
        .ok_or(PersistentPromptCacheModelContractError::BlockTokenCountOverflow)
}

fn storage_contract_fingerprint(
    model_id: &str,
    model_revision: &str,
    decoder_cache_layout: &DecoderCacheLayout,
    block_token_count: usize,
    common_prefix_checkpoint_stride_blocks: u32,
) -> [u8; 32] {
    // Every variable-length field is length-prefixed to prevent concatenation
    // ambiguity (`ab` + `c` versus `a` + `bc`). Numeric fields use fixed-width
    // big-endian bytes so the digest is independent of host architecture.
    let mut fingerprint_digest = Sha256::new();
    update_length_prefixed_bytes(
        &mut fingerprint_digest,
        PERSISTENT_PROMPT_CACHE_FORMAT_VERSION.as_bytes(),
    );
    update_length_prefixed_bytes(&mut fingerprint_digest, model_id.as_bytes());
    update_length_prefixed_bytes(&mut fingerprint_digest, model_revision.as_bytes());
    fingerprint_digest.update((block_token_count as u64).to_be_bytes());
    fingerprint_digest.update(common_prefix_checkpoint_stride_blocks.to_be_bytes());
    update_tensor_layout_fingerprint(
        &mut fingerprint_digest,
        b"sequence",
        decoder_cache_layout.sequence_tensor_layouts(),
    );
    update_tensor_layout_fingerprint(
        &mut fingerprint_digest,
        b"boundary",
        decoder_cache_layout.boundary_tensor_layouts(),
    );
    fingerprint_digest.finalize().into()
}

fn update_tensor_layout_fingerprint(
    fingerprint_digest: &mut Sha256,
    state_kind_name: &[u8],
    persisted_tensor_layouts: Vec<DecoderCachePersistedTensorLayout>,
) {
    update_length_prefixed_bytes(fingerprint_digest, state_kind_name);
    fingerprint_digest.update((persisted_tensor_layouts.len() as u64).to_be_bytes());
    for persisted_tensor_layout in persisted_tensor_layouts {
        fingerprint_digest
            .update((persisted_tensor_layout.decoder_layer_index() as u64).to_be_bytes());
        let tensor_layout = persisted_tensor_layout.tensor_layout();
        update_length_prefixed_bytes(
            fingerprint_digest,
            tensor_layout.tensor_role_name().as_bytes(),
        );
        update_length_prefixed_bytes(
            fingerprint_digest,
            tensor_layout.dtype().safetensors_dtype_name().as_bytes(),
        );
        fingerprint_digest
            .update((tensor_layout.sequence_axis().unwrap_or(usize::MAX) as u64).to_be_bytes());
        fingerprint_digest.update((tensor_layout.dimensions().len() as u64).to_be_bytes());
        for tensor_dimension in tensor_layout.dimensions() {
            fingerprint_digest.update((*tensor_dimension as u64).to_be_bytes());
        }
    }
}

fn update_length_prefixed_bytes(fingerprint_digest: &mut Sha256, bytes: &[u8]) {
    fingerprint_digest.update((bytes.len() as u64).to_be_bytes());
    fingerprint_digest.update(bytes);
}

fn hex_encode(bytes: [u8; 32]) -> String {
    bytes
        .iter()
        .map(|fingerprint_byte| format!("{fingerprint_byte:02x}"))
        .collect()
}
