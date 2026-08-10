//! Stable, model-isolated hashing for persistent decoder-state blocks.

use sha2::{Digest, Sha256};

use super::model_contract::PersistentPromptCacheModelContract;

const PERSISTENT_PROMPT_CACHE_ROOT_SEED: &[u8] = b"astronomical-decoder-cache-root";

/// One immutable, content-addressed block identity inside a model-state chain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentPromptCacheBlockKey {
    block_hash: [u8; 32],
    block_index: u32,
    token_count: u32,
    block_token_count: usize,
    storage_contract_fingerprint: [u8; 32],
}

impl PersistentPromptCacheBlockKey {
    /// Hashes the first block under one exact model storage contract.
    pub fn for_root_block(
        persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
        block_tokens: &[u32],
    ) -> Result<Self, PersistentPromptCacheBlockKeyError> {
        Self::for_root_block_with_image_digests(
            persistent_prompt_cache_model_contract,
            block_tokens,
            &[],
        )
    }

    /// Hashes the first block while binding the ordered images in the prompt.
    pub fn for_root_block_with_image_digests(
        persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
        block_tokens: &[u32],
        ordered_image_sha256_digests: &[[u8; 32]],
    ) -> Result<Self, PersistentPromptCacheBlockKeyError> {
        let block_token_count = persistent_prompt_cache_model_contract.block_token_count();
        validate_block_tokens(block_tokens, block_token_count)?;
        // Persisted tensors are reusable only when their complete storage geometry agrees.
        // Carrying the contract fingerprint in the root digest prevents equal prompt tokens from
        // crossing model revisions, dtypes, layer layouts, or model-derived block lengths.
        let storage_contract_fingerprint =
            persistent_prompt_cache_model_contract.storage_contract_fingerprint();
        let block_hash = chain_hash_with_image_digests(
            None,
            &storage_contract_fingerprint,
            block_tokens,
            ordered_image_sha256_digests,
        );
        Ok(Self {
            block_hash,
            block_index: 0,
            token_count: u32::try_from(block_tokens.len())
                .map_err(|_| PersistentPromptCacheBlockKeyError::BlockTokenCountOverflow)?,
            block_token_count,
            storage_contract_fingerprint,
        })
    }

    /// Hashes the next block in the chain, carrying the complete storage identity forward.
    pub fn for_child_block(
        &self,
        block_tokens: &[u32],
    ) -> Result<Self, PersistentPromptCacheBlockKeyError> {
        validate_block_tokens(block_tokens, self.block_token_count)?;
        // Include the parent digest rather than only the child tokens so a shared suffix cannot
        // restore state produced by a divergent prompt prefix.
        let block_hash = chain_hash_with_image_digests(
            Some(&self.block_hash),
            &self.storage_contract_fingerprint,
            block_tokens,
            &[],
        );
        let next_block_index = self
            .block_index
            .checked_add(1)
            .ok_or(PersistentPromptCacheBlockKeyError::BlockIndexOverflow)?;
        Ok(Self {
            block_hash,
            block_index: next_block_index,
            token_count: u32::try_from(block_tokens.len())
                .map_err(|_| PersistentPromptCacheBlockKeyError::BlockTokenCountOverflow)?,
            block_token_count: self.block_token_count,
            storage_contract_fingerprint: self.storage_contract_fingerprint,
        })
    }

    #[must_use]
    pub const fn block_hash(&self) -> [u8; 32] {
        self.block_hash
    }

    #[must_use]
    pub const fn block_index(&self) -> u32 {
        self.block_index
    }

    #[must_use]
    pub const fn token_count(&self) -> usize {
        self.token_count as usize
    }

    #[must_use]
    pub const fn block_token_count(&self) -> usize {
        self.block_token_count
    }
}

fn validate_block_tokens(
    block_tokens: &[u32],
    block_token_count: usize,
) -> Result<(), PersistentPromptCacheBlockKeyError> {
    if block_tokens.is_empty() {
        return Err(PersistentPromptCacheBlockKeyError::EmptyBlockTokens);
    }
    if block_tokens.len() > block_token_count {
        return Err(
            PersistentPromptCacheBlockKeyError::BlockTokenCountExceedsBlock {
                actual_token_count: block_tokens.len(),
                maximum_token_count: block_token_count,
            },
        );
    }
    Ok(())
}

fn chain_hash_with_image_digests(
    parent_hash: Option<&[u8; 32]>,
    storage_contract_fingerprint: &[u8; 32],
    block_tokens: &[u32],
    ordered_image_sha256_digests: &[[u8; 32]],
) -> [u8; 32] {
    let mut block_digest = Sha256::new();
    if let Some(parent_hash) = parent_hash {
        block_digest.update(parent_hash);
    } else {
        block_digest.update(PERSISTENT_PROMPT_CACHE_ROOT_SEED);
    }
    block_digest.update(storage_contract_fingerprint);
    if parent_hash.is_none() && !ordered_image_sha256_digests.is_empty() {
        block_digest.update(b"astronomical-decoder-cache-prompt-attachments");
        block_digest.update((ordered_image_sha256_digests.len() as u64).to_be_bytes());
        for encoded_image_sha256_digest in ordered_image_sha256_digests {
            block_digest.update(encoded_image_sha256_digest);
        }
    }
    for block_token in block_tokens {
        block_digest.update(block_token.to_be_bytes());
    }
    block_digest.finalize().into()
}

/// A persistent model-state block identity could not be produced.
#[derive(Debug, thiserror::Error)]
pub enum PersistentPromptCacheBlockKeyError {
    #[error("persistent model-state block tokens must not be empty")]
    EmptyBlockTokens,
    #[error(
        "persistent model-state block has {actual_token_count} tokens, maximum {maximum_token_count}"
    )]
    BlockTokenCountExceedsBlock {
        actual_token_count: usize,
        maximum_token_count: usize,
    },
    #[error("persistent model-state block token count exceeds the u32 range")]
    BlockTokenCountOverflow,
    #[error("persistent model-state block index exceeds the u32 range")]
    BlockIndexOverflow,
}
