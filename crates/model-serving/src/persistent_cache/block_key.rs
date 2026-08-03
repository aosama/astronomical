//! Stable, model-isolated hashing for persistent decoder-cache blocks.

use sha2::{Digest, Sha256};

use super::block_format::PERSISTENT_PROMPT_CACHE_FORMAT_VERSION;

/// Number of prompt tokens captured by one persisted Qwen3.5-MoE prompt-cache block.
///
/// 2,048 tokens balance useful prefix reuse against repeated Qwen3.5-MoE
/// GatedDeltaNet boundary snapshots. Smaller 256-token SSD blocks duplicate that
/// large Float32 state eight times more often and fill a single-user laptop cache
/// with snapshots instead of useful prefix history. This value is part of the
/// persistent file contract: changing it requires invalidating or versioning old
/// blocks.
pub const PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT: usize = 2_048;

/// Fixed root seed so identical prompt content still hashes differently
/// across distinct products and across unrelated cache implementations.
/// This is namespacing, not a secret or authentication mechanism.
const PERSISTENT_PROMPT_CACHE_ROOT_SEED: &[u8] = b"astronomical-decoder-cache-root";

/// One immutable, content-addressed block identity inside the Qwen3.5-MoE prompt-cache chain.
///
/// Each child hash binds its parent, so requests sharing an initial block sequence
/// reuse that prefix while divergent suffixes cannot collide. The hash also binds
/// the validated model ID and revision so a stale prompt-cache directory from a
/// previous model or execution-math version can never be loaded as current.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentPromptCacheBlockKey {
    block_hash: [u8; 32],
    block_index: u32,
    token_count: u32,
    model_id: String,
    model_revision: String,
}

impl PersistentPromptCacheBlockKey {
    /// Hashes the first block in a fresh prompt, rooted at the validated model identity.
    pub fn for_root_block(
        model_id: &str,
        model_revision: &str,
        block_tokens: &[u32],
    ) -> Result<Self, PersistentPromptCacheBlockKeyError> {
        // The root accepts the identity explicitly so tests can prove model
        // isolation. Production passes the validated model identity.
        validate_block_tokens(block_tokens)?;
        let block_hash = chain_hash(None, model_id, model_revision, block_tokens);
        Ok(Self {
            block_hash,
            block_index: 0,
            token_count: u32::try_from(block_tokens.len())
                .map_err(|_| PersistentPromptCacheBlockKeyError::BlockTokenCountOverflow)?,
            model_id: model_id.to_owned(),
            model_revision: model_revision.to_owned(),
        })
    }

    /// Hashes the first block while binding the ordered images in the prompt.
    pub fn for_root_block_with_image_digests(
        model_id: &str,
        model_revision: &str,
        block_tokens: &[u32],
        ordered_image_sha256_digests: &[[u8; 32]],
    ) -> Result<Self, PersistentPromptCacheBlockKeyError> {
        validate_block_tokens(block_tokens)?;
        let block_hash = chain_hash_with_image_digests(
            None,
            model_id,
            model_revision,
            block_tokens,
            ordered_image_sha256_digests,
        );
        Ok(Self {
            block_hash,
            block_index: 0,
            token_count: u32::try_from(block_tokens.len())
                .map_err(|_| PersistentPromptCacheBlockKeyError::BlockTokenCountOverflow)?,
            model_id: model_id.to_owned(),
            model_revision: model_revision.to_owned(),
        })
    }

    /// Hashes the next block in the chain, carrying the parent identity forward.
    pub fn for_child_block(
        &self,
        block_tokens: &[u32],
    ) -> Result<Self, PersistentPromptCacheBlockKeyError> {
        // A child carries forward the parent's complete content history,
        // including the model identity. This ensures a future model revision
        // loaded with the same code automatically produces a different hash
        // namespace without any code change.
        validate_block_tokens(block_tokens)?;
        let block_hash = chain_hash(
            Some(&self.block_hash),
            &self.model_id,
            &self.model_revision,
            block_tokens,
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
            model_id: self.model_id.clone(),
            model_revision: self.model_revision.clone(),
        })
    }

    /// Returns the 32-byte content hash used as the on-disk file name.
    #[must_use]
    pub fn block_hash(&self) -> [u8; 32] {
        self.block_hash
    }

    /// Returns the zero-based position of this block within its chain.
    #[must_use]
    pub fn block_index(&self) -> u32 {
        self.block_index
    }

    /// Returns the number of prompt tokens captured by this block.
    #[must_use]
    pub fn token_count(&self) -> usize {
        self.token_count as usize
    }
}

fn validate_block_tokens(block_tokens: &[u32]) -> Result<(), PersistentPromptCacheBlockKeyError> {
    // Empty blocks would make a chain position exist without advancing prompt
    // state. Partial nonempty blocks are valid identities, although the engine
    // persists only complete boundary blocks.
    if block_tokens.is_empty() {
        return Err(PersistentPromptCacheBlockKeyError::EmptyBlockTokens);
    }
    if block_tokens.len() > PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT {
        return Err(
            PersistentPromptCacheBlockKeyError::BlockTokenCountExceedsBlock {
                actual_token_count: block_tokens.len(),
                maximum_token_count: PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT,
            },
        );
    }
    Ok(())
}

fn chain_hash(
    parent_hash: Option<&[u8; 32]>,
    model_id: &str,
    model_revision: &str,
    block_tokens: &[u32],
) -> [u8; 32] {
    chain_hash_with_image_digests(parent_hash, model_id, model_revision, block_tokens, &[])
}

fn chain_hash_with_image_digests(
    parent_hash: Option<&[u8; 32]>,
    model_id: &str,
    model_revision: &str,
    block_tokens: &[u32],
    ordered_image_sha256_digests: &[[u8; 32]],
) -> [u8; 32] {
    let mut digest = Sha256::new();
    if let Some(parent) = parent_hash {
        digest.update(parent);
    } else {
        digest.update(PERSISTENT_PROMPT_CACHE_ROOT_SEED);
    }
    digest.update(PERSISTENT_PROMPT_CACHE_FORMAT_VERSION.as_bytes());
    digest.update(model_id.as_bytes());
    digest.update(model_revision.as_bytes());
    if parent_hash.is_none() && !ordered_image_sha256_digests.is_empty() {
        digest.update(b"astronomical-decoder-cache-prompt-attachments");
        digest.update((ordered_image_sha256_digests.len() as u64).to_be_bytes());
        for encoded_image_sha256 in ordered_image_sha256_digests {
            digest.update(encoded_image_sha256);
        }
    }
    // Big-endian token encoding keeps on-disk hashes stable across machines
    // regardless of the host byte order, which matters because the cache
    // directory may move between laptops.
    for token in block_tokens {
        digest.update(token.to_be_bytes());
    }
    digest.finalize().into()
}

/// A persistent prompt-cache block identity could not be produced for the requested block.
#[derive(Debug, thiserror::Error)]
pub enum PersistentPromptCacheBlockKeyError {
    #[error("qwen3.5-moe persistent prompt-cache block tokens must not be empty")]
    EmptyBlockTokens,
    #[error(
        "qwen3.5-moe persistent prompt-cache block has {actual_token_count} tokens, maximum {maximum_token_count}"
    )]
    BlockTokenCountExceedsBlock {
        actual_token_count: usize,
        maximum_token_count: usize,
    },
    #[error("qwen3.5-moe persistent prompt-cache block token count exceeds the u32 range")]
    BlockTokenCountOverflow,
    #[error("qwen3.5-moe persistent prompt-cache block index exceeds the u32 range")]
    BlockIndexOverflow,
}
