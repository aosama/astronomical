//! Stable identities for projected Qwen3.5-MoE visual embeddings.

use sha2::{Digest, Sha256};

/// Version of the persisted projected visual-embedding tensor contract.
pub const PERSISTENT_VISUAL_EMBEDDING_FORMAT_VERSION: &str = "2";

const VISUAL_EMBEDDING_HASH_DOMAIN: &[u8] = b"astronomical-qwen3-5-moe-visual-embedding";

/// Content identity for one exact encoded image's projected visual embeddings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PersistentVisualEmbeddingKey {
    visual_embedding_hash: [u8; 32],
    encoded_image_sha256: [u8; 32],
}

impl PersistentVisualEmbeddingKey {
    /// Creates the model- and format-isolated identity for one encoded image.
    ///
    /// The model ID and revision bind this visual embedding to the validated
    /// model namespace so a different model revision automatically produces a
    /// different hash without any code change.
    #[must_use]
    pub fn for_image(encoded_image_sha256: [u8; 32], model_id: &str, model_revision: &str) -> Self {
        let mut visual_embedding_hash_builder = Sha256::new();
        update_length_prefixed_bytes(
            &mut visual_embedding_hash_builder,
            VISUAL_EMBEDDING_HASH_DOMAIN,
        );
        update_length_prefixed_bytes(
            &mut visual_embedding_hash_builder,
            PERSISTENT_VISUAL_EMBEDDING_FORMAT_VERSION.as_bytes(),
        );
        update_length_prefixed_bytes(&mut visual_embedding_hash_builder, model_id.as_bytes());
        update_length_prefixed_bytes(
            &mut visual_embedding_hash_builder,
            model_revision.as_bytes(),
        );
        update_length_prefixed_bytes(&mut visual_embedding_hash_builder, &encoded_image_sha256);
        Self {
            visual_embedding_hash: visual_embedding_hash_builder.finalize().into(),
            encoded_image_sha256,
        }
    }

    /// Returns the 32-byte hash used as the visual file name.
    #[must_use]
    pub const fn visual_embedding_hash(&self) -> [u8; 32] {
        self.visual_embedding_hash
    }

    /// Returns the exact encoded-image digest bound to this visual identity.
    #[must_use]
    pub const fn encoded_image_sha256(&self) -> [u8; 32] {
        self.encoded_image_sha256
    }
}

fn update_length_prefixed_bytes(digest: &mut Sha256, byte_sequence: &[u8]) {
    digest.update((byte_sequence.len() as u64).to_be_bytes());
    digest.update(byte_sequence);
}
