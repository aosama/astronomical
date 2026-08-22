//! Architecture-neutral causal input attached to one decoder-state cache block.
//!
//! Model families own the canonical bytes because only they understand which
//! non-token inputs enter the decoder at each prompt position. The persistent
//! cache treats those bytes as opaque identity and never interprets model syntax.

use sha2::{Digest, Sha256};

/// Canonical non-token causal input introduced by one prompt-cache block.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PersistentPromptCacheBlockCausalInput {
    canonical_digest: Option<[u8; 32]>,
}

impl PersistentPromptCacheBlockCausalInput {
    /// Represents a block whose decoder inputs are fully identified by its tokens and ancestry.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            canonical_digest: None,
        }
    }

    /// Reduces model-owned canonical identity to one bounded digest.
    #[must_use]
    pub fn from_canonical_bytes(canonical_bytes: &[u8]) -> Self {
        if canonical_bytes.is_empty() {
            return Self::empty();
        }
        Self {
            canonical_digest: Some(Sha256::digest(canonical_bytes).into()),
        }
    }

    /// Returns whether this block introduces no additional non-token input.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.canonical_digest.is_none()
    }

    pub(crate) const fn canonical_digest(&self) -> Option<&[u8; 32]> {
        self.canonical_digest.as_ref()
    }
}
