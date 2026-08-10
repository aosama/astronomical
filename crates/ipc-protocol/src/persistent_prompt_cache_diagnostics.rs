//! Bounded per-request evidence explaining persistent prompt-cache behavior.
//!
//! These types cross the worker/supervisor boundary and can enter local JSONL
//! performance logs. They intentionally contain counters, enums, and a short hash
//! prefix only—never prompts, complete hashes, model paths, or tensor contents.

use serde::{Deserialize, Deserializer, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerPersistentPromptCacheLookupOutcome {
    Hit,
    Miss,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerPersistentPromptCacheMissReason {
    /// The prompt does not contain one complete cache block.
    PromptTooShortForPersistentPromptCache,
    /// The content-addressed root sequence block was not durable.
    RootSequenceStateBlockMissing,
    /// Sequence ancestry matched, but no required restart boundary was available.
    BoundaryStateSnapshotMissing,
}

/// Count and byte total for one startup-cleanup reason.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerPersistentPromptCacheStartupCleanupCategory {
    /// Standalone files removed for this reason.
    pub artifact_count: u64,
    /// Atomic block directories removed for this reason.
    pub block_count: u64,
    /// Total non-directory bytes removed for this reason.
    pub byte_count: u64,
}

/// Bounded reason-separated evidence retained from prompt-cache startup.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerPersistentPromptCacheStartupCleanupEvidence {
    pub interrupted_transaction_recovery: WorkerPersistentPromptCacheStartupCleanupCategory,
    pub obsolete_format: WorkerPersistentPromptCacheStartupCleanupCategory,
    pub corrupt_current_format: WorkerPersistentPromptCacheStartupCleanupCategory,
    pub quota_eviction: WorkerPersistentPromptCacheStartupCleanupCategory,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct WorkerPersistentPromptCacheExpectedBlockHashPrefix(String);

impl WorkerPersistentPromptCacheExpectedBlockHashPrefix {
    #[must_use]
    pub fn from_block_hash(block_hash: [u8; 32]) -> Self {
        // Eight bytes (16 hexadecimal characters) are enough to correlate local
        // diagnostics while avoiding publication of complete content identity.
        Self(
            block_hash
                .iter()
                .take(8)
                .map(|block_hash_byte| format!("{block_hash_byte:02x}"))
                .collect(),
        )
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for WorkerPersistentPromptCacheExpectedBlockHashPrefix {
    fn deserialize<DeserializerType>(
        deserializer: DeserializerType,
    ) -> Result<Self, DeserializerType::Error>
    where
        DeserializerType: Deserializer<'de>,
    {
        let hash_prefix = String::deserialize(deserializer)?;
        if hash_prefix.len() != 16
            || !hash_prefix.bytes().all(|hash_character| {
                hash_character.is_ascii_digit() || (b'a'..=b'f').contains(&hash_character)
            })
        {
            return Err(serde::de::Error::custom(
                "expected block hash prefix must be 16 lowercase hexadecimal characters",
            ));
        }
        Ok(Self(hash_prefix))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerPersistentPromptCacheRequestDiagnostics {
    /// Final user-visible classification after longest-prefix lookup.
    pub lookup_outcome: WorkerPersistentPromptCacheLookupOutcome,
    /// Immutable model-derived block geometry used for this request.
    pub block_token_count: u64,
    /// Number of complete blocks represented by the prompt.
    pub complete_prompt_block_count: u64,
    /// Complete blocks eligible after retaining the final generation-start token.
    pub maximum_restorable_block_count: u64,
    /// Consecutive sequence-state blocks found from root.
    pub matched_sequence_state_block_count: u64,
    /// Blocks actually reconstructed after boundary selection.
    pub restored_block_count: u64,
    /// First expected sequence block absent from the durable chain, if any.
    pub first_missing_sequence_state_block_index: Option<u64>,
    pub miss_reason: Option<WorkerPersistentPromptCacheMissReason>,
    /// Bounded correlation hint for the first expected missing block.
    pub expected_block_hash_prefix: Option<WorkerPersistentPromptCacheExpectedBlockHashPrefix>,
    /// Startup cleanup that can explain this first structural cold miss.
    pub startup_cleanup_evidence: Option<WorkerPersistentPromptCacheStartupCleanupEvidence>,
    /// Blocks physically committed by this request; idempotent reuse is excluded.
    pub published_block_count: u64,
    /// Allocator cache released before direct tensor materialization.
    pub allocator_bytes_cleared_for_publication: u64,
    /// Pageable expert payload evicted to satisfy publication memory pressure.
    pub expert_bytes_reclaimed_for_publication: u64,
}

impl WorkerPersistentPromptCacheRequestDiagnostics {
    pub fn record_published_block(&mut self) {
        self.published_block_count = self.published_block_count.saturating_add(1);
    }
}
