//! Bounded reason-separated evidence for cleanup performed while opening the cache.

/// Count and byte total for one startup-cleanup reason.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PersistentPromptCacheStartupCleanupCategory {
    pub artifact_count: u64,
    pub block_count: u64,
    pub byte_count: u64,
}

impl PersistentPromptCacheStartupCleanupCategory {
    pub(crate) fn record_artifact(&mut self, removed_byte_count: u64) {
        self.artifact_count = self.artifact_count.saturating_add(1);
        self.byte_count = self.byte_count.saturating_add(removed_byte_count);
    }

    pub(crate) fn record_block(&mut self, removed_byte_count: u64) {
        self.block_count = self.block_count.saturating_add(1);
        self.byte_count = self.byte_count.saturating_add(removed_byte_count);
    }

    pub(crate) fn record_blocks(&mut self, removed_block_count: usize, removed_byte_count: u64) {
        self.block_count = self
            .block_count
            .saturating_add(u64::try_from(removed_block_count).unwrap_or(u64::MAX));
        self.byte_count = self.byte_count.saturating_add(removed_byte_count);
    }

    fn is_empty(self) -> bool {
        self.artifact_count == 0 && self.block_count == 0 && self.byte_count == 0
    }

    fn merge(&mut self, additional_category: Self) {
        self.artifact_count = self
            .artifact_count
            .saturating_add(additional_category.artifact_count);
        self.block_count = self
            .block_count
            .saturating_add(additional_category.block_count);
        self.byte_count = self
            .byte_count
            .saturating_add(additional_category.byte_count);
    }
}

/// Cleanup evidence retained until the first structural zero-restoration miss.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PersistentPromptCacheStartupCleanupEvidence {
    pub interrupted_transaction_recovery: PersistentPromptCacheStartupCleanupCategory,
    pub obsolete_format: PersistentPromptCacheStartupCleanupCategory,
    pub corrupt_current_format: PersistentPromptCacheStartupCleanupCategory,
    pub quota_eviction: PersistentPromptCacheStartupCleanupCategory,
}

impl PersistentPromptCacheStartupCleanupEvidence {
    pub(crate) fn merge(&mut self, additional_evidence: Self) {
        self.interrupted_transaction_recovery
            .merge(additional_evidence.interrupted_transaction_recovery);
        self.obsolete_format
            .merge(additional_evidence.obsolete_format);
        self.corrupt_current_format
            .merge(additional_evidence.corrupt_current_format);
        self.quota_eviction
            .merge(additional_evidence.quota_eviction);
    }

    pub(crate) fn into_non_empty(self) -> Option<Self> {
        (!self.is_empty()).then_some(self)
    }

    fn is_empty(self) -> bool {
        self.interrupted_transaction_recovery.is_empty()
            && self.obsolete_format.is_empty()
            && self.corrupt_current_format.is_empty()
            && self.quota_eviction.is_empty()
    }
}
