//! Canonical Laguna execution profile used to isolate optimizer measurements.

use sha2::{Digest, Sha256};

use astronomical_ipc_protocol::ExpertMemoryMode;

use crate::laguna::LagunaTargetContract;

/// Stable digest of Laguna geometry, cache topology, storage, and residency.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LagunaPromptProcessingExecutionProfile {
    execution_profile_digest: u64,
    are_sparse_experts_paged: bool,
    is_prompt_cache_capture_eligible: bool,
}

impl LagunaPromptProcessingExecutionProfile {
    /// Digests canonical descriptors without repo names or raw tensor namespaces.
    #[must_use]
    pub fn from_canonical_descriptors(
        target_contract: &LagunaTargetContract,
        storage_fingerprint: &[u8; 32],
        expert_memory_mode: ExpertMemoryMode,
        is_prompt_cache_capture_eligible: bool,
    ) -> Self {
        let are_sparse_experts_paged = !matches!(expert_memory_mode, ExpertMemoryMode::Resident);
        let mut profile_hasher = Sha256::new();
        profile_hasher.update(b"astronomical-laguna-prompt-profile-v1");
        profile_hasher.update(format!("{:?}", target_contract.model()).as_bytes());
        for layer_descriptor in target_contract.layers() {
            profile_hasher.update(format!("{:?}", layer_descriptor.attention().cache()).as_bytes());
            profile_hasher.update(format!("{:?}", layer_descriptor.feed_forward()).as_bytes());
        }
        profile_hasher.update(storage_fingerprint);
        profile_hasher.update([match expert_memory_mode {
            ExpertMemoryMode::Resident => 0,
            ExpertMemoryMode::Hybrid => 1,
            ExpertMemoryMode::Paged => 2,
        }]);
        let digest_bytes = profile_hasher.finalize();
        let execution_profile_digest = u64::from_be_bytes(
            digest_bytes[..8]
                .try_into()
                .unwrap_or([0, 0, 0, 0, 0, 0, 0, 0]),
        );
        Self {
            execution_profile_digest,
            are_sparse_experts_paged,
            is_prompt_cache_capture_eligible,
        }
    }

    #[must_use]
    pub const fn execution_profile_digest(self) -> u64 {
        self.execution_profile_digest
    }

    #[must_use]
    pub const fn are_sparse_experts_paged(self) -> bool {
        self.are_sparse_experts_paged
    }

    #[must_use]
    pub const fn is_prompt_cache_capture_eligible(self) -> bool {
        self.is_prompt_cache_capture_eligible
    }

    #[must_use]
    pub const fn with_prompt_cache_capture_eligible(
        mut self,
        is_prompt_cache_capture_eligible: bool,
    ) -> Self {
        self.is_prompt_cache_capture_eligible = is_prompt_cache_capture_eligible;
        self
    }
}
