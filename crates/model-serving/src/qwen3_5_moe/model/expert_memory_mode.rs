use astronomical_ipc_protocol::ExpertMemoryMode;

use super::model::Qwen3_5MoEModel;

impl Qwen3_5MoEModel {
    /// Returns whether every sparse-expert layer is resident or paging remains necessary.
    #[must_use]
    pub fn expert_memory_mode(&self) -> ExpertMemoryMode {
        if self.expert_pager.is_none()
            || self
                .expert_weight_memory_cache
                .borrow()
                .has_complete_expert_layers_for_every_decoder_layer()
        {
            ExpertMemoryMode::Resident
        } else {
            ExpertMemoryMode::Paged
        }
    }

    /// Returns whether sparse-expert pages remain necessary for the next forward.
    #[must_use]
    pub(in crate::qwen3_5_moe) fn sparse_experts_are_paged(&self) -> bool {
        self.expert_memory_mode() == ExpertMemoryMode::Paged
    }
}
