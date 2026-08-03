use astronomical_ipc_protocol::ExpertMemoryMode;

use crate::qwen3_5::model::Qwen3_5Model;

impl Qwen3_5Model {
    /// Returns whether every sparse-expert layer is resident or paging remains necessary.
    #[must_use]
    pub fn expert_memory_mode(&self) -> ExpertMemoryMode {
        let Some(expert_weight_memory_cache) = self.expert_weight_memory_cache.as_ref() else {
            return ExpertMemoryMode::Resident;
        };
        if self.expert_pager.is_some()
            && !expert_weight_memory_cache
                .borrow()
                .has_complete_expert_layers_for_every_decoder_layer()
        {
            ExpertMemoryMode::Paged
        } else {
            ExpertMemoryMode::Resident
        }
    }

    /// Returns whether sparse-expert pages remain necessary for the next forward.
    #[must_use]
    pub(crate) fn sparse_experts_are_paged(&self) -> bool {
        self.expert_memory_mode() == ExpertMemoryMode::Paged
    }
}
