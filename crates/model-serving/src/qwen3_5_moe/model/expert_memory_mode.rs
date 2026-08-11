use astronomical_ipc_protocol::ExpertMemoryMode;

use crate::qwen3_5::model::Qwen3_5Model;

impl Qwen3_5Model {
    /// Returns whether the model uses demand-loaded sparse experts.
    #[must_use]
    pub fn expert_memory_mode(&self) -> ExpertMemoryMode {
        if self.expert_pager.is_some() {
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
