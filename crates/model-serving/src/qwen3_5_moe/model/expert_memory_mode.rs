use astronomical_ipc_protocol::ExpertMemoryMode;

use crate::qwen3_5::model::Qwen3_5Model;

impl Qwen3_5Model {
    /// Returns whether complete sparse experts are installed or demand-paged.
    #[must_use]
    pub fn expert_memory_mode(&self) -> ExpertMemoryMode {
        if self.expert_pager.is_some() && self.resident_expert_weights.is_none() {
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
