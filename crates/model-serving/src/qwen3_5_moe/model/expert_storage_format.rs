use astronomical_ipc_protocol::ExpertStorageFormat;

use crate::qwen3_5::model::Qwen3_5Model;

use super::super::expert_paging::ExpertPager;

impl Qwen3_5Model {
    /// Returns the expert file layout selected for request-time paging.
    #[must_use]
    pub fn expert_storage_format(&self) -> ExpertStorageFormat {
        self.expert_pager.as_ref().map_or(
            ExpertStorageFormat::StandardSafetensors,
            ExpertPager::expert_storage_format,
        )
    }
}
