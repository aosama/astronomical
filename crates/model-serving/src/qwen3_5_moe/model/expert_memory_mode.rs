use astronomical_ipc_protocol::ExpertMemoryMode;

use crate::ExpertResidencyTelemetry;
use crate::classify_expert_memory_mode;
use crate::qwen3_5::model::Qwen3_5Model;

impl Qwen3_5Model {
    /// Returns retained-expert count and payload for status and IPC.
    #[must_use]
    pub(crate) fn expert_residency_telemetry(&self) -> ExpertResidencyTelemetry {
        let total_layer_count = self
            .expert_pager
            .as_ref()
            .map_or(0, |expert_pager| expert_pager.layer_count());
        if let Some(resident_expert_weights) = self.resident_expert_weights.as_ref() {
            let resident_expert_count = self
                .expert_pager
                .as_ref()
                .map_or(resident_expert_weights.layer_count(), |expert_pager| {
                    expert_pager.complete_expert_entry_count()
                });
            let resident_expert_payload_bytes =
                self.expert_pager.as_ref().map_or(0, |expert_pager| {
                    expert_pager
                        .complete_expert_payload_byte_count()
                        .unwrap_or(0)
                });
            return ExpertResidencyTelemetry {
                total_layer_count: u32::try_from(total_layer_count).unwrap_or(u32::MAX),
                resident_expert_count: u32::try_from(resident_expert_count).unwrap_or(u32::MAX),
                resident_expert_payload_bytes,
            };
        }
        let expert_statistics = self.expert_weight_memory_cache_statistics();
        ExpertResidencyTelemetry {
            total_layer_count: u32::try_from(total_layer_count).unwrap_or(u32::MAX),
            resident_expert_count: u32::try_from(expert_statistics.entry_count).unwrap_or(u32::MAX),
            resident_expert_payload_bytes: expert_statistics.resident_payload_byte_count,
        }
    }

    /// Returns whether complete sparse experts are installed or demand-paged.
    #[must_use]
    pub fn expert_memory_mode(&self) -> ExpertMemoryMode {
        let retained_paged_expert_payload_bytes =
            self.retained_experts
                .as_ref()
                .map_or(0, |retained_experts| {
                    retained_experts
                        .borrow()
                        .statistics()
                        .resident_payload_byte_count
                });
        classify_expert_memory_mode(
            self.resident_expert_weights.is_some(),
            self.expert_pager.is_some(),
            retained_paged_expert_payload_bytes,
        )
    }

    /// Returns whether sparse-expert pages remain necessary for the next forward.
    #[must_use]
    pub(crate) fn sparse_experts_are_paged(&self) -> bool {
        self.expert_memory_mode() != ExpertMemoryMode::Resident
    }
}
