use astronomical_ipc_protocol::ExpertMemoryMode;

use crate::ExpertResidencyTelemetry;
use crate::MlxActiveMemoryBreakdown;
use crate::classify_expert_memory_mode;
use crate::qwen3_5::model::Qwen3_5Model;

impl Qwen3_5Model {
    /// Builds the residency claim from the reconciled breakdown of the same MLX
    /// measurement. In paged mode the retained cache counts adopted lazy pages —
    /// layers seated for ownership before their arrays are materialized by the
    /// layer-interval eval — so its bookkeeping payload exceeds the physically
    /// resident bytes during the seat-to-first-eval window. The only truthful
    /// resident-payload figure is therefore the measured attribution from the
    /// snapshot this breakdown reconciles (issue #337). Complete residency is
    /// fully materialized by definition, so it keeps reporting owner figures.
    #[must_use]
    pub(crate) fn expert_residency_telemetry_for_breakdown(
        &self,
        mlx_memory_breakdown: &MlxActiveMemoryBreakdown,
    ) -> ExpertResidencyTelemetry {
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
            resident_expert_payload_bytes: mlx_memory_breakdown.expert_payload_bytes,
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
