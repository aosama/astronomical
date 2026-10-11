use astronomical_ipc_protocol::ExpertMemoryMode;

use super::streaming_model::Qwen3_5StreamingModel;
use crate::ExpertResidencyTelemetry;
use crate::MlxActiveMemoryBreakdown;

impl Qwen3_5StreamingModel {
    /// Builds the residency claim from the reconciled breakdown of the same MLX
    /// measurement. The retained cache counts adopted lazy pages —
    /// layers seated for ownership before their arrays are materialized by the
    /// layer-interval eval — so its bookkeeping payload exceeds the physically
    /// resident bytes during the seat-to-first-eval window. The only truthful
    /// resident-payload figure is therefore the measured attribution from the
    /// snapshot this breakdown reconciles (issue #337).
    #[must_use]
    pub(crate) fn expert_residency_telemetry_for_breakdown(
        &self,
        mlx_memory_breakdown: &MlxActiveMemoryBreakdown,
    ) -> ExpertResidencyTelemetry {
        let total_layer_count = self
            .expert_pager
            .as_ref()
            .map_or(0, |expert_pager| expert_pager.layer_count());
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
        crate::classify_expert_memory_mode(
            false,
            self.expert_pager.is_some(),
            retained_paged_expert_payload_bytes,
        )
    }
}
