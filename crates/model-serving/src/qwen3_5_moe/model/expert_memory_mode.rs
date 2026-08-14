use astronomical_ipc_protocol::ExpertMemoryMode;

use crate::ExpertResidencyTelemetry;
use crate::qwen3_5::model::Qwen3_5Model;

impl Qwen3_5Model {
    /// Returns complete and partial retained layer counts for phase telemetry.
    #[must_use]
    pub(crate) fn retained_expert_layer_counts(&self) -> (usize, usize, usize) {
        if let Some(resident_expert_weights) = self.resident_expert_weights.as_ref() {
            let complete_layer_count = resident_expert_weights.layer_count();
            return (complete_layer_count, complete_layer_count, 0);
        }
        let total_layer_count = self
            .expert_pager
            .as_ref()
            .map_or(0, |expert_pager| expert_pager.layer_count());
        let Some(retained_expert_layers) = self.retained_expert_layers.as_ref() else {
            return (total_layer_count, 0, 0);
        };
        let retained_expert_layers = retained_expert_layers.borrow();
        let mut complete_layer_count = 0_usize;
        let mut partial_layer_count = 0_usize;
        for layer_index in 0..total_layer_count {
            let Some(retained_layer) = retained_expert_layers.retained_layer(layer_index) else {
                continue;
            };
            if retained_layer.manifest.contains_all_experts() {
                complete_layer_count = complete_layer_count.saturating_add(1);
            } else {
                partial_layer_count = partial_layer_count.saturating_add(1);
            }
        }
        (total_layer_count, complete_layer_count, partial_layer_count)
    }

    /// Returns the current concrete complete/partial ownership and payload bytes.
    #[must_use]
    pub(crate) fn expert_residency_telemetry(&self) -> ExpertResidencyTelemetry {
        let (total_layer_count, _, _) = self.retained_expert_layer_counts();
        let expert_statistics = self.expert_weight_memory_cache_statistics();
        ExpertResidencyTelemetry {
            total_layer_count: u32::try_from(total_layer_count).unwrap_or(u32::MAX),
            complete_layer_count: u32::try_from(expert_statistics.complete_layer_count)
                .unwrap_or(u32::MAX),
            complete_layer_payload_bytes: expert_statistics.complete_layer_payload_byte_count,
            partial_layer_count: u32::try_from(expert_statistics.partial_layer_count)
                .unwrap_or(u32::MAX),
            partial_layer_payload_bytes: expert_statistics.partial_layer_payload_byte_count,
        }
    }

    /// Returns whether complete sparse experts are installed or demand-paged.
    #[must_use]
    pub fn expert_memory_mode(&self) -> ExpertMemoryMode {
        if self.resident_expert_weights.is_some() || self.expert_pager.is_none() {
            return ExpertMemoryMode::Resident;
        }
        if self
            .retained_expert_layers
            .as_ref()
            .is_some_and(|retained_layers| {
                retained_layers
                    .borrow()
                    .statistics()
                    .resident_payload_byte_count
                    > 0
            })
        {
            ExpertMemoryMode::Hybrid
        } else {
            ExpertMemoryMode::Paged
        }
    }

    /// Returns whether sparse-expert pages remain necessary for the next forward.
    #[must_use]
    pub(crate) fn sparse_experts_are_paged(&self) -> bool {
        self.expert_memory_mode() != ExpertMemoryMode::Resident
    }
}
