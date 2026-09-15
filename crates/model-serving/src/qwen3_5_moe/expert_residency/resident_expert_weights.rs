use super::Qwen3_5ResidentExpertLayerWeights;

/// One atomic owner for every target and optional MTP sparse expert layer.
#[derive(Debug)]
pub(crate) struct Qwen3_5ResidentExpertWeights {
    /// Uses the pager plan order: target decoder layers followed by optional MTP.
    layers: Vec<Qwen3_5ResidentExpertLayerWeights>,
    /// Number of `(layer, expert)` entries represented by the complete owner.
    expert_entry_count: usize,
    /// Exact source payload, excluding MLX allocator bookkeeping and graph work.
    payload_byte_count: u64,
}

impl Qwen3_5ResidentExpertWeights {
    pub(super) fn new(
        layers: Vec<Qwen3_5ResidentExpertLayerWeights>,
        expert_entry_count: usize,
        payload_byte_count: u64,
    ) -> Self {
        Self {
            layers,
            expert_entry_count,
            payload_byte_count,
        }
    }

    pub(crate) fn layer(&self, layer_index: usize) -> Option<&Qwen3_5ResidentExpertLayerWeights> {
        // Keeping the resident and pager layer indices identical lets target and
        // MTP forwards switch owners without remapping global expert identifiers.
        self.layers.get(layer_index)
    }

    pub(crate) const fn expert_entry_count(&self) -> usize {
        self.expert_entry_count
    }

    pub(crate) const fn payload_byte_count(&self) -> u64 {
        self.payload_byte_count
    }

    pub(crate) fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Builds a complete owner whose final layer serves the sidecar-fused MTP
    /// head. The base layers come from the pager plans; the appended layer owns
    /// the sidecar's per-expert BF16 arrays directly, so the MTP forward can
    /// route through the same resident gather path without a pager plan.
    pub(crate) fn with_appended_sidecar_mtp_layer(
        base_layers: Vec<Qwen3_5ResidentExpertLayerWeights>,
        base_expert_entry_count: usize,
        base_payload_byte_count: u64,
        sidecar_mtp_layer: Qwen3_5ResidentExpertLayerWeights,
        sidecar_mtp_payload_byte_count: u64,
    ) -> Self {
        let mut layers = base_layers;
        layers.push(sidecar_mtp_layer);
        Self {
            layers,
            expert_entry_count: base_expert_entry_count,
            payload_byte_count: base_payload_byte_count
                .saturating_add(sidecar_mtp_payload_byte_count),
        }
    }
}
