use super::Qwen3_5ProbeExpertLayerWeights;

/// One atomic owner for every target sparse expert layer.
#[derive(Debug)]
pub struct Qwen3_5ProbeExpertWeights {
    /// Uses the pager plan order: target decoder layers in declaration order.
    layers: Vec<Qwen3_5ProbeExpertLayerWeights>,
    /// Number of `(layer, expert)` entries represented by the complete owner.
    expert_entry_count: usize,
    /// Exact source payload, excluding MLX allocator bookkeeping and graph work.
    payload_byte_count: u64,
}

impl Qwen3_5ProbeExpertWeights {
    pub(super) fn new(
        layers: Vec<Qwen3_5ProbeExpertLayerWeights>,
        expert_entry_count: usize,
        payload_byte_count: u64,
    ) -> Self {
        Self {
            layers,
            expert_entry_count,
            payload_byte_count,
        }
    }

    pub fn layer(&self, layer_index: usize) -> Option<&Qwen3_5ProbeExpertLayerWeights> {
        // Keeping the held and pager layer indices identical lets every
        // forward switch owners without remapping global expert identifiers.
        self.layers.get(layer_index)
    }

    /// One layer's arrays, mutably: the sensitivity probe swaps a single
    /// projection for a re-encoded copy and restores it after its forwards.
    pub fn layer_mut(&mut self, layer_index: usize) -> Option<&mut Qwen3_5ProbeExpertLayerWeights> {
        self.layers.get_mut(layer_index)
    }

    pub const fn expert_entry_count(&self) -> usize {
        self.expert_entry_count
    }

    pub const fn payload_byte_count(&self) -> u64 {
        self.payload_byte_count
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}
