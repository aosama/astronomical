use std::collections::BTreeMap;

use super::{tensor_assembly::LagunaTensorAssembly, tensor_id::LagunaTensorId};

/// Gate/up packaging selected for one routed-expert layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LagunaExpertGateUpLayout {
    Split,
    Fused,
}

/// Deterministic canonical tensor map consumed by later Laguna validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LagunaTensorNameContract {
    assemblies: BTreeMap<LagunaTensorId, LagunaTensorAssembly>,
    expert_gate_up_layouts: BTreeMap<usize, LagunaExpertGateUpLayout>,
}

impl LagunaTensorNameContract {
    pub(super) fn new(
        assemblies: BTreeMap<LagunaTensorId, LagunaTensorAssembly>,
        expert_gate_up_layouts: BTreeMap<usize, LagunaExpertGateUpLayout>,
    ) -> Self {
        Self {
            assemblies,
            expert_gate_up_layouts,
        }
    }

    /// Returns canonical IDs and their declarative source construction in ID order.
    #[must_use]
    pub const fn assemblies(&self) -> &BTreeMap<LagunaTensorId, LagunaTensorAssembly> {
        &self.assemblies
    }

    /// Returns the validated gate/up packaging for one routed-expert layer.
    #[must_use]
    pub fn expert_gate_up_layout(&self, layer_index: usize) -> Option<LagunaExpertGateUpLayout> {
        self.expert_gate_up_layouts.get(&layer_index).copied()
    }
}
