use super::tensor_id::LagunaExpertProjection;

/// Neutral-inventory input restricted to the raw tensor name for this unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LagunaRawTensorNameRecord {
    raw_name: String,
}

impl LagunaRawTensorNameRecord {
    /// Captures one evidenced name before Laguna-owned interpretation begins.
    pub fn new(raw_name: impl Into<String>) -> Self {
        Self {
            raw_name: raw_name.into(),
        }
    }

    pub(super) fn raw_name(&self) -> &str {
        &self.raw_name
    }
}

/// Raw source member retained only as provenance for a canonical assembly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LagunaTensorSource {
    raw_name: String,
}

impl LagunaTensorSource {
    pub(super) fn new(raw_name: &str) -> Self {
        Self {
            raw_name: raw_name.to_owned(),
        }
    }

    /// Returns the exact inventory name needed by the later source validator.
    #[must_use]
    pub fn raw_name(&self) -> &str {
        &self.raw_name
    }
}

/// Declarative construction of one canonical tensor from raw source members.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LagunaTensorAssembly {
    /// One source directly supplies a global, layer, or alias-normalized tensor.
    DirectAlias { sources: Vec<LagunaTensorSource> },
    /// One already-stacked source supplies every routed expert.
    StackedSource { sources: Vec<LagunaTensorSource> },
    /// Ordered per-expert sources must be stacked by increasing expert index.
    PerExpertStack { sources: Vec<LagunaTensorSource> },
    /// One stacked fused source supplies the selected canonical gate or up half.
    FusedGateUpSource {
        sources: Vec<LagunaTensorSource>,
        projection: LagunaExpertProjection,
    },
    /// Per-expert fused sources must be stacked and split into the selected half.
    FusedPerExpertGateUp {
        sources: Vec<LagunaTensorSource>,
        projection: LagunaExpertProjection,
    },
}

impl LagunaTensorAssembly {
    pub(super) fn direct(raw_name: &str) -> Self {
        Self::DirectAlias {
            sources: vec![LagunaTensorSource::new(raw_name)],
        }
    }

    pub(super) fn stacked(raw_name: &str) -> Self {
        Self::StackedSource {
            sources: vec![LagunaTensorSource::new(raw_name)],
        }
    }

    pub(super) fn per_expert(raw_names: Vec<String>) -> Self {
        Self::PerExpertStack {
            sources: to_sources(raw_names),
        }
    }

    pub(super) fn fused_stacked(raw_name: &str, projection: LagunaExpertProjection) -> Self {
        Self::FusedGateUpSource {
            sources: vec![LagunaTensorSource::new(raw_name)],
            projection,
        }
    }

    pub(super) fn fused_per_expert(
        raw_names: Vec<String>,
        projection: LagunaExpertProjection,
    ) -> Self {
        Self::FusedPerExpertGateUp {
            sources: to_sources(raw_names),
            projection,
        }
    }

    /// Returns exact raw members in their required assembly order.
    #[must_use]
    pub fn sources(&self) -> &[LagunaTensorSource] {
        match self {
            Self::DirectAlias { sources }
            | Self::StackedSource { sources }
            | Self::PerExpertStack { sources }
            | Self::FusedGateUpSource { sources, .. }
            | Self::FusedPerExpertGateUp { sources, .. } => sources,
        }
    }
}

fn to_sources(raw_names: Vec<String>) -> Vec<LagunaTensorSource> {
    raw_names
        .into_iter()
        .map(|raw_name| LagunaTensorSource { raw_name })
        .collect()
}
