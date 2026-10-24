use crate::phase::MemoryPhase;
use thiserror::Error;

/// Lifetime and eviction priority of one retained expert page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetainedExpertPageClass {
    /// A complete layer pinned until an exact deficit requires reclamation.
    StableCompleteLayer,
    /// Routed pages that yield before complete layers.
    ElasticRoutedExperts,
}

/// Exact immutable geometry for one sparse decoder layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertLayerGeometry {
    pub layer_index: usize,
    pub complete_layer_payload_bytes: u64,
    pub expert_payload_bytes: u64,
    pub expert_capacity: usize,
    pub experts_per_token: usize,
}

/// Current materialized ownership and bounded route-demand coverage for one layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CurrentExpertLayerResidency {
    pub layer_index: usize,
    pub class: RetainedExpertPageClass,
    pub retained_expert_ids: Vec<usize>,
    pub payload_bytes: u64,
    pub covered_weighted_demand: u64,
}

/// One disposition per sparse decoder layer. Execution enacts the target
/// without introducing a second residency policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertLayerResidencyTarget {
    PreserveComplete,
    /// Retain the complete payload when the execution-required read loads it.
    PromoteCompleteOnMandatoryRead,
    PreservePartial,
    AdmitPartialOnMandatoryRouteRead,
    StreamOperationLocal,
    ReleasePartial,
    ReleaseCompleteForExactDeficit,
    /// A mandatory multi-token read already materialized the complete payload;
    /// keep it when admission funds it rather than re-reading next chunk.
    AdmitCompleteOnMandatoryRead,
}

/// One deterministic topology plan that never commands eager source reads.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertResidencyPlan {
    pub phase: MemoryPhase,
    pub retained_expert_ceiling_bytes: u64,
    pub complete_layer_targets: Vec<usize>,
    pub layer_targets: Vec<ExpertLayerResidencyTarget>,
    pub reserved_routed_overlay_bytes: u64,
    pub expected_preserved_bytes: u64,
    pub maximum_new_retained_bytes: u64,
    pub deterministic_release_order: Vec<usize>,
    pub is_low_budget_partial_mode: bool,
}

/// Structural input defect that must stop planning before ownership changes.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ExpertResidencyPlanError {
    #[error("expert geometry must contain at least one layer")]
    EmptyGeometry,
    #[error("expected contiguous layer index {expected_layer_index}, found {actual_layer_index}")]
    NonContiguousLayerIndex {
        expected_layer_index: usize,
        actual_layer_index: usize,
    },
    #[error("layer {layer_index} has zero expert geometry")]
    ZeroGeometry { layer_index: usize },
    #[error("layer {layer_index} complete payload does not match exact expert geometry")]
    InconsistentCompletePayload { layer_index: usize },
    #[error("current residency contains duplicate or unordered layer {layer_index}")]
    DuplicateOrUnorderedCurrentLayer { layer_index: usize },
    #[error("current residency references out-of-range layer {layer_index}")]
    CurrentLayerOutOfRange { layer_index: usize },
    #[error(
        "layer {layer_index} retained expert identifiers are empty, unsorted, duplicated, or out of range"
    )]
    InvalidRetainedExpertIds { layer_index: usize },
    #[error(
        "layer {layer_index} retained payload {payload_bytes} does not match geometry {geometry_expert_payload_bytes} * {retained_count} = {expected_payload_bytes}"
    )]
    InconsistentCurrentPayload {
        layer_index: usize,
        payload_bytes: u64,
        geometry_expert_payload_bytes: u64,
        retained_count: usize,
        expected_payload_bytes: u64,
    },
    #[error("current retained payload exceeds the composed ceiling")]
    CurrentResidencyExceedsCeiling,
    #[error("planned retained payload exceeds the composed ceiling")]
    PlannedResidencyExceedsCeiling,
    #[error("expert residency byte arithmetic overflowed")]
    ByteCountOverflow,
}
