//! Family-neutral expert residency policy: plan, keep, release, and stream.
//!
//! This module owns no arrays and performs no I/O. Execution families ask it
//! which experts to keep after a mandatory read, whether a planned release may
//! run in the current phase, and whether weight-stationary prefill is legal.
//! They must not invent a second answer to those questions.
//!
//! Prefill uses `RequestExpertResidency`. Generation discards that contract.
//!
//! Submodules, by question: `page_commit.rs` (may this page be committed to
//! retained RAM now?), `release.rs` (may the planned release run in this
//! phase?), `request_plan.rs` (which layers does prefill need stable?),
//! `validation.rs` (are the supplied plan inputs internally consistent?),
//! `ownership_mode.rs` (the single Resident/Hybrid/Paged classification),
//! `decode_seating.rs` (which complete layers must be seated before the first
//! decode token?), and `complete_headroom.rs` (the ceiling below which paging
//! is mandatory despite complete residency).

mod complete_headroom;
mod decode_seating;
mod ownership_mode;
mod page_commit;
mod release;
mod request_plan;
mod validation;

pub use complete_headroom::CompleteResidencyHeadroomBoundary;
pub use decode_seating::complete_layer_indexes_required_before_decode;
pub use ownership_mode::classify_expert_memory_mode;
pub use page_commit::{
    hot_expert_warm_slot_count, should_commit_mandatory_complete_layer,
    should_commit_mandatory_routed_page,
};
pub use release::should_enact_planned_expert_release;
pub use request_plan::{
    RequestExpertLayerRole, RequestExpertResidency, publish_request_stable_residency_plan,
    retained_complete_layer_ceiling_after_prefill_budget_refresh,
};

use crate::memory::MemoryPhase;
use thiserror::Error;

use validation::{checked_sum, routed_floor_payload_bytes, validate_inputs};

/// Lifetime and eviction priority of one retained expert page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetainedExpertPageClass {
    /// A complete decoder layer's experts, pinned for the request's lifetime.
    /// Decode never streams it; reclamation may only release it by exact
    /// deficit, never opportunistically.
    StableCompleteLayer,
    /// Routed-expert pages retained opportunistically; first to yield when a
    /// later phase needs the budget back.
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

/// Planned behavior for one decoder layer.
///
/// One target per sparse decoder layer per plan; the family enacts exactly the
/// named target and invents no second disposition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertLayerResidencyTarget {
    /// The layer's complete expert payload is already resident and stays.
    PreserveComplete,
    /// The layer is not resident, but this operation must read it anyway:
    /// promote the complete payload to retained RAM rather than streaming it.
    PromoteCompleteOnMandatoryRead,
    /// A partial page set is already retained and stays for this operation.
    PreservePartial,
    /// A partial layer is admitted because this operation's routes demand it;
    /// the admission charges the admitted pages to the retained budget.
    AdmitPartialOnMandatoryRouteRead,
    /// The layer is streamed for this operation only and retained by nobody.
    StreamOperationLocal,
    /// The layer's partial retention yields (fully or partly) to free budget.
    ReleasePartial,
    /// The layer's complete retention yields by exactly the computed deficit,
    /// leaving the rest of the layer retained.
    ReleaseCompleteForExactDeficit,
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

/// Produces a deterministic complete-foundation plus routed-overlay plan.
pub fn plan_expert_residency(
    phase: MemoryPhase,
    retained_expert_ceiling_bytes: u64,
    layer_geometries: &[ExpertLayerGeometry],
    current_residencies: &[CurrentExpertLayerResidency],
) -> Result<ExpertResidencyPlan, ExpertResidencyPlanError> {
    let current_by_layer = validate_inputs(
        retained_expert_ceiling_bytes,
        layer_geometries,
        current_residencies,
    )?;
    // Generation must keep experts prefill already paid to read. Seating new
    // complete layers over those pages would evict them. After an atomic
    // complete-owner demote the topology is empty: leftover budget must still
    // seat complete layers, or generate runs with zero expert RAM while tens of
    // gigabytes of leftover sit unused.
    if matches!(
        phase,
        MemoryPhase::GenerationPreparation | MemoryPhase::Decode
    ) && !current_residencies.is_empty()
    {
        return preserve_existing_expert_pages_for_generation(
            phase,
            retained_expert_ceiling_bytes,
            layer_geometries,
            &current_by_layer,
        );
    }
    let complete_model_payload_bytes = checked_sum(
        layer_geometries
            .iter()
            .map(|geometry| geometry.complete_layer_payload_bytes),
    )?;
    if complete_model_payload_bytes <= retained_expert_ceiling_bytes {
        return complete_model_plan(
            phase,
            retained_expert_ceiling_bytes,
            layer_geometries,
            &current_by_layer,
            complete_model_payload_bytes,
        );
    }

    let routed_floor_bytes = layer_geometries
        .iter()
        .map(routed_floor_payload_bytes)
        .collect::<Result<Vec<_>, _>>()?;
    let all_layer_routed_floor_bytes = checked_sum(routed_floor_bytes.iter().copied())?;
    if all_layer_routed_floor_bytes > retained_expert_ceiling_bytes {
        return low_budget_partial_plan(
            phase,
            retained_expert_ceiling_bytes,
            layer_geometries,
            &current_by_layer,
        );
    }
    foundation_and_overlay_plan(
        phase,
        retained_expert_ceiling_bytes,
        layer_geometries,
        &current_by_layer,
        &routed_floor_bytes,
        all_layer_routed_floor_bytes,
    )
}

fn complete_model_plan(
    phase: MemoryPhase,
    ceiling_bytes: u64,
    geometries: &[ExpertLayerGeometry],
    current_by_layer: &[Option<&CurrentExpertLayerResidency>],
    complete_model_payload_bytes: u64,
) -> Result<ExpertResidencyPlan, ExpertResidencyPlanError> {
    let mut preserved_bytes = 0_u64;
    let mut layer_targets = Vec::with_capacity(geometries.len());
    for current_residency in current_by_layer {
        if current_residency.is_some_and(|residency| {
            residency.class == RetainedExpertPageClass::StableCompleteLayer
        }) {
            layer_targets.push(ExpertLayerResidencyTarget::PreserveComplete);
        } else {
            layer_targets.push(ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead);
        }
        if let Some(residency) = current_residency {
            preserved_bytes = preserved_bytes
                .checked_add(residency.payload_bytes)
                .ok_or(ExpertResidencyPlanError::ByteCountOverflow)?;
        }
    }
    Ok(ExpertResidencyPlan {
        phase,
        retained_expert_ceiling_bytes: ceiling_bytes,
        complete_layer_targets: (0..geometries.len()).collect(),
        layer_targets,
        reserved_routed_overlay_bytes: 0,
        expected_preserved_bytes: preserved_bytes,
        maximum_new_retained_bytes: complete_model_payload_bytes.saturating_sub(preserved_bytes),
        deterministic_release_order: release_order(current_by_layer),
        is_low_budget_partial_mode: false,
    })
}

fn preserve_existing_expert_pages_for_generation(
    phase: MemoryPhase,
    ceiling_bytes: u64,
    geometries: &[ExpertLayerGeometry],
    current_by_layer: &[Option<&CurrentExpertLayerResidency>],
) -> Result<ExpertResidencyPlan, ExpertResidencyPlanError> {
    let mut candidate_layers = current_by_layer
        .iter()
        .enumerate()
        .filter_map(|(layer_index, residency)| residency.map(|residency| (layer_index, residency)))
        .collect::<Vec<_>>();
    candidate_layers.sort_unstable_by(|left, right| compare_preservation_priority(*left, *right));
    let mut selected = vec![false; geometries.len()];
    let mut preserved_bytes = 0_u64;
    for (layer_index, residency) in candidate_layers {
        if residency.payload_bytes <= ceiling_bytes.saturating_sub(preserved_bytes) {
            selected[layer_index] = true;
            preserved_bytes = preserved_bytes
                .checked_add(residency.payload_bytes)
                .ok_or(ExpertResidencyPlanError::ByteCountOverflow)?;
        }
    }
    let layer_targets = current_by_layer
        .iter()
        .enumerate()
        .map(
            |(layer_index, residency)| match (residency, selected[layer_index]) {
                (Some(residency), true)
                    if residency.class == RetainedExpertPageClass::StableCompleteLayer =>
                {
                    ExpertLayerResidencyTarget::PreserveComplete
                }
                (Some(_), true) => ExpertLayerResidencyTarget::PreservePartial,
                (Some(residency), false)
                    if residency.class == RetainedExpertPageClass::StableCompleteLayer =>
                {
                    ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit
                }
                (Some(_), false) => ExpertLayerResidencyTarget::ReleasePartial,
                (None, _) => ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead,
            },
        )
        .collect();
    Ok(ExpertResidencyPlan {
        phase,
        retained_expert_ceiling_bytes: ceiling_bytes,
        complete_layer_targets: selected
            .iter()
            .enumerate()
            .filter_map(|(layer_index, is_selected)| {
                (*is_selected
                    && current_by_layer[layer_index].is_some_and(|residency| {
                        residency.class == RetainedExpertPageClass::StableCompleteLayer
                    }))
                .then_some(layer_index)
            })
            .collect(),
        layer_targets,
        reserved_routed_overlay_bytes: 0,
        expected_preserved_bytes: preserved_bytes,
        maximum_new_retained_bytes: ceiling_bytes.saturating_sub(preserved_bytes),
        deterministic_release_order: release_order(current_by_layer),
        is_low_budget_partial_mode: false,
    })
}

fn low_budget_partial_plan(
    phase: MemoryPhase,
    ceiling_bytes: u64,
    geometries: &[ExpertLayerGeometry],
    current_by_layer: &[Option<&CurrentExpertLayerResidency>],
) -> Result<ExpertResidencyPlan, ExpertResidencyPlanError> {
    let mut candidate_layers = current_by_layer
        .iter()
        .enumerate()
        .filter_map(|(layer_index, residency)| residency.map(|residency| (layer_index, residency)))
        .collect::<Vec<_>>();
    candidate_layers.sort_unstable_by(|left, right| compare_preservation_priority(*left, *right));
    let mut selected = vec![false; geometries.len()];
    let mut preserved_bytes = 0_u64;
    for (layer_index, residency) in candidate_layers {
        if residency.payload_bytes <= ceiling_bytes.saturating_sub(preserved_bytes) {
            selected[layer_index] = true;
            preserved_bytes = preserved_bytes
                .checked_add(residency.payload_bytes)
                .ok_or(ExpertResidencyPlanError::ByteCountOverflow)?;
        }
    }
    let layer_targets = current_by_layer
        .iter()
        .enumerate()
        .map(
            |(layer_index, residency)| match (residency, selected[layer_index]) {
                (Some(residency), true)
                    if residency.class == RetainedExpertPageClass::StableCompleteLayer =>
                {
                    ExpertLayerResidencyTarget::PreserveComplete
                }
                (Some(_), true) => ExpertLayerResidencyTarget::PreservePartial,
                (Some(residency), false)
                    if residency.class == RetainedExpertPageClass::StableCompleteLayer =>
                {
                    ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit
                }
                (Some(_), false) => ExpertLayerResidencyTarget::ReleasePartial,
                (None, _) => ExpertLayerResidencyTarget::StreamOperationLocal,
            },
        )
        .collect();
    Ok(ExpertResidencyPlan {
        phase,
        retained_expert_ceiling_bytes: ceiling_bytes,
        complete_layer_targets: selected
            .iter()
            .enumerate()
            .filter_map(|(layer_index, is_selected)| {
                (*is_selected
                    && current_by_layer[layer_index].is_some_and(|residency| {
                        residency.class == RetainedExpertPageClass::StableCompleteLayer
                    }))
                .then_some(layer_index)
            })
            .collect(),
        layer_targets,
        reserved_routed_overlay_bytes: 0,
        expected_preserved_bytes: preserved_bytes,
        maximum_new_retained_bytes: 0,
        deterministic_release_order: release_order(current_by_layer),
        is_low_budget_partial_mode: true,
    })
}

fn foundation_and_overlay_plan(
    phase: MemoryPhase,
    ceiling_bytes: u64,
    geometries: &[ExpertLayerGeometry],
    current_by_layer: &[Option<&CurrentExpertLayerResidency>],
    routed_floor_bytes: &[u64],
    all_layer_routed_floor_bytes: u64,
) -> Result<ExpertResidencyPlan, ExpertResidencyPlanError> {
    let mut complete_targets = vec![false; geometries.len()];
    let mut foundation_and_floor_bytes = all_layer_routed_floor_bytes;
    let mut complete_candidates = geometries
        .iter()
        .map(|geometry| {
            let layer_index = geometry.layer_index;
            (
                layer_index,
                geometry.complete_layer_payload_bytes - routed_floor_bytes[layer_index],
                current_by_layer[layer_index].is_some_and(|residency| {
                    residency.class == RetainedExpertPageClass::StableCompleteLayer
                }),
            )
        })
        .collect::<Vec<_>>();
    complete_candidates.sort_unstable_by_key(|(layer_index, incremental_bytes, is_complete)| {
        (!*is_complete, *incremental_bytes, *layer_index)
    });
    for (layer_index, incremental_bytes, _) in complete_candidates {
        if incremental_bytes <= ceiling_bytes.saturating_sub(foundation_and_floor_bytes) {
            complete_targets[layer_index] = true;
            foundation_and_floor_bytes = foundation_and_floor_bytes
                .checked_add(incremental_bytes)
                .ok_or(ExpertResidencyPlanError::ByteCountOverflow)?;
        }
    }

    let mut overlay_extra_bytes = ceiling_bytes.saturating_sub(foundation_and_floor_bytes);
    let mut preserve_partial = vec![false; geometries.len()];
    let mut partial_candidates = current_by_layer
        .iter()
        .enumerate()
        .filter_map(|(layer_index, residency)| {
            let residency = residency.as_ref().copied()?;
            (!complete_targets[layer_index]
                && residency.class == RetainedExpertPageClass::ElasticRoutedExperts)
                .then_some((layer_index, residency))
        })
        .collect::<Vec<_>>();
    partial_candidates.sort_unstable_by(|left, right| compare_partial_coverage(*right, *left));
    for (layer_index, residency) in partial_candidates {
        let incremental_bytes = residency
            .payload_bytes
            .saturating_sub(routed_floor_bytes[layer_index]);
        if incremental_bytes <= overlay_extra_bytes {
            preserve_partial[layer_index] = true;
            overlay_extra_bytes = overlay_extra_bytes.saturating_sub(incremental_bytes);
        }
    }

    let mut preserved_bytes = 0_u64;
    let mut layer_targets = Vec::with_capacity(geometries.len());
    for layer_index in 0..geometries.len() {
        let current_residency = current_by_layer[layer_index];
        let target = if complete_targets[layer_index] {
            if current_residency.is_some_and(|residency| {
                residency.class == RetainedExpertPageClass::StableCompleteLayer
            }) {
                ExpertLayerResidencyTarget::PreserveComplete
            } else {
                ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead
            }
        } else {
            match current_residency {
                Some(residency)
                    if residency.class == RetainedExpertPageClass::StableCompleteLayer =>
                {
                    ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit
                }
                Some(_) if preserve_partial[layer_index] => {
                    ExpertLayerResidencyTarget::PreservePartial
                }
                Some(_) => ExpertLayerResidencyTarget::ReleasePartial,
                None => ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead,
            }
        };
        if matches!(
            target,
            ExpertLayerResidencyTarget::PreserveComplete
                | ExpertLayerResidencyTarget::PreservePartial
                | ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead
        ) && let Some(residency) = current_residency
        {
            preserved_bytes = preserved_bytes
                .checked_add(residency.payload_bytes)
                .ok_or(ExpertResidencyPlanError::ByteCountOverflow)?;
        }
        layer_targets.push(target);
    }
    let reserved_routed_overlay_bytes =
        checked_sum(routed_floor_bytes.iter().enumerate().filter_map(
            |(layer_index, floor_bytes)| (!complete_targets[layer_index]).then_some(*floor_bytes),
        ))?;
    let target_capacity_bytes = checked_sum(geometries.iter().map(|geometry| {
        if complete_targets[geometry.layer_index] {
            geometry.complete_layer_payload_bytes
        } else {
            routed_floor_bytes[geometry.layer_index]
        }
    }))?;
    if target_capacity_bytes > ceiling_bytes {
        return Err(ExpertResidencyPlanError::PlannedResidencyExceedsCeiling);
    }
    Ok(ExpertResidencyPlan {
        phase,
        retained_expert_ceiling_bytes: ceiling_bytes,
        complete_layer_targets: complete_targets
            .iter()
            .enumerate()
            .filter_map(|(layer_index, is_complete)| is_complete.then_some(layer_index))
            .collect(),
        layer_targets,
        reserved_routed_overlay_bytes,
        expected_preserved_bytes: preserved_bytes,
        maximum_new_retained_bytes: target_capacity_bytes.saturating_sub(preserved_bytes),
        deterministic_release_order: release_order(current_by_layer),
        is_low_budget_partial_mode: false,
    })
}

fn compare_partial_coverage(
    left: (usize, &CurrentExpertLayerResidency),
    right: (usize, &CurrentExpertLayerResidency),
) -> std::cmp::Ordering {
    let left_score = u128::from(left.1.covered_weighted_demand) * u128::from(right.1.payload_bytes);
    let right_score =
        u128::from(right.1.covered_weighted_demand) * u128::from(left.1.payload_bytes);
    left_score
        .cmp(&right_score)
        .then_with(|| right.0.cmp(&left.0))
}

fn compare_preservation_priority(
    left: (usize, &CurrentExpertLayerResidency),
    right: (usize, &CurrentExpertLayerResidency),
) -> std::cmp::Ordering {
    let left_is_complete = left.1.class == RetainedExpertPageClass::StableCompleteLayer;
    let right_is_complete = right.1.class == RetainedExpertPageClass::StableCompleteLayer;
    left_is_complete
        .cmp(&right_is_complete)
        .then_with(|| compare_partial_coverage(right, left))
}

fn release_order(current_by_layer: &[Option<&CurrentExpertLayerResidency>]) -> Vec<usize> {
    let mut partial_layers = current_by_layer
        .iter()
        .enumerate()
        .filter_map(|(layer_index, residency)| {
            let residency = residency.as_ref().copied()?;
            (residency.class == RetainedExpertPageClass::ElasticRoutedExperts)
                .then_some((layer_index, residency))
        })
        .collect::<Vec<_>>();
    partial_layers.sort_unstable_by(|left, right| compare_partial_coverage(*left, *right));
    let mut release_order = partial_layers
        .into_iter()
        .map(|(layer_index, _)| layer_index)
        .collect::<Vec<_>>();
    release_order.extend(current_by_layer.iter().enumerate().rev().filter_map(
        |(layer_index, residency)| {
            residency
                .is_some_and(|residency| {
                    residency.class == RetainedExpertPageClass::StableCompleteLayer
                })
                .then_some(layer_index)
        },
    ));
    release_order
}
