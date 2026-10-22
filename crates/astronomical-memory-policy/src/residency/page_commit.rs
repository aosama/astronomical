//! Whether just-read experts may stay in the retained cache.
//!
//! Two caching strategies live here, both answering "may this page stay":
//! whole-layer caching seats an entire decoder layer's experts or none of
//! them, and hot-expert caching keeps the individual routed experts the
//! router keeps choosing. In "hot-expert," *hot* means routing frequency, not
//! temperature.

use super::{CurrentExpertLayerResidency, RetainedExpertPageClass};

/// Seat a complete layer after a mandatory read when the plan asked for it.
///
/// Prefill does this for multi-token chunks. Decode must do it too after an
/// atomic complete-owner demote, or leftover budget never becomes resident RAM
/// and every generate token streams from SSD. Streamed-role layers admit too:
/// a multi-token chunk routes nearly every expert, so the mandatory read
/// already paid the complete-layer cost and discarding it would re-stream the
/// same bytes every later chunk. Budget admission remains the only refusal.
#[must_use]
pub const fn should_commit_mandatory_complete_layer(
    _route_token_count: i32,
    production_default_paging: bool,
    residency_target: Option<super::ExpertLayerResidencyTarget>,
) -> bool {
    production_default_paging
        && matches!(
            residency_target,
            Some(
                super::ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead
                    | super::ExpertLayerResidencyTarget::AdmitCompleteOnMandatoryRead
            )
        )
}

/// Keep routed experts after a mandatory read so the next forward can hit them
/// (hot-expert caching). Overflow is handled by evicting the least-used
/// retained page, not by refusing to cache what this forward used.
///
/// The active residency plan is the phase's will for the layer, so the commit
/// honors it: a layer the plan streams operation-local keeps streaming, and a
/// layer the plan is releasing is not refilled behind the plan's back.
#[must_use]
pub const fn should_commit_mandatory_routed_page(
    _route_token_count: i32,
    production_default_paging: bool,
    residency_target: Option<super::ExpertLayerResidencyTarget>,
    _layer_has_no_retained_page: bool,
) -> bool {
    if !production_default_paging {
        return false;
    }
    match residency_target {
        Some(super::ExpertLayerResidencyTarget::StreamOperationLocal) => false,
        Some(
            super::ExpertLayerResidencyTarget::ReleasePartial
            | super::ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit,
        ) => false,
        Some(_) | None => true,
    }
}

/// Structural slot cap for one decode warm table (hot-expert caching).
///
/// Issue #514: a history-length multiplier (`experts_per_token * 8`) left
/// leftover decode entitlement unclaimed on every machine, because the cap
/// did not depend on leftover RAM. The structural ceiling is the layer's
/// own expert count — a table that fills becomes a complete layer, which
/// the complete-layer residency machinery already understands. The economic
/// ceiling is leftover decode entitlement, applied by the caller
/// (`decode_warm_slot_count`) as `min(this, budget_affordable_capacity)`.
/// Least-frequently-used eviction still drops one-off routing noise; budget
/// admission still refuses a table the machine cannot hold.
#[must_use]
pub const fn hot_expert_warm_slot_count(expert_capacity: usize) -> usize {
    expert_capacity
}

/// Chooses which complete layers to evict so the decode warm floor fits.
///
/// Pure topology-and-budget arithmetic (issue #373's mixed-serving
/// prerequisite): returns the coldest complete layers (lowest covered
/// routed demand first) whose release grows the warm budget to the floor.
/// The floor grows as complete layers are evicted (their layers become
/// unseated), so the loop recomputes it per candidate; evicting each
/// layer frees its resident payload, which grows the warm budget one for
/// one. A model whose sparse layers are all fully seated needs no floor
/// and evicts nothing.
#[must_use]
pub fn decode_warm_floor_eviction_indexes(
    layer_residencies: &[CurrentExpertLayerResidency],
    warm_budget_bytes: u64,
    per_expert_payload_bytes: u64,
    minimum_warm_slot_count: usize,
    sparse_layer_count: usize,
) -> Vec<usize> {
    let mut complete_layer_residencies = layer_residencies
        .iter()
        .filter(|layer_residency| {
            layer_residency.class == RetainedExpertPageClass::StableCompleteLayer
        })
        .collect::<Vec<_>>();
    if complete_layer_residencies.is_empty() {
        return Vec::new();
    }
    // Coldest first: the layers the request routes to least are the ones
    // whose concentrated seating helps decode the least.
    complete_layer_residencies.sort_by_key(|layer_residency| {
        (
            layer_residency.covered_weighted_demand,
            layer_residency.layer_index,
        )
    });
    let complete_layer_count = complete_layer_residencies.len();
    let mut freed_payload_bytes = 0_u64;
    let mut eviction_indexes = Vec::new();
    for layer_residency in complete_layer_residencies {
        let unseated_sparse_layer_count =
            sparse_layer_count.saturating_sub(complete_layer_count - eviction_indexes.len());
        let minimum_warm_budget_bytes = u128::from(per_expert_payload_bytes)
            .saturating_mul(u128::try_from(unseated_sparse_layer_count).unwrap_or(u128::MAX))
            .saturating_mul(u128::try_from(minimum_warm_slot_count).unwrap_or(u128::MAX));
        if u128::from(warm_budget_bytes.saturating_add(freed_payload_bytes))
            >= minimum_warm_budget_bytes
        {
            break;
        }
        freed_payload_bytes = freed_payload_bytes.saturating_add(layer_residency.payload_bytes);
        eviction_indexes.push(layer_residency.layer_index);
    }
    eviction_indexes
}

/// Minimum decode warm slots every sparse layer is entitled to before
/// complete-layer seating may consume the retained budget.
///
/// Complete-layer seating and distributed warm tables compete for the same
/// retained bytes at the decode handoff. Concentrating the whole budget in
/// a few fully covered layers leaves every other layer re-streaming each
/// routed page per token, which mixed serving then pays at full
/// stream-and-decode cost; two tokens' routing worth of warm slots per
/// layer is the floor at which consecutive tokens' routes overlap heavily
/// (measured route-reuse on the served geometry) and the prefetch
/// machinery has a table to land in. Two tokens rather than one because a
/// single token's set is exactly the set the previous token already
/// streamed — the floor must cover the union of two consecutive tokens to
/// keep the previous token's experts resident while the next token's
/// missing experts stream. The decode handoff enforces this floor by
/// evicting the coldest complete layers until the warm budget can seat it
/// (or no complete layers remain).
#[must_use]
pub const fn minimum_decode_warm_slot_count(experts_per_token: usize) -> usize {
    if experts_per_token == 0 {
        1
    } else {
        experts_per_token.saturating_mul(2)
    }
}
