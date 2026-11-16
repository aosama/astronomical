//! Whether just-read experts may stay in the `astro_expert_cache`.
//!
//! Two holding strategies live here, both answering "may this page stay":
//! whole-layer holding seats an entire decoder layer's experts or none of
//! them, and hot-expert holding keeps the individual routed experts the
//! router keeps choosing. In "hot-expert," *hot* means routing frequency, not
//! temperature.

use super::{CurrentExpertLayerResidency, RetainedExpertPageClass};

/// Keep a whole layer that was just read when the layer should be kept.
///
/// While a prompt is read every layer is in that state, so a layer read
/// whole is kept as long as the `astro_expert_cache` has room: the read already paid the
/// whole-layer cost and discarding it would read the same bytes again. A
/// reply does it too for the layers its plan keeps, or room left over never
/// becomes resident memory and every token reads from storage. The `astro_expert_cache`
/// capacity remains the only refusal.
#[must_use]
pub const fn should_commit_mandatory_complete_layer(
    _route_token_count: i32,
    production_default_paging: bool,
    layer_state: super::ExpertLayerState,
) -> bool {
    production_default_paging && matches!(layer_state, super::ExpertLayerState::Kept)
}

/// Keep routed experts after a mandatory read so the next forward can hit them
/// (hot-expert holding). Overflow is handled by evicting the least-used
/// retained page, not by refusing to hold what this forward used.
///
/// A layer the reply's plan leaves absent keeps nothing new: it is not
/// refilled behind the plan's back.
#[must_use]
pub const fn should_commit_mandatory_routed_page(
    _route_token_count: i32,
    production_default_paging: bool,
    layer_state: super::ExpertLayerState,
    _layer_has_no_retained_page: bool,
) -> bool {
    if !production_default_paging {
        return false;
    }
    !matches!(layer_state, super::ExpertLayerState::Absent)
}

/// Whether a multi-token prefill chunk's operation-local routed-union stream
/// may be offered to retained ownership as a complete layer.
///
/// The host-visible prefill path streams the chunk's routed union, and a
/// chunk large enough to route every expert of a layer has already paid the
/// complete-layer read cost; discarding it re-reads the same bytes from
/// storage on every later chunk (the 2026-10-23 Ornith flight streamed
/// 389,018,225,728 bytes for one 23,976-token prompt — about 12.6 passes
/// over the model's whole expert set — while 26.95 GB of ceiling sat unused).
/// The offer is honest only when the plan funds the layer's complete claim
/// (the layer should be kept whole) and the union covers the layer exactly;
/// budget admission inside the `astro_expert_cache` remains the only refusal path.
#[must_use]
pub const fn should_commit_multi_token_union_as_complete_layer(
    route_token_count: i32,
    production_default_paging: bool,
    layer_state: super::ExpertLayerState,
    routed_expert_count: usize,
    expert_capacity: usize,
) -> bool {
    route_token_count > 1
        && production_default_paging
        && expert_capacity > 0
        && routed_expert_count == expert_capacity
        && matches!(layer_state, super::ExpertLayerState::Kept)
}

/// Whether a partial prefill union may seed a funded complete-layer table.
///
/// A multi-token chunk often routes most, but not all, experts of a layer.
/// Retaining that union lets the next read fetch only the remainder and finish
/// the table. `astro_expert_cache` admission still enforces the actual retained-byte ceiling.
#[must_use]
pub const fn should_seed_partial_layer_on_prefill_read(
    route_token_count: i32,
    production_default_paging: bool,
    layer_state: super::ExpertLayerState,
    resident_expert_count: usize,
    routed_expert_count: usize,
    expert_capacity: usize,
) -> bool {
    route_token_count > 1
        && production_default_paging
        && resident_expert_count == 0
        && expert_capacity > 0
        && routed_expert_count > 0
        && routed_expert_count < expert_capacity
        && matches!(layer_state, super::ExpertLayerState::Kept)
}

/// Whether a prefill read that must stream some of a layer's routed experts
/// should extend the read to every non-resident expert of the layer, so the
/// warm table converges to a complete layer instead of waiting for a lucky
/// full-coverage union.
///
/// Partial tables are the steady state of skewed routing: unions rarely
/// select every expert, so coverage grows by a few experts per chunk while
/// every chunk still re-reads the missing ones (the 2026-10-23 flights held
/// ~17 GB of partial tables yet re-read 27.8 GB per request). When the plan
/// funds the layer's complete claim, one read of the remaining experts pays
/// for itself by turning every later chunk into a full table hit.
#[must_use]
pub const fn should_complete_partial_layer_on_prefill_read(
    production_default_paging: bool,
    layer_state: super::ExpertLayerState,
    resident_expert_count: usize,
    expert_capacity: usize,
) -> bool {
    production_default_paging
        && expert_capacity > 0
        && resident_expert_count > 0
        && resident_expert_count < expert_capacity
        && matches!(layer_state, super::ExpertLayerState::Kept)
}

/// Structural slot cap for one decode warm table (hot-expert holding).
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
