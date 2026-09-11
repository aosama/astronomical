//! Whether just-read experts may stay in the retained cache.
//!
//! Two caching strategies live here, both answering "may this page stay":
//! whole-layer caching seats an entire decoder layer's experts or none of
//! them, and hot-expert caching keeps the individual routed experts the
//! router keeps choosing. In "hot-expert," *hot* means routing frequency, not
//! temperature.

/// Seat a complete layer after a mandatory read when the plan asked for it.
///
/// Prefill does this for multi-token chunks. Decode must do it too after an
/// atomic complete-owner demote, or leftover budget never becomes resident RAM
/// and every generate token streams from SSD.
#[must_use]
pub const fn should_commit_mandatory_complete_layer(
    _route_token_count: i32,
    production_default_paging: bool,
    residency_target: Option<super::ExpertLayerResidencyTarget>,
) -> bool {
    production_default_paging
        && matches!(
            residency_target,
            Some(super::ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead)
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
