//! How much expert RAM decode-warm may pin, and when prefill recovery must demote.
//!
//! # Mental model for a later reader
//!
//! Mixture-of-experts models own many specialist weight groups. "Complete
//! residency" means every specialist sits in RAM as one atomic owner. That
//! owner cannot be nibbled: paging cannot evict three specialists and keep the
//! rest. The only legal shrink is "demote the whole owner, then stream pages".
//!
//! Prefill activations are large. If they do not fit beside the complete owner,
//! request pressure demotes that owner and may install a temporary retained-page
//! cap so the rest of the prompt can finish. That cap is not the decode budget.
//!
//! After prefill, activations shrink. Decode-warm may then spend the leftover
//! composed RAM plan on demand-selected pages. Restoring the complete owner is
//! a separate promotion admit in `expert_residency_transition.rs`. This file
//! only answers two policy questions:
//!
//! 1. How many bytes may decode-warm pin?
//! 2. Must native prefill recovery demote the complete owner before retry?

/// Bytes decode-warm may pin after composing the leftover retained-expert budget.
///
/// `planned_retained_expert_budget_bytes` is the leftover after subtracting
/// model core, learned context reserve, activation headroom, and one complete
/// layer-loading slot. That leftover is how generation reclaims RAM the user
/// already granted.
///
/// `requested_retained_expert_payload_bytes` is an optional smaller caller
/// ceiling. Decode handoff passes `u64::MAX` so the composed plan wins.
/// Tests and diagnostic callers may pass a smaller number.
///
/// Do not shrink this to "one routed top-K page per layer" after request
/// pressure. That old 1 GB working-set rule rejected every useful page and
/// forced solid-state-drive streaming despite free RAM.
#[must_use]
pub fn retained_expert_fill_budget_bytes(
    planned_retained_expert_budget_bytes: u64,
    requested_retained_expert_payload_bytes: u64,
) -> u64 {
    planned_retained_expert_budget_bytes.min(requested_retained_expert_payload_bytes)
}

/// Complete resident experts are one atomic owner.
///
/// Paged page reclamation cannot nibble them. Native prefill recovery that
/// needs expert bytes back must therefore demote that owner before retry.
/// Returning `false` means the model is already paged, so recovery can reclaim
/// individual retained pages instead.
#[must_use]
pub const fn prefill_recovery_must_demote_complete_resident_owner(
    complete_experts_are_resident: bool,
) -> bool {
    complete_experts_are_resident
}
