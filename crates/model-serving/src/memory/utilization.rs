//! Why is part of the MLX memory ceiling unused?
//!
//! Issue #507. The status surface reports unused headroom by subtracting active
//! memory from the ceiling, which answers *how much* but never *why*. Without a
//! reason, an idle gigabyte is indistinguishable from a deliberate reservation,
//! and optimizing expert residency against a bare number is guesswork.
//!
//! # The headroom is not one thing
//!
//! Unused headroom decomposes into owners with opposite responses:
//!
//! - **Reserved for other owners** — context growth, activation workspace, and
//!   the routed-page stream slot, each charged only for the part of its reserve
//!   that transient work does not already occupy. These bytes are held
//!   deliberately. Spending
//!   them on expert weights invites an eviction or an out-of-memory during the
//!   next forward, so the correct response is to leave them alone.
//! - **Unseated expert entitlement** — expert budget the budget owner granted
//!   but warming never filled. This is genuinely recoverable, and it is the only
//!   term a residency change should target.
//! - **Unexplained** — whatever is left after every named owner is charged. A
//!   non-trivial residual means a decision point exists that this module does
//!   not name yet. Surfacing it is the point: an unnamed owner must be visible
//!   rather than silently rounded into a deliberate reservation.
//!
//! # Accounting identity
//!
//! The budget owner composes the expert entitlement as a documented subtraction
//! (`memory/budget/ram.rs`), so the identity below is not re-derived here — it
//! is the same split, rearranged as "reserved minus actually used" per owner:
//!
//! ```text
//! unused = model_core_slack
//!        + reserved_context_growth
//!        + reserved_activation_and_workspace
//!        + unseated_expert_entitlement
//!        + unexplained
//! ```
//!
//! This module observes. It never decides residency.

use crate::memory::MlxRamBudgetSnapshot;
use crate::memory::telemetry::MlxActiveMemoryBreakdown;

/// The reason-tagged split of one MLX ceiling's unused headroom.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MemoryCeilingUtilization {
    /// Total active-memory ceiling the plan was composed against.
    pub mlx_active_memory_ceiling_bytes: u64,
    /// Measured active memory at the same instant as the breakdown.
    pub active_memory_bytes: u64,
    /// Ceiling minus active: the bytes a reader sees sitting idle.
    pub unused_headroom_bytes: u64,
    /// Model-core reserve that the loaded core does not occupy.
    pub reserved_model_core_slack_bytes: u64,
    /// Context-window reserve that persistent request state does not occupy.
    pub reserved_context_growth_bytes: u64,
    /// Activation workspace, stream slot, and other fixed reserves.
    pub reserved_activation_and_workspace_bytes: u64,
    /// Expert entitlement the budget owner granted but warming never filled.
    pub unseated_expert_entitlement_bytes: u64,
    /// Headroom not accounted for by any owner named above.
    pub unexplained_headroom_bytes: u64,
    /// Headroom an owner consumed beyond its reserve. Nonzero means a named
    /// owner overran, which the clamped terms alone would hide.
    pub owner_overrun_bytes: u64,
}

impl MemoryCeilingUtilization {
    /// Splits one ceiling measurement into named owners plus a residual.
    ///
    /// The expert payload comes from the breakdown itself, not from cache
    /// bookkeeping: the measured attribution is the truthful physically
    /// resident figure, and using one source for both the unattributed-active
    /// subtraction and the entitlement keeps the identity exact.
    #[must_use]
    pub fn compose(
        budget_snapshot: MlxRamBudgetSnapshot,
        active_memory_bytes: u64,
        active_breakdown: MlxActiveMemoryBreakdown,
    ) -> Self {
        let mlx_active_memory_ceiling_bytes = budget_snapshot.mlx_active_memory_ceiling_bytes;
        let unused_headroom_bytes =
            mlx_active_memory_ceiling_bytes.saturating_sub(active_memory_bytes);
        // Each owner contributes the part of its reserve that is not yet
        // occupied. Saturating at zero keeps an owner that overruns its reserve
        // from producing a negative contribution, which would otherwise shrink
        // the residual and hide the overrun.
        let reserved_model_core_slack_bytes = budget_snapshot
            .model_core_payload_bytes
            .saturating_sub(active_breakdown.model_core_payload_bytes);
        let reserved_context_growth_bytes = budget_snapshot
            .context_window_reserve_bytes
            .saturating_sub(active_breakdown.context_state_payload_bytes);
        // Active bytes the breakdown cannot attribute to a named owner are
        // transient work living inside the activation and workspace reserve.
        // Charging them against that reserve is what keeps the identity exact:
        // without it, the same bytes would count twice, once inside active
        // memory and once as if the reserve were still free.
        let attributed_active_bytes = active_breakdown
            .expert_payload_bytes
            .saturating_add(active_breakdown.model_core_payload_bytes)
            .saturating_add(active_breakdown.context_state_payload_bytes);
        let unattributed_active_bytes = active_memory_bytes.saturating_sub(attributed_active_bytes);
        let reserved_activation_and_workspace_bytes = budget_snapshot
            .activation_headroom_bytes
            .saturating_add(budget_snapshot.complete_layer_stream_slot_bytes)
            .saturating_add(budget_snapshot.other_fixed_bytes)
            .saturating_sub(unattributed_active_bytes);
        let unseated_expert_entitlement_bytes = budget_snapshot
            .retained_expert_budget_bytes
            .saturating_sub(active_breakdown.expert_payload_bytes);
        let named_headroom_bytes = reserved_model_core_slack_bytes
            .saturating_add(reserved_context_growth_bytes)
            .saturating_add(reserved_activation_and_workspace_bytes)
            .saturating_add(unseated_expert_entitlement_bytes);
        let unexplained_headroom_bytes = unused_headroom_bytes.saturating_sub(named_headroom_bytes);
        // Clamping a negative term to zero inflates the named total above what
        // the headroom can explain. That excess is the overrun, and reporting
        // it keeps an owner exceeding its reserve visible instead of silent.
        let owner_overrun_bytes = named_headroom_bytes.saturating_sub(unused_headroom_bytes);
        Self {
            mlx_active_memory_ceiling_bytes,
            active_memory_bytes,
            unused_headroom_bytes,
            reserved_model_core_slack_bytes,
            reserved_context_growth_bytes,
            reserved_activation_and_workspace_bytes,
            unseated_expert_entitlement_bytes,
            unexplained_headroom_bytes,
            owner_overrun_bytes,
        }
    }

    /// Share of unused headroom that only a residency change could recover.
    ///
    /// The rest is held for other owners by policy, so a reader comparing
    /// "unused" against "recoverable" learns how much of the gap is even worth
    /// pursuing before writing any optimization.
    #[must_use]
    pub fn recoverable_share_of_unused_headroom(&self) -> f64 {
        if self.unused_headroom_bytes == 0 {
            return 0.0;
        }
        (self.unseated_expert_entitlement_bytes as f64 / self.unused_headroom_bytes as f64) * 100.0
    }

    /// True when every unused byte has a named owner and no owner overran.
    #[must_use]
    pub fn is_fully_explained(&self) -> bool {
        self.unexplained_headroom_bytes == 0 && self.owner_overrun_bytes == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIGABYTE: u64 = 1_000_000_000;

    fn budget_snapshot(
        ceiling_bytes: u64,
        retained_expert_budget_bytes: u64,
    ) -> MlxRamBudgetSnapshot {
        let non_expert_bytes = ceiling_bytes - retained_expert_budget_bytes;
        MlxRamBudgetSnapshot {
            mlx_active_memory_ceiling_bytes: ceiling_bytes,
            model_core_payload_bytes: 2_600_000_000,
            context_window_reserve_bytes: 2_000_000_000,
            activation_headroom_bytes: non_expert_bytes
                - 2_600_000_000
                - 2_000_000_000
                - 200_000_000,
            complete_layer_stream_slot_bytes: 200_000_000,
            other_fixed_bytes: 0,
            retained_expert_budget_bytes,
        }
    }

    #[test]
    fn should_charge_every_unused_byte_to_a_named_owner_when_owners_are_within_their_reserves() {
        let budget = budget_snapshot(23 * GIGABYTE, 15 * GIGABYTE);
        let breakdown = MlxActiveMemoryBreakdown {
            expert_payload_bytes: 12_900_000_000,
            model_core_payload_bytes: 2_600_000_000,
            context_state_payload_bytes: 690_000_000,
        };
        let active_memory_bytes = 16_620_000_000;

        let utilization = MemoryCeilingUtilization::compose(budget, active_memory_bytes, breakdown);

        assert_eq!(
            utilization.unused_headroom_bytes,
            23 * GIGABYTE - active_memory_bytes
        );
        assert_eq!(utilization.reserved_model_core_slack_bytes, 0);
        assert_eq!(
            utilization.reserved_context_growth_bytes,
            2_000_000_000 - 690_000_000
        );
        // The activation and workspace reserve is 3.4 GB, and 0.43 GB of active
        // memory is unattributed transient work living inside it.
        assert_eq!(
            utilization.reserved_activation_and_workspace_bytes,
            3_400_000_000 - 430_000_000
        );
        assert_eq!(
            utilization.unseated_expert_entitlement_bytes,
            15 * GIGABYTE - 12_900_000_000
        );
        assert!(
            utilization.is_fully_explained(),
            "named owners must reconstruct unused headroom: {utilization:?}"
        );
    }

    #[test]
    fn should_surface_an_owner_overrun_instead_of_hiding_it_behind_its_reserve() {
        let budget = budget_snapshot(23 * GIGABYTE, 15 * GIGABYTE);
        // Context state of 3.0 GB overruns its 2.0 GB reserve by 1.0 GB. The
        // clamped context term reads zero, so the reported named total exceeds
        // the headroom it explains by exactly the overrun.
        let breakdown = MlxActiveMemoryBreakdown {
            expert_payload_bytes: 12_900_000_000,
            model_core_payload_bytes: 2_600_000_000,
            context_state_payload_bytes: 3_000_000_000,
        };

        let utilization = MemoryCeilingUtilization::compose(budget, 18_500_000_000, breakdown);

        assert_eq!(utilization.reserved_context_growth_bytes, 0);
        assert_eq!(
            utilization.unexplained_headroom_bytes, 0,
            "a clamp must not manufacture unexplained headroom"
        );
        assert!(
            !utilization.is_fully_explained(),
            "an overrun must surface through the explained flag"
        );
        assert_eq!(utilization.owner_overrun_bytes, 1_000_000_000);
    }

    #[test]
    fn should_separate_recoverable_entitlement_from_deliberately_reserved_headroom() {
        let budget = budget_snapshot(23 * GIGABYTE, 15 * GIGABYTE);
        let breakdown = MlxActiveMemoryBreakdown {
            expert_payload_bytes: 12_900_000_000,
            model_core_payload_bytes: 2_600_000_000,
            context_state_payload_bytes: 690_000_000,
        };

        let utilization = MemoryCeilingUtilization::compose(budget, 16_620_000_000, breakdown);

        // Only the unseated entitlement is a residency target; the context and
        // workspace reserves must stay free for their own owners.
        assert!(
            utilization.recoverable_share_of_unused_headroom() < 100.0,
            "reserves held for other owners must not read as recoverable"
        );
        assert!(
            utilization.recoverable_share_of_unused_headroom() > 0.0,
            "unseated entitlement must read as recoverable"
        );
    }

    // The resident 25 GB decode-retention cell of issue #1236, byte for byte.
    // After the repeated cached request, the retained live prefix (issue
    // #1167) keeps ~153.5 MB of KV active, while the finalized plan protects
    // only ~27.6 MB of one-decode-step transient work. The budget partitions
    // the ceiling exactly, so a partitioned identity can only break through a
    // clamped term: zeroing the live context state charged the retained
    // prefix to the activation-and-workspace reserve, whose clamp manufactured
    // a 125,881,648-byte phantom owner overrun. Attributing the retained
    // state to the context owner closes the identity exactly.
    #[test]
    fn should_attribute_retained_live_context_bytes_to_the_context_owner_so_the_finalized_split_closes()
     {
        let expert_payload_bytes = 19_595_788_288;
        let model_core_payload_bytes = 2_568_911_104;
        let retained_context_payload_bytes = 153_523_590;
        let decode_activation_reserve_bytes = 11_159_842;
        let routed_page_slot_bytes = 16_482_100;
        let decode_transient_reserve_bytes =
            decode_activation_reserve_bytes + routed_page_slot_bytes;
        let budget = MlxRamBudgetSnapshot {
            mlx_active_memory_ceiling_bytes: 25 * GIGABYTE,
            model_core_payload_bytes,
            context_window_reserve_bytes: GIGABYTE,
            activation_headroom_bytes: decode_activation_reserve_bytes,
            complete_layer_stream_slot_bytes: routed_page_slot_bytes,
            other_fixed_bytes: 0,
            retained_expert_budget_bytes: 25 * GIGABYTE
                - model_core_payload_bytes
                - GIGABYTE
                - decode_transient_reserve_bytes,
        };
        let active_memory_bytes =
            expert_payload_bytes + model_core_payload_bytes + retained_context_payload_bytes;

        // The defect shape: the finalized breakdown reported zero context
        // state, so the retained prefix landed in unattributed active bytes
        // and overran the transient reserve.
        let unattributed_breakdown = MlxActiveMemoryBreakdown {
            expert_payload_bytes,
            model_core_payload_bytes,
            context_state_payload_bytes: 0,
        };
        let defective =
            MemoryCeilingUtilization::compose(budget, active_memory_bytes, unattributed_breakdown);

        assert_eq!(
            defective.reserved_activation_and_workspace_bytes, 0,
            "the transient reserve must clamp when live context is misattributed"
        );
        assert_eq!(defective.owner_overrun_bytes, 125_881_648);
        assert!(
            !defective.is_fully_explained(),
            "the misattributed shape must not read as fully explained"
        );

        // The corrected shape: the retained prefix is attributed to the
        // context owner it belongs to, and every byte has a named owner.
        let attributed_breakdown = MlxActiveMemoryBreakdown {
            expert_payload_bytes,
            model_core_payload_bytes,
            context_state_payload_bytes: retained_context_payload_bytes,
        };
        let corrected =
            MemoryCeilingUtilization::compose(budget, active_memory_bytes, attributed_breakdown);

        assert_eq!(
            corrected.reserved_context_growth_bytes,
            GIGABYTE - retained_context_payload_bytes,
        );
        assert_eq!(
            corrected.reserved_activation_and_workspace_bytes, decode_transient_reserve_bytes,
            "with context attributed, the transient reserve is fully unoccupied"
        );
        assert_eq!(corrected.unexplained_headroom_bytes, 0);
        assert_eq!(corrected.owner_overrun_bytes, 0);
        assert!(
            corrected.is_fully_explained(),
            "attributing live context must close the identity: {corrected:?}"
        );
    }
}
