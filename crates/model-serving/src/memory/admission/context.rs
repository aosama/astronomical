//! Admission: may this request's context growth begin?
//!
//! Families measure the live MLX active bytes and the exact decoder-state
//! growth; `ContextAdmissionRequirements::decide()` returns the shared
//! `MemoryAdmissionDecision`. The workspace byte functions here are the exact
//! persistent/temporary accounting that prefill, persistent-cache restore,
//! and complete-resident seating charge.

use crate::memory::{MemoryAdmissionDecision, MemoryBoundary};

/// Visible-answer KV reserve beyond the thinking allowance, in tokens.
///
/// One prefill chunk (the repo's canonical token granularity): long-form
/// answers beyond it still generate, with per-step KV growth admitted by the
/// adaptive growth guard instead of pre-reserved.
pub const VISIBLE_ANSWER_RESERVE_TOKENS: usize = 2_048;

/// Output tokens the generation-context memory reservation must cover.
///
/// Reserving the caller's full `maximum_output_token_count` charges
/// (max_output - thinking_budget - answer reserve) x per-token KV bytes as a
/// permanent phantom hold at admission. An agent client asking for 65,535
/// output tokens with a 16,384-token thinking budget pinned ~1.3 GB of KV
/// that never materialized, which pushed every turn of a long conversation
/// over the active-memory ceiling and forced a reclaim of warm expert layers
/// before each prefill (2026-10-16 incident). Generation beyond the reserved
/// count remains legal: that KV grows lazily and is admitted per step by the
/// growth guard, exactly like context growth beyond any other reserve.
#[must_use]
pub const fn bounded_output_reservation_token_count(
    maximum_output_token_count: usize,
    thinking_budget_token_count: Option<usize>,
) -> usize {
    let bounded_output_token_count = match thinking_budget_token_count {
        Some(thinking_budget_token_count) => {
            thinking_budget_token_count.saturating_add(VISIBLE_ANSWER_RESERVE_TOKENS)
        }
        None => VISIBLE_ANSWER_RESERVE_TOKENS,
    };
    if maximum_output_token_count < bounded_output_token_count {
        maximum_output_token_count
    } else {
        bounded_output_token_count
    }
}

/// Exact context and workspace categories supplied by execution owners.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContextAdmissionRequirements {
    /// Live MLX active bytes sampled before admitting context growth.
    pub current_active_memory_bytes: usize,
    /// Exact persistent decoder-state growth required by the operation.
    pub context_growth_bytes: usize,
    /// Largest expert page that may coexist with the context operation.
    pub expert_page_reservation_bytes: usize,
    /// Explicit temporary owner, such as prompt-cache reconstruction workspace.
    pub temporary_workspace_bytes: usize,
    /// Elastic paged-expert payload available for reclamation.
    pub retained_expert_payload_bytes: usize,
    /// Stable MLX active-memory ceiling resolved for the worker.
    pub active_memory_ceiling_bytes: usize,
    /// Whether the current expert owner is indivisible and must demote first.
    pub complete_experts_are_resident: bool,
}

impl ContextAdmissionRequirements {
    /// Returns the complete active-memory projection or `None` on overflow.
    #[must_use]
    pub fn projected_active_memory_bytes(self) -> Option<usize> {
        self.current_active_memory_bytes
            .checked_add(self.context_growth_bytes)?
            .checked_add(self.expert_page_reservation_bytes)?
            .checked_add(self.temporary_workspace_bytes)
    }

    /// Decides admission without mutating expert ownership.
    #[must_use]
    pub fn decide(self) -> MemoryAdmissionDecision {
        let Some(projected_active_memory_bytes) = self.projected_active_memory_bytes() else {
            return MemoryAdmissionDecision::Reject {
                boundary: MemoryBoundary::StableActiveCeiling,
                shortfall_bytes: u64::MAX,
            };
        };
        if projected_active_memory_bytes <= self.active_memory_ceiling_bytes {
            return MemoryAdmissionDecision::Admit;
        }
        if self.complete_experts_are_resident {
            return MemoryAdmissionDecision::DemoteCompleteResidency {
                reassess_after_demotion: true,
            };
        }
        let required_bytes =
            projected_active_memory_bytes.saturating_sub(self.active_memory_ceiling_bytes);
        if required_bytes <= self.retained_expert_payload_bytes {
            MemoryAdmissionDecision::Reclaim {
                required_bytes: u64::try_from(required_bytes).unwrap_or(u64::MAX),
            }
        } else {
            MemoryAdmissionDecision::Reject {
                boundary: MemoryBoundary::StableActiveCeiling,
                shortfall_bytes: u64::try_from(
                    required_bytes.saturating_sub(self.retained_expert_payload_bytes),
                )
                .unwrap_or(u64::MAX),
            }
        }
    }
}

/// Context-cache reconstruction temporarily owns both loaded and concatenated state.
///
/// `restored_context_token_count` is tokens that may already exist as prompt-cache
/// blocks. Future output-budget tokens have no blocks and must not be multiplied in.
#[must_use]
pub fn persistent_context_restore_workspace_bytes(
    context_memory_reservation_bytes_per_token: usize,
    restored_context_token_count: usize,
) -> Option<usize> {
    context_memory_reservation_bytes_per_token.checked_mul(restored_context_token_count)
}

/// Peak active memory while complete experts are already seated in `current_active_memory_bytes`.
///
/// Cache restore and full request KV are exclusive phases: restore finishes before
/// generation grows the output cache. Prefill layer-weight heuristics and SSD stream
/// slots are already inside the seated active snapshot, so they are not added again.
#[must_use]
pub fn seated_complete_expert_request_peak_active_memory_bytes(
    current_active_memory_bytes: usize,
    context_growth_bytes: usize,
    restore_overlap_workspace_bytes: usize,
    publication_workspace_bytes: usize,
) -> Option<usize> {
    let restore_phase_active_memory_bytes = current_active_memory_bytes
        .checked_add(restore_overlap_workspace_bytes)?
        .checked_add(publication_workspace_bytes)?;
    let serving_phase_active_memory_bytes = current_active_memory_bytes
        .checked_add(context_growth_bytes)?
        .checked_add(publication_workspace_bytes)?;
    Some(restore_phase_active_memory_bytes.max(serving_phase_active_memory_bytes))
}

/// Temporary workspace charged against a request: seated peak extras, or paging extras.
///
/// When complete experts are already in `current_active`, layer-weight activation
/// and SSD stream-slot bytes are ignored. Those weights are not a second owner.
/// In paged mode the admission projection separately reserves the largest expert
/// page as `expert_page_reservation_bytes`, so that reservation is the single
/// owner of streaming payload; charging a complete-layer scratch here as well
/// double-counted the same bytes (issue #1108).
#[must_use]
pub fn request_context_temporary_workspace_bytes(
    complete_experts_are_resident: bool,
    context_growth_bytes: usize,
    restore_overlap_workspace_bytes: usize,
    publication_workspace_bytes: usize,
    prefill_activation_workspace_bytes: usize,
) -> Option<usize> {
    if complete_experts_are_resident {
        // Seated experts do not pre-pay the chunk transient: every resident
        // prefill forward still allocates its activation working set on top
        // of the seated snapshot, so the chunk-scoped reserve rides along in
        // both modes. Measured 2026-10-16 on the 8-bit resident shape under
        // a wired budget (issue #1297): 2048-token chunks transiently held
        // 2.0-2.27 GB that a zero-workspace projection never saw, so a long
        // prompt admitted near the ceiling drove its live footprint across
        // the wired budget and the GPU failed the command buffer fatally.
        seated_complete_expert_request_temporary_workspace_bytes(
            context_growth_bytes,
            restore_overlap_workspace_bytes,
            publication_workspace_bytes,
        )?
        .checked_add(prefill_activation_workspace_bytes)
    } else {
        publication_workspace_bytes
            .checked_add(restore_overlap_workspace_bytes)?
            .checked_add(prefill_activation_workspace_bytes)
    }
}

/// Temporary workspace that makes `current + context_growth + temporary` equal the seated peak.
///
/// When request KV already covers prompt restore, this is publication workspace only.
#[must_use]
pub fn seated_complete_expert_request_temporary_workspace_bytes(
    context_growth_bytes: usize,
    restore_overlap_workspace_bytes: usize,
    publication_workspace_bytes: usize,
) -> Option<usize> {
    let exclusive_restore_beyond_context_growth_bytes =
        restore_overlap_workspace_bytes.saturating_sub(context_growth_bytes);
    publication_workspace_bytes.checked_add(exclusive_restore_beyond_context_growth_bytes)
}

/// Combines independently owned persistent growth categories.
#[must_use]
pub fn combined_persistent_growth_bytes(
    target_persistent_state_growth_bytes: usize,
    additional_persistent_state_growth_bytes: usize,
) -> Option<usize> {
    target_persistent_state_growth_bytes.checked_add(additional_persistent_state_growth_bytes)
}

/// Smallest safe idle ceiling after reclaiming elastic expert payload.
#[must_use]
pub const fn safe_minimum_active_memory_ceiling_bytes(
    current_idle_active_memory_bytes: u64,
    evictable_retained_expert_payload_bytes: u64,
    maximum_expert_page_reserve_bytes: u64,
) -> u64 {
    current_idle_active_memory_bytes
        .saturating_sub(evictable_retained_expert_payload_bytes)
        .saturating_add(maximum_expert_page_reserve_bytes)
}
