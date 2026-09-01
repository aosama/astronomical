//! Memory policy for request-scoped multi-token-prediction admission.
//!
//! The projection and depth selection here are pure arithmetic over measured
//! byte facts supplied by the execution owner; family code enacts the result
//! and must not re-derive it.

use thiserror::Error;

use super::mtp_draft_depth::MtpDraftDepth;

/// Why one request temporarily executes below its resolved MTP depth.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MtpDepthDowngradeReason {
    /// The response output window cannot fit the requested depth's proposals.
    OutputWindow,
    /// The remaining context window cannot fit the requested depth's rows.
    ContextWindow,
    /// The thinking budget cannot cover the requested depth's verification.
    ThinkingWindow,
    /// The projected bytes for the deepest depth do not fit the allowance.
    Memory,
}

/// Exact request-owned and operation-local bytes projected for one MTP depth.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MtpMemoryProjection {
    depth: MtpDraftDepth,
    mtp_persistent_growth_bytes: usize,
    target_persistent_growth_bytes: usize,
    target_expert_page_reservation_bytes: usize,
    boundary_snapshot_bytes: usize,
    transient_array_bytes: usize,
    committed_owner_bytes: usize,
    total_required_bytes: usize,
}

impl MtpMemoryProjection {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        depth: MtpDraftDepth,
        mtp_persistent_growth_bytes: usize,
        target_persistent_growth_bytes: usize,
        target_expert_page_reservation_bytes: usize,
        boundary_snapshot_bytes: usize,
        transient_array_bytes: usize,
        committed_owner_bytes: usize,
    ) -> Result<Self, MtpMemoryProjectionError> {
        let total_required_bytes = [
            mtp_persistent_growth_bytes,
            target_persistent_growth_bytes,
            target_expert_page_reservation_bytes,
            boundary_snapshot_bytes,
            transient_array_bytes,
            committed_owner_bytes,
        ]
        .into_iter()
        .try_fold(0_usize, usize::checked_add)
        .ok_or(MtpMemoryProjectionError)?;
        Ok(Self {
            depth,
            mtp_persistent_growth_bytes,
            target_persistent_growth_bytes,
            target_expert_page_reservation_bytes,
            boundary_snapshot_bytes,
            transient_array_bytes,
            committed_owner_bytes,
            total_required_bytes,
        })
    }

    #[must_use]
    pub const fn depth(&self) -> MtpDraftDepth {
        self.depth
    }

    #[must_use]
    pub const fn mtp_persistent_growth_bytes(&self) -> usize {
        self.mtp_persistent_growth_bytes
    }

    #[must_use]
    pub const fn target_persistent_growth_bytes(&self) -> usize {
        self.target_persistent_growth_bytes
    }

    #[must_use]
    pub const fn total_required_bytes(&self) -> usize {
        self.total_required_bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[error("MTP memory projection overflowed")]
pub struct MtpMemoryProjectionError;

/// One descending-depth admission candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MtpMemoryCandidate {
    depth: MtpDraftDepth,
    required_bytes: usize,
}

impl MtpMemoryCandidate {
    #[must_use]
    pub const fn new(depth: MtpDraftDepth, required_bytes: usize) -> Self {
        Self {
            depth,
            required_bytes,
        }
    }
}

/// Pure depth selection result; `None` means target-only remains safe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MtpDepthSelection {
    effective_depth: Option<MtpDraftDepth>,
    downgrade_reason: Option<MtpDepthDowngradeReason>,
}

impl MtpDepthSelection {
    #[must_use]
    pub const fn effective_depth(self) -> Option<MtpDraftDepth> {
        self.effective_depth
    }

    #[must_use]
    pub const fn downgrade_reason(self) -> Option<MtpDepthDowngradeReason> {
        self.downgrade_reason
    }
}

/// Namespace owner for multi-token-prediction memory admission decisions.
pub struct MtpAdmission;

impl MtpAdmission {
    /// Exact operation-local logits, hidden rows, and decision vectors for one
    /// verification window.
    ///
    /// Boundary snapshots are excluded so admission can charge them once as a
    /// separate owner. Proposal logits, verifier logits, hidden rows, and
    /// decision ids share one lazy graph and therefore coexist at the single
    /// completion boundary.
    pub fn verification_transient_array_bytes(
        depth: MtpDraftDepth,
        vocabulary_size: usize,
        hidden_size: usize,
        verification_samples_acceptance: bool,
    ) -> Result<usize, MtpMemoryProjectionError> {
        let draft_count = usize::from(depth.get());
        let verifier_row_count = draft_count.checked_add(1).ok_or(MtpMemoryProjectionError)?;
        let logits_bytes = draft_count
            .checked_add(verifier_row_count)
            .and_then(|row_count| row_count.checked_mul(vocabulary_size))
            .and_then(|element_count| element_count.checked_mul(std::mem::size_of::<f32>()))
            .ok_or(MtpMemoryProjectionError)?;
        // Sampled acceptance keeps masked and probability copies of the same rows,
        // plus the residual `max(0, p − q)` rows, resident beside the logits.
        let sampled_distribution_bytes = if verification_samples_acceptance {
            let sampled_row_count = 2 * draft_count + 2 * verifier_row_count;
            let sampled_row_bytes = sampled_row_count
                .checked_mul(vocabulary_size)
                .and_then(|element_count| element_count.checked_mul(std::mem::size_of::<f32>()))
                .and_then(|bytes| bytes.checked_add(draft_count * std::mem::size_of::<f32>()))
                .ok_or(MtpMemoryProjectionError)?;
            sampled_row_bytes
        } else {
            0
        };
        // Hidden rows remain BF16 (2 bytes) through verification.
        let hidden_row_bytes = draft_count
            .checked_add(verifier_row_count)
            .and_then(|row_count| row_count.checked_mul(hidden_size))
            .and_then(|element_count| element_count.checked_mul(2))
            .ok_or(MtpMemoryProjectionError)?;
        let decision_bytes = draft_count
            .checked_add(verifier_row_count)
            .and_then(|element_count| element_count.checked_mul(std::mem::size_of::<u32>()))
            .ok_or(MtpMemoryProjectionError)?;
        [
            logits_bytes,
            sampled_distribution_bytes,
            hidden_row_bytes,
            decision_bytes,
        ]
        .into_iter()
        .try_fold(0_usize, usize::checked_add)
        .ok_or(MtpMemoryProjectionError)
    }

    /// Chooses the deepest candidate that fits without rejecting target-only work.
    #[must_use]
    pub fn select_effective_depth(
        descending_candidates: &[MtpMemoryCandidate],
        available_bytes: usize,
    ) -> MtpDepthSelection {
        let requested_depth = descending_candidates
            .first()
            .map(|candidate| candidate.depth);
        let effective_depth = descending_candidates
            .iter()
            .find(|candidate| candidate.required_bytes <= available_bytes)
            .map(|candidate| candidate.depth);
        MtpDepthSelection {
            effective_depth,
            downgrade_reason: if effective_depth == requested_depth {
                None
            } else {
                Some(MtpDepthDowngradeReason::Memory)
            },
        }
    }
}
