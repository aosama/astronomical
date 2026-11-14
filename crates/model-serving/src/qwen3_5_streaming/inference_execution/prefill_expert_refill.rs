//! Puts expert layers back in memory after a prefill memory recovery, once
//! the engine knows what a chunk really needs.
//!
//! A prefill memory recovery sheds every retained expert and freezes retention
//! at zero, because at that moment the engine has no evidence of how much
//! workspace a chunk takes: the failed chunk never completed, and the reserve
//! in force was the static floor or a smaller chunk's observation. Keeping
//! the freeze until the decode handoff was safe, but it made the first long
//! prompt on a model whose real chunk workspace exceeds that floor run its
//! whole remaining prefill with nothing in memory. Measured on the 122B at a
//! 32 GiB ceiling (2026-11-14): 3.6 GB active of 34.4 GB allowed, every one of
//! 48 layers re-read from storage for every chunk, while the planner, which
//! by then had the measured 6.4 GB workspace, wanted 29 complete layers kept.
//! The same prompt sent again was fast only because the next request started
//! unfrozen with that measurement.
//!
//! The retried chunk is the evidence. Once a chunk at least as large as any
//! chunk still to come has completed, and its measurement has raised the
//! workspace reserve above the one in force when the chunk failed, every
//! later plan is sized by a real number, so the freeze is lifted and the
//! contract is reopened from the measured leftover. A measurement that
//! taught nothing new leaves the freeze alone: refilling against the same
//! reserve would rebuild the state that just failed. That is the case with
//! performance attribution on, where a fully streamed chunk records no
//! workspace at all.
//!
//! The refill happens at most once per request. If a chunk then runs out of
//! memory anyway, recovery sheds and freezes again and the freeze stays until
//! decode, which is the behaviour before this module existed.

use astronomical_ipc_protocol::RequestId;

use crate::MemoryPhase;

use super::Qwen3_5EngineState;
use super::engine_request::Qwen3_5EngineRequest;
use super::prefill_memory_recovery::MAXIMUM_PREFILL_MEMORY_RECOVERY_ATTEMPTS_PER_REQUEST;

/// Where one request stands with refilling expert memory after a recovery.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(in crate::qwen3_5_streaming) enum PostRecoveryExpertRefill {
    /// No refill has happened; one is allowed when a recovery froze retention.
    #[default]
    NotAttempted,
    /// The freeze was lifted on measured evidence and layers are kept again.
    /// A memory failure now is the refill's doing, so it may be recovered.
    Active,
    /// A chunk failed after the refill. Retention stays frozen until decode.
    Spent,
}

impl PostRecoveryExpertRefill {
    /// The state after a recovery shed and froze retention: an active refill
    /// is used up, and a request that never refilled keeps its one chance.
    pub(super) const fn after_recovery(self) -> Self {
        match self {
            Self::Active | Self::Spent => Self::Spent,
            Self::NotAttempted => Self::NotAttempted,
        }
    }

    /// How many prefill memory recoveries the request may spend in this
    /// state. A failure while a refill is active earns one more: kept layers,
    /// not the prompt's own workspace, are again the likely consumer, and
    /// that recovery ends the refill, so the count stays bounded.
    pub(super) const fn allowed_recovery_attempt_count(self) -> u8 {
        match self {
            Self::Active => MAXIMUM_PREFILL_MEMORY_RECOVERY_ATTEMPTS_PER_REQUEST.saturating_add(1),
            Self::NotAttempted | Self::Spent => {
                MAXIMUM_PREFILL_MEMORY_RECOVERY_ATTEMPTS_PER_REQUEST
            }
        }
    }
}

/// What a completed chunk tells the engine about the rest of the prompt.
#[derive(Clone, Copy, Debug)]
pub(super) struct CompletedChunkEvidence {
    pub(super) completed_chunk_token_count: usize,
    pub(super) remaining_prefill_token_count: usize,
    pub(super) maximum_chunk_token_count: usize,
    /// The prefill workspace reserve in force when the chunk failed.
    pub(super) planned_workspace_bytes_at_recovery: u64,
    /// The reserve now, after the completed chunk was measured.
    pub(super) planned_workspace_bytes_now: u64,
}

/// Whether a completed chunk is evidence enough to keep expert layers again.
///
/// The completed chunk must be at least as large as every chunk still to
/// come, because the workspace reserve resolves from observations at or
/// below the planned chunk size: a smaller chunk's measurement would leave
/// the next, larger one under-reserved exactly as before the recovery. And
/// the reserve must have grown since the failure, or the refill would be
/// planned against the number that just proved too small.
pub(super) const fn completed_chunk_measures_remaining_prefill(
    post_recovery_expert_refill: PostRecoveryExpertRefill,
    evidence: CompletedChunkEvidence,
) -> bool {
    let largest_remaining_chunk_token_count =
        if evidence.remaining_prefill_token_count < evidence.maximum_chunk_token_count {
            evidence.remaining_prefill_token_count
        } else {
            evidence.maximum_chunk_token_count
        };
    matches!(
        post_recovery_expert_refill,
        PostRecoveryExpertRefill::NotAttempted
    ) && evidence.remaining_prefill_token_count > 0
        && evidence.completed_chunk_token_count >= largest_remaining_chunk_token_count
        && evidence.planned_workspace_bytes_now > evidence.planned_workspace_bytes_at_recovery
}

impl Qwen3_5EngineState {
    /// The workspace the budget reserves for one full prefill chunk: the
    /// static floor until a chunk of that size has been measured.
    pub(super) fn planned_prefill_workspace_bytes(&self) -> u64 {
        let Some(model) = self.model.as_ref() else {
            return 0;
        };
        let maximum_chunk_token_count = u64::try_from(
            self.prompt_processing_chunk_sizer
                .maximum_prompt_processing_chunk_size_tokens(),
        )
        .unwrap_or(u64::MAX);
        model
            .mlx_ram_budget()
            .activation_headroom_bytes(MemoryPhase::Prefill, maximum_chunk_token_count)
    }

    /// Lifts a prefill memory-recovery freeze after a chunk that measured the
    /// workspace of everything still to come, so the rest of the prompt keeps
    /// expert layers in memory again. Does nothing when retention is not
    /// frozen, which is every request that never needed a recovery.
    pub(super) fn refill_expert_memory_after_measured_prefill_chunk(
        &self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
        completed_chunk_token_count: usize,
    ) {
        let remaining_prefill_token_count = active_request
            .input_token_ids
            .len()
            .saturating_sub(active_request.prefill_cursor);
        let evidence = CompletedChunkEvidence {
            completed_chunk_token_count,
            remaining_prefill_token_count,
            maximum_chunk_token_count: self
                .prompt_processing_chunk_sizer
                .maximum_prompt_processing_chunk_size_tokens(),
            planned_workspace_bytes_at_recovery: active_request
                .engine_extension
                .planned_prefill_workspace_bytes_at_recovery,
            planned_workspace_bytes_now: self.planned_prefill_workspace_bytes(),
        };
        if !completed_chunk_measures_remaining_prefill(
            active_request.engine_extension.post_recovery_expert_refill,
            evidence,
        ) {
            return;
        }
        let Some(model) = self.model.as_ref() else {
            return;
        };
        if !model.lift_prefill_memory_recovery_freeze() {
            return;
        }
        // The contract opened while frozen was packed against the reserve
        // that just proved too small. Clearing it makes the next chunk's
        // refresh open a new one from the measured leftover.
        model.clear_phase_aware_expert_residency_plan();
        active_request.engine_extension.post_recovery_expert_refill =
            PostRecoveryExpertRefill::Active;
        tracing::info!(
            request_id = request_id.value(),
            completed_chunk_token_count,
            remaining_prefill_token_count,
            planned_workspace_bytes_at_recovery = evidence.planned_workspace_bytes_at_recovery,
            planned_workspace_bytes_now = evidence.planned_workspace_bytes_now,
            "lifted the prefill memory-recovery freeze after a measured chunk; expert layers are kept again for the rest of the prompt"
        );
    }
}

#[cfg(test)]
#[path = "prefill_expert_refill_tests.rs"]
mod tests;
