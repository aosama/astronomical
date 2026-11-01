//! Pure planning units for request-side persistent prompt-cache capture,
//! shared by the streaming and resident engines.
//!
//! Boundary validation, causal-input selection, block-key construction
//! parents, and the publication workspace decision previously lived as two
//! byte-identical copies inside the engines' capture functions; a correction
//! that landed in one engine silently missed the other. The engines now call
//! these single units, and each engine's capture function keeps only the
//! engine-specific extraction, measurement, and publication plumbing.

use crate::persistent_cache::block_causal_input::PersistentPromptCacheBlockCausalInput;
// Block-key parents exist only for the direct-MLX engines' capture helper
// below, so the imports ride its feature gate.
#[cfg(feature = "direct-mlx")]
use crate::persistent_cache::block_key::{
    PersistentPromptCacheBlockKey, PersistentPromptCacheBlockKeyError,
};
#[cfg(feature = "direct-mlx")]
use crate::persistent_cache::model_contract::PersistentPromptCacheModelContract;

/// Comfort margin below the active-memory ceiling at which block publication
/// stops wiping the MLX allocator cache. Publication materializes tens of MB;
/// only a ceiling within this margin of the live active memory needs the
/// reclaimed room.
pub(crate) const PUBLICATION_COMFORT_HEADROOM_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// The token window of one capture block, derived from a validated boundary
/// checkpoint.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) struct CaptureBlockWindow {
    pub(crate) block_start: usize,
    pub(crate) block_end: usize,
    pub(crate) block_index: usize,
}

/// Why a boundary checkpoint cannot become a capture block. The descriptions
/// are the exact user-facing strings the engines wrap into their persistence
/// failures, so planning and error construction cannot drift apart.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum CaptureBoundaryRejection {
    PositionOverflowed,
    PositionInvalid,
    BlockStartUnderflowed,
}

impl CaptureBoundaryRejection {
    pub(crate) fn description(self) -> &'static str {
        match self {
            CaptureBoundaryRejection::PositionOverflowed => {
                "prompt-cache boundary position overflowed"
            }
            CaptureBoundaryRejection::PositionInvalid => {
                "prompt-cache boundary position is invalid"
            }
            CaptureBoundaryRejection::BlockStartUnderflowed => {
                "prompt-cache block start underflowed"
            }
        }
    }
}

/// Derives one capture block's token window from a boundary checkpoint.
///
/// Boundary checkpoints are emitted by the successful forward; this recomputes
/// the absolute position and validates it before any caller slices user
/// tokens — never trust a model-specific checkpoint to be aligned implicitly.
/// The boundary must sit inside the successful prefill span, land on a whole
/// block multiple, and leave room for one full block below it.
pub(crate) fn capture_block_window(
    successful_prefill_start: usize,
    completed_prefill_chunk_tokens: usize,
    persistent_prompt_cache_block_token_count: usize,
    successful_prefill_end: usize,
) -> Result<CaptureBlockWindow, CaptureBoundaryRejection> {
    let Some(absolute_boundary) =
        successful_prefill_start.checked_add(completed_prefill_chunk_tokens)
    else {
        return Err(CaptureBoundaryRejection::PositionOverflowed);
    };
    if !absolute_boundary.is_multiple_of(persistent_prompt_cache_block_token_count)
        || absolute_boundary > successful_prefill_end
    {
        return Err(CaptureBoundaryRejection::PositionInvalid);
    }
    let Some(block_start) =
        absolute_boundary.checked_sub(persistent_prompt_cache_block_token_count)
    else {
        return Err(CaptureBoundaryRejection::BlockStartUnderflowed);
    };
    Ok(CaptureBlockWindow {
        block_start,
        block_end: absolute_boundary,
        block_index: block_start / persistent_prompt_cache_block_token_count,
    })
}

/// One block's causal-input resolution: an empty plan means every block's
/// decoder inputs are fully identified by its tokens and ancestry.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum BlockCausalInputSelection<'a> {
    Empty,
    FromPlan(&'a PersistentPromptCacheBlockCausalInput),
}

/// Why a causal-input plan cannot serve a captured block.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum CaptureCausalInputRejection {
    PlanDoesNotCoverBlock,
}

impl CaptureCausalInputRejection {
    pub(crate) fn description(self) -> &'static str {
        match self {
            CaptureCausalInputRejection::PlanDoesNotCoverBlock => {
                "prompt-cache causal input plan does not cover captured block"
            }
        }
    }
}

/// Resolves the causal input one captured block binds. Each block binds only
/// new causal inputs; descendants inherit them through ancestry. An empty plan
/// selects the empty input everywhere; a non-empty plan must cover the block's
/// index or the capture stops — a partial plan is a contract violation, not a
/// fallback case.
pub(crate) fn select_block_causal_input(
    persistent_prompt_cache_block_causal_inputs: &[PersistentPromptCacheBlockCausalInput],
    block_index: usize,
) -> Result<BlockCausalInputSelection<'_>, CaptureCausalInputRejection> {
    if persistent_prompt_cache_block_causal_inputs.is_empty() {
        return Ok(BlockCausalInputSelection::Empty);
    }
    match persistent_prompt_cache_block_causal_inputs.get(block_index) {
        Some(block_causal_input) => Ok(BlockCausalInputSelection::FromPlan(block_causal_input)),
        None => Err(CaptureCausalInputRejection::PlanDoesNotCoverBlock),
    }
}

/// Builds the identity key of one captured block: the root key for the first
/// captured block, chained from the restored parent for every later one.
/// Only the direct-MLX streaming engine drives block capture, so the helper
/// is gated with it.
#[cfg(feature = "direct-mlx")]
pub(crate) fn block_key_for_captured_block(
    parent_persistent_prompt_cache_block_key: Option<&PersistentPromptCacheBlockKey>,
    persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
    block_tokens: &[u32],
    block_causal_input: &PersistentPromptCacheBlockCausalInput,
) -> Result<PersistentPromptCacheBlockKey, PersistentPromptCacheBlockKeyError> {
    match parent_persistent_prompt_cache_block_key {
        None => PersistentPromptCacheBlockKey::for_root_block_with_causal_input(
            persistent_prompt_cache_model_contract,
            block_tokens,
            block_causal_input,
        ),
        Some(parent_persistent_prompt_cache_block_key) => parent_persistent_prompt_cache_block_key
            .for_child_block_with_causal_input(block_tokens, block_causal_input),
    }
}

/// Decides whether block publication must first release allocator workspace.
///
/// Publication materializes the block's state tensors (tens of MB). Wiping the
/// whole allocator cache for that forced the chunk after each published block
/// to re-allocate every intermediate buffer fresh, which measured as a large
/// hidden prefill cost on roomy ceilings. Only pay the wipe when the
/// active-memory ceiling actually needs the room; an unreadable snapshot or
/// limit keeps the old unconditional behavior.
pub(crate) fn publication_workspace_decision(
    active_memory_bytes: Option<u64>,
    active_memory_ceiling_bytes: Option<u64>,
) -> bool {
    match (active_memory_bytes, active_memory_ceiling_bytes) {
        (Some(active_memory_bytes), Some(active_memory_ceiling_bytes)) => {
            active_memory_bytes + PUBLICATION_COMFORT_HEADROOM_BYTES > active_memory_ceiling_bytes
        }
        _ => true,
    }
}

/// Allocator-cache bytes a publication workspace release actually cleared,
/// measured between the snapshots taken before and after the cleanup. An
/// unreadable before-snapshot counts as zero cleared bytes; an unreadable
/// after-snapshot reads as zero bytes remaining, so the full before value
/// counts as cleared.
pub(crate) fn allocator_bytes_cleared(
    allocator_cache_memory_bytes_before_cleanup: Option<u64>,
    allocator_cache_memory_bytes_after_cleanup: Option<u64>,
) -> u64 {
    allocator_cache_memory_bytes_before_cleanup
        .unwrap_or(0)
        .saturating_sub(allocator_cache_memory_bytes_after_cleanup.unwrap_or(0))
}
