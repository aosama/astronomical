//! Session position derivation from one persistent prompt-cache restore outcome.

use crate::PersistentPromptCacheBlockKey;
use astronomical_ipc_protocol::WorkerPersistentPromptCacheRequestDiagnostics;

use super::decoder_state_reuse::PersistentPromptCacheRestoreOutcome;

/// Where the admitted session begins: how much prompt work already exists
/// and which token position the first new token occupies.
pub(super) struct GenerationSessionPosition {
    pub(super) persistent_prompt_cache_token_count: u32,
    pub(super) prefill_cursor: usize,
    pub(super) next_position_tokens: u32,
    pub(super) restored_target_work_token_count: u64,
    pub(super) last_restored_persistent_prompt_cache_block_key:
        Option<PersistentPromptCacheBlockKey>,
    pub(super) persistent_prompt_cache_diagnostics:
        Option<WorkerPersistentPromptCacheRequestDiagnostics>,
}

/// Maps one restore outcome onto the session's starting position.
///
/// `None` (store absent, reuse disabled, or nothing restored) yields the
/// all-zero origin, matching the previous default initialization exactly.
/// In the `Some` arm the cursor and the next position come from different
/// outcome fields on purpose: the cursor advances by restored tokens while
/// positions continue from the persistent prompt-cache token count, and the
/// two diverge whenever a partial tail block is recovered.
pub(super) fn derive_generation_session_position(
    restore_outcome: Option<PersistentPromptCacheRestoreOutcome>,
) -> GenerationSessionPosition {
    let Some(restore_outcome) = restore_outcome else {
        return GenerationSessionPosition {
            persistent_prompt_cache_token_count: 0,
            prefill_cursor: 0,
            next_position_tokens: 0,
            restored_target_work_token_count: 0,
            last_restored_persistent_prompt_cache_block_key: None,
            persistent_prompt_cache_diagnostics: None,
        };
    };
    GenerationSessionPosition {
        persistent_prompt_cache_token_count: restore_outcome.persistent_prompt_cache_token_count,
        prefill_cursor: restore_outcome.restored_token_count,
        next_position_tokens: restore_outcome.persistent_prompt_cache_token_count,
        restored_target_work_token_count: u64::from(
            restore_outcome.persistent_prompt_cache_token_count,
        ),
        last_restored_persistent_prompt_cache_block_key: restore_outcome
            .last_restored_persistent_prompt_cache_block_key,
        persistent_prompt_cache_diagnostics: Some(
            restore_outcome.persistent_prompt_cache_diagnostics,
        ),
    }
}

#[cfg(test)]
#[path = "generation_restore_position_derivation_tests.rs"]
mod generation_restore_position_derivation_tests;
