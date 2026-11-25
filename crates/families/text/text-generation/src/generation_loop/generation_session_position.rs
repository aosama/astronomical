//! Where a generation session starts, derived from one `astro_ssd_prompt_cache`
//! restore outcome; every text engine start their sessions from it
//! (issue #2473).

use crate::AstroSsdPromptCacheBlockKey;
use ipc_protocol::WorkerAstroSsdPromptCacheRequestDiagnostics;

use crate::generation_loop::astro_ssd_prompt_cache_restore::AstroSsdPromptCacheRestoreOutcome;

/// Where the admitted session begins: how much prompt work already exists
/// and which token position the first new token occupies.
pub struct GenerationSessionPosition {
    pub astro_ssd_prompt_cache_restored_token_count: u32,
    pub prefill_cursor: usize,
    pub next_position_tokens: u32,
    pub restored_target_work_token_count: u64,
    pub last_restored_astro_ssd_prompt_cache_block_key: Option<AstroSsdPromptCacheBlockKey>,
    pub astro_ssd_prompt_cache_diagnostics: Option<WorkerAstroSsdPromptCacheRequestDiagnostics>,
}

/// Maps one restore outcome onto the session's starting position.
///
/// `None` (store absent, reuse disabled, or nothing restored) yields the
/// all-zero origin, matching the previous default initialization exactly.
/// In the `Some` arm the cursor and the next position come from different
/// outcome fields on purpose: the cursor advances by restored tokens while
/// positions continue from the `astro_ssd_prompt_cache` token count, and the
/// two diverge whenever a partial tail block is recovered.
pub fn derive_generation_session_position(
    restore_outcome: Option<AstroSsdPromptCacheRestoreOutcome>,
) -> GenerationSessionPosition {
    let Some(restore_outcome) = restore_outcome else {
        return GenerationSessionPosition {
            astro_ssd_prompt_cache_restored_token_count: 0,
            prefill_cursor: 0,
            next_position_tokens: 0,
            restored_target_work_token_count: 0,
            last_restored_astro_ssd_prompt_cache_block_key: None,
            astro_ssd_prompt_cache_diagnostics: None,
        };
    };
    GenerationSessionPosition {
        astro_ssd_prompt_cache_restored_token_count: restore_outcome
            .astro_ssd_prompt_cache_restored_token_count,
        prefill_cursor: restore_outcome.restored_token_count,
        next_position_tokens: restore_outcome.astro_ssd_prompt_cache_restored_token_count,
        restored_target_work_token_count: u64::from(
            restore_outcome.astro_ssd_prompt_cache_restored_token_count,
        ),
        last_restored_astro_ssd_prompt_cache_block_key: restore_outcome
            .last_restored_astro_ssd_prompt_cache_block_key,
        astro_ssd_prompt_cache_diagnostics: Some(
            restore_outcome.astro_ssd_prompt_cache_diagnostics,
        ),
    }
}

#[cfg(test)]
#[path = "generation_session_position_tests.rs"]
mod generation_session_position_tests;
