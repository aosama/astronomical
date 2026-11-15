//! Structured bounded logging for astro-ssd-prompt-cache cleanup completed during model load.

use crate::AstroSsdPromptCacheDiskStore;

pub(super) fn log_astro_ssd_prompt_cache_startup_cleanup(
    astro_ssd_prompt_cache_role: &'static str,
    astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
) {
    let Some(startup_cleanup_evidence) = astro_ssd_prompt_cache.startup_cleanup_evidence() else {
        return;
    };
    let interrupted_transaction = startup_cleanup_evidence.interrupted_transaction_recovery;
    let obsolete_format = startup_cleanup_evidence.obsolete_format;
    let corrupt_current_format = startup_cleanup_evidence.corrupt_current_format;
    let quota_eviction = startup_cleanup_evidence.quota_eviction;
    tracing::info!(
        astro_ssd_prompt_cache_role,
        interrupted_transaction_artifact_count = interrupted_transaction.artifact_count,
        interrupted_transaction_block_count = interrupted_transaction.block_count,
        interrupted_transaction_byte_count = interrupted_transaction.byte_count,
        obsolete_format_artifact_count = obsolete_format.artifact_count,
        obsolete_format_block_count = obsolete_format.block_count,
        obsolete_format_byte_count = obsolete_format.byte_count,
        corrupt_current_format_artifact_count = corrupt_current_format.artifact_count,
        corrupt_current_format_block_count = corrupt_current_format.block_count,
        corrupt_current_format_byte_count = corrupt_current_format.byte_count,
        quota_eviction_artifact_count = quota_eviction.artifact_count,
        quota_eviction_block_count = quota_eviction.block_count,
        quota_eviction_byte_count = quota_eviction.byte_count,
        "persistent astro-ssd-prompt-cache startup cleanup completed"
    );
}
