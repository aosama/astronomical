//! Prompt-cache acceptance: REST restore, append-only follow-up, and astro SSD astro SSD prompt cache stats.

#[cfg(feature = "serving-acceptance")]
mod append_only_rest;
#[cfg(feature = "serving-acceptance")]
mod astro_ssd_prompt_cache_stats_e2e;
#[cfg(feature = "serving-acceptance")]
mod astro_ssd_prompt_cache_stats_worker_launcher;
#[cfg(feature = "serving-acceptance")]
mod dense_cached_token_report;
#[cfg(feature = "serving-acceptance")]
pub(crate) mod rest_support;
#[cfg(feature = "serving-acceptance")]
mod restore_rest;
