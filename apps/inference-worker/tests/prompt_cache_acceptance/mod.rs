//! Prompt-cache acceptance: REST restore, append-only follow-up, and cache stats.

#[cfg(feature = "serving-acceptance")]
mod append_only_rest;
#[cfg(feature = "serving-acceptance")]
mod cache_stats_worker_launcher;
#[cfg(feature = "serving-acceptance")]
mod persistent_prompt_cache_stats_e2e;
#[cfg(feature = "serving-acceptance")]
pub(crate) mod rest_support;
#[cfg(feature = "serving-acceptance")]
mod restore_rest;
