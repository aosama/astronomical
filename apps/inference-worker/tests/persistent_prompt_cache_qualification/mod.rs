//! End-to-end persistent prompt cache tests that exercise the full SSD cache pipeline
//! through the supervisor and HTTP endpoint.
//!
//! **These tests are dead-slow**: they launch the complete supervisor + worker stack,
//! load the complete Qwen3.6 model, and exercise persistent prompt cache save and restore with
//! a 2K-word prompt. The test has an internal 115-second timeout.
//!
//! Do **not** run these as part of the normal development loop. Use the dedicated
//! cache Cargo alias:
//!
//! ```sh
//! cargo qualify-persistent-prompt-cache
//! ```

#[cfg(feature = "model-artifact-qualification")]
mod cache_stats_worker_launcher;
#[cfg(feature = "model-artifact-qualification")]
mod persistent_prompt_cache_stats_e2e;
