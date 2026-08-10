//! End-to-end persistent prompt cache tests that exercise the full SSD cache pipeline.
//!
//! **These tests are intentionally slow**: they load Ornith and exercise the
//! persistent prompt cache end-to-end, including cold prefill, cache save, and cache
//! restore. Each test has an internal 115-second timeout.
//!
//! Do **not** run these as part of the normal development loop. Use the dedicated
//! cache Cargo alias:
//!
//! ```sh
//! cargo qualify-qwen3-5-persistent-prompt-cache
//! ```

#[cfg(feature = "direct-mlx")]
mod cache_interaction_matrix;
mod engine_prompt_cache;
mod large_prefill_prompt;
#[cfg(feature = "direct-mlx")]
mod startup_cleanup_attribution;
#[cfg(feature = "direct-mlx")]
mod vision_prompt_cache;
