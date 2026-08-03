//! End-to-end persistent prompt cache tests that exercise the full SSD cache pipeline.
//!
//! **These tests are dead-slow**: they load the 22 GB Ornith model and exercise the
//! persistent prompt cache end-to-end, including cold prefill, cache save, and cache
//! restore. Each test has an internal 115-second timeout.
//!
//! Do **not** run these as part of the normal development loop. Use the dedicated
//! cache Cargo alias:
//!
//! ```sh
//! cargo qualify-persistent-prompt-cache
//! ```

#[cfg(feature = "direct-mlx")]
mod engine_prompt_cache;
