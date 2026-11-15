//! End-to-end astro SSD astro SSD prompt cache tests that exercise the full Astro SSD astro SSD prompt cache pipeline.
//!
//! **These tests are intentionally slow**: they load Ornith and exercise the
//! astro SSD astro SSD prompt cache end-to-end, including cold prefill, astro SSD astro SSD prompt cache save, and astro SSD astro SSD prompt cache
//! restore. Each test has an internal 115-second timeout.
//!
//! Do **not** run these as part of the normal development loop. Use the dedicated
//! astro SSD astro SSD prompt cache acceptance journey:
//!
//! ```sh
//! cargo xtask test-run --package astronomical-acceptance --test engine_astro_ssd_prompt_cache --features direct-mlx --release --configuration PATH --filter <full_test_name> --exact --ignored
//! ```

mod astro_ssd_prompt_cache_disabled;
#[cfg(feature = "direct-mlx")]
mod astro_ssd_prompt_cache_interaction_matrix;
#[cfg(feature = "direct-mlx")]
mod astro_ssd_prompt_cache_miss_memory;
#[cfg(feature = "direct-mlx")]
mod astro_ssd_prompt_cache_restore_peak;
mod engine_astro_ssd_prompt_cache;
mod large_prefill_prompt;
mod partial_tail_reuse;
#[cfg(feature = "direct-mlx")]
mod startup_cleanup_attribution;
#[cfg(feature = "direct-mlx")]
mod vision_astro_ssd_prompt_cache;
