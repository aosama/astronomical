//! End-to-end `astro_ssd_prompt_cache` tests that exercise the full `astro_ssd_prompt_cache` pipeline.
//!
//! **These tests are intentionally slow**: they load Ornith and exercise the
//! `astro_ssd_prompt_cache` end-to-end, including cold prefill, `astro_ssd_prompt_cache` save, and `astro_ssd_prompt_cache`
//! restore. Each test has an internal 115-second timeout.
//!
//! Do **not** run these as part of the normal development loop. Use the dedicated
//! `astro_ssd_prompt_cache` acceptance journey:
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

/// A package may carry a drafter. Drafted rows are verified by a batched
/// kernel whose logits differ from the one-row step in the last bits, so a
/// drafted reply depends on which drafts were kept. These journeys compare a
/// cold reply with a restored one token for token, so the drafter stays
/// hosted and silent and only the `astro_ssd_prompt_cache` is judged.
#[cfg(feature = "direct-mlx")]
async fn keep_the_hosted_drafter_silent(qwen3_5_engine: &model_serving::Qwen3_5TextEngine) {
    use model_serving::Qwen3_5TextEngineTestControls;
    qwen3_5_engine
        .set_hosted_drafter_writes_replies_for_tests(false)
        .await
        .expect("the engine takes the drafter's part");
}

/// The MLX runtime the loaded engine already configured. The engine sets its
/// own MLX buffer cache limit at load, so the limits a journey asked for may
/// not be the ones in force; MLX names the ones that are when it refuses.
#[cfg(feature = "direct-mlx")]
fn runtime_the_engine_configured(
    requested_memory_limits: runtime_integration::MlxMemoryLimits,
) -> runtime_integration::MlxRuntime {
    use runtime_integration::{MlxMemoryLimits, MlxRuntime, MlxRuntimeError};
    match MlxRuntime::initialize(requested_memory_limits) {
        Ok(mlx_runtime) => mlx_runtime,
        Err(MlxRuntimeError::RuntimeAlreadyConfigured {
            active_memory_limit_bytes,
            mlx_buffer_cache_limit_bytes,
        }) => MlxRuntime::initialize(
            MlxMemoryLimits::new(active_memory_limit_bytes, mlx_buffer_cache_limit_bytes)
                .expect("the limits in force are valid"),
        )
        .expect("the journey should share the engine's MLX runtime"),
        Err(runtime_error) => panic!("the MLX runtime should be reachable: {runtime_error}"),
    }
}
