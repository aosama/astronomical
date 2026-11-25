//! The decoder-state part of an `astro_ssd_prompt_cache` restore: it
//! installs the restored attention and recurrent state into the request's
//! decoder state. Everything before that, from the block lookup
//! to the loaded tensors, is family-neutral and lives in
//! `engine_core::astro_ssd_prompt_cache::restore_support`.

use std::collections::HashMap;

use mlx_bridge::MlxArray;
use runtime_integration::MlxRuntime;

use crate::decoder_state::RequestDecoderStateStack;
use crate::{
    AstroSsdPromptCacheBlockKey, AstroSsdPromptCacheDiskStore, InferenceEngineError,
    PerformanceAttribution, PerformanceOperation,
};

/// Reconstructs the full-attention KV state of the `astro_decoder_state_cache` from the loaded blocks with one
/// concatenation per layer, then releases the source blocks: the snapshot
/// phase must never hold the block set and the destination at once.
pub fn restore_concatenated_full_attention_state(
    runtime: &MlxRuntime,
    request_decoder_state: &mut RequestDecoderStateStack,
    restored_astro_ssd_prompt_cache_kv_block_tensors: Vec<HashMap<String, MlxArray>>,
    restored_token_count: usize,
    complete_block_count: usize,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(), InferenceEngineError> {
    let reconstruction_result = performance_attribution.measure_operation(
        PerformanceOperation::AstroSsdPromptCacheStateReconstruction,
        |_performance_attribution| {
            request_decoder_state.restore_full_attention_kv_concat(
                runtime,
                &restored_astro_ssd_prompt_cache_kv_block_tensors,
                restored_token_count,
            )
        },
    );
    drop(restored_astro_ssd_prompt_cache_kv_block_tensors);
    reconstruction_result.map_err(|astro_ssd_prompt_cache_error| InferenceEngineError::Fatal {
        reason: format!(
            "failed to restore request decoder state from {complete_block_count} \
                 persistent astro-ssd-prompt-cache blocks: {astro_ssd_prompt_cache_error}"
        ),
    })
}

/// Loads the newest recurrent snapshot and absorbs it into live state.
///
/// Sequence state is append-only across every matched block, but recurrent
/// state needs only the newest boundary corresponding to the restored end.
pub fn absorb_newest_recurrent_snapshot(
    runtime: &MlxRuntime,
    astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
    request_decoder_state: &mut RequestDecoderStateStack,
    recurrent_snapshot_block_key: &AstroSsdPromptCacheBlockKey,
    complete_block_count: usize,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(), InferenceEngineError> {
    let mut astro_ssd_prompt_cache_recurrent_snapshot_tensors = performance_attribution
        .measure_operation(
            PerformanceOperation::AstroSsdPromptCacheRecurrentSnapshotRead,
            |performance_attribution| {
                astro_ssd_prompt_cache.load_recurrent_snapshot(
                    runtime,
                    recurrent_snapshot_block_key,
                    performance_attribution.positional_file_read_metrics(),
                )
            },
        )
        .map_err(|astro_ssd_prompt_cache_error| InferenceEngineError::Fatal {
            reason: format!(
                "failed to load persistent astro-ssd-prompt-cache recurrent snapshot: \
                     {astro_ssd_prompt_cache_error}"
            ),
        })?
        .ok_or_else(|| InferenceEngineError::Fatal {
            reason: "persistent astro-ssd-prompt-cache recurrent snapshot was reported as present \
                     but load returned None"
                .to_owned(),
        })?;
    let absorb_result = performance_attribution.measure_operation(
        PerformanceOperation::AstroSsdPromptCacheStateReconstruction,
        |_performance_attribution| {
            request_decoder_state.absorb_astro_ssd_prompt_cache_recurrent_snapshot(
                runtime,
                &mut astro_ssd_prompt_cache_recurrent_snapshot_tensors,
            )
        },
    );
    drop(astro_ssd_prompt_cache_recurrent_snapshot_tensors);
    absorb_result.map_err(|astro_ssd_prompt_cache_error| InferenceEngineError::Fatal {
        reason: format!(
            "failed to restore request decoder state from {complete_block_count} \
                 persistent astro-ssd-prompt-cache blocks: {astro_ssd_prompt_cache_error}"
        ),
    })
}

/// Materializes the restored state on the GPU and releases every transient
/// buffer before the caller re-runs memory admission.
pub fn materialize_and_release_restored_state(
    runtime: &MlxRuntime,
    request_decoder_state: &mut RequestDecoderStateStack,
    complete_block_count: usize,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(), InferenceEngineError> {
    performance_attribution
        .measure_operation(
            PerformanceOperation::AstroSsdPromptCacheStateMaterializationSynchronizationWait,
            |_performance_attribution| {
                request_decoder_state.materialize_restored_astro_ssd_prompt_cache_state(runtime)
            },
        )
        .map_err(|astro_ssd_prompt_cache_error| InferenceEngineError::Fatal {
            reason: format!(
                "failed to materialize restored request decoder state from \
                     {complete_block_count} persistent astro-ssd-prompt-cache blocks: \
                     {astro_ssd_prompt_cache_error}"
            ),
        })?;
    performance_attribution
        .measure_operation(
            PerformanceOperation::MlxBufferCacheCleanup,
            |_performance_attribution| runtime.clear_mlx_buffer_cache(),
        )
        .map_err(|runtime_error| InferenceEngineError::Fatal {
            reason: format!(
                "failed to clear allocator memory after persistent astro-ssd-prompt-cache restore: \
                     {runtime_error}"
            ),
        })
}
