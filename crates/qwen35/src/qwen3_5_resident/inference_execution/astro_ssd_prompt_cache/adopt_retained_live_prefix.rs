//! Installs the retained live decoder state for an adopting request.
//!
//! In-memory prefix adoption (issue #1167): when the retained chain matches
//! at least the disk-matched prefix, the live state replaces the fresh empty
//! stack — zero disk block reads, zero recopies. The append-only layers are
//! truncated to the adopted boundary (their stale tail is overwritten by the
//! next prefill's in-place slab writes), and the composite (recurrent) layers
//! absorb the boundary snapshot from the disk store, because the retained
//! recurrent state sits at the generation-end position, not at the boundary.

use ipc_protocol::{WorkerAstroSsdPromptCacheLookupOutcome, WorkerInMemoryPrefixAdoptionOutcome};

use crate::astro_decoder_state_cache::AstroDecoderStateCacheState;
use crate::astro_ssd_prompt_cache::AstroSsdPromptCacheDiskStore;
use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::{
    AstroSsdPromptCachePrefixLookupResult, InferenceEngineError, PerformanceAttribution,
    PerformanceOperation,
};

use crate::astro_ssd_prompt_cache::restore_support::astro_ssd_prompt_cache_request_diagnostics;

use super::decoder_state_reuse::AstroSsdPromptCacheRestoreOutcome;
use super::retained_live_prefix::RetainedLiveDecoderPrefix;
use super::{Qwen3_5EngineState, fatal_engine_error};

impl Qwen3_5EngineState {
    pub(super) fn install_adopted_live_prefix(
        &mut self,
        astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
        retained_live_prefix: RetainedLiveDecoderPrefix,
        retained_lookup: AstroSsdPromptCachePrefixLookupResult,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<AstroSsdPromptCacheRestoreOutcome, InferenceEngineError> {
        let adopted_token_count = retained_lookup.restored_token_count();
        let adopted_boundary_block_key = retained_lookup
            .last_restored_astro_ssd_prompt_cache_block_key()
            .cloned()
            .ok_or_else(|| fatal_engine_error("the adopted prefix must end at a complete block"))?;
        let adopted_token_count_i32 = i32::try_from(adopted_token_count).map_err(|_| {
            fatal_engine_error("the adopted prefix token count exceeds the i32 range")
        })?;

        *request_decoder_state = retained_live_prefix.decoder_state;
        for layer_index in 0..request_decoder_state.layer_count() {
            if let Some(AstroDecoderStateCacheState::AppendOnlyAttention { attention }) =
                request_decoder_state.layer_mut(layer_index)
            {
                attention
                    .truncate_to_offset(adopted_token_count_i32)
                    .map_err(|truncate_error| {
                        fatal_engine_error(format!(
                            "the adopted live prefix failed to truncate layer {layer_index}: \
                             {truncate_error}"
                        ))
                    })?;
            }
        }

        let model = self
            .core
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let mut astro_ssd_prompt_cache_recurrent_snapshot_tensors = performance_attribution
            .measure_operation(
                PerformanceOperation::AstroSsdPromptCacheRecurrentSnapshotRead,
                |performance_attribution| {
                    astro_ssd_prompt_cache.load_recurrent_snapshot(
                        model.runtime(),
                        &adopted_boundary_block_key,
                        performance_attribution.positional_file_read_metrics(),
                    )
                },
            )
            .map_err(|astro_ssd_prompt_cache_error| {
                fatal_engine_error(format!(
                    "the adopted prefix's boundary snapshot failed to load: \
                     {astro_ssd_prompt_cache_error}"
                ))
            })?
            .ok_or_else(|| {
                fatal_engine_error(
                    "the adopted prefix's boundary snapshot was reported present but missing",
                )
            })?;
        performance_attribution
            .measure_operation(
                PerformanceOperation::AstroSsdPromptCacheStateReconstruction,
                |_performance_attribution| {
                    request_decoder_state.absorb_astro_ssd_prompt_cache_recurrent_snapshot(
                        model.runtime(),
                        &mut astro_ssd_prompt_cache_recurrent_snapshot_tensors,
                    )
                },
            )
            .map_err(|astro_ssd_prompt_cache_error| {
                fatal_engine_error(format!(
                    "the adopted prefix's recurrent state failed to absorb its boundary snapshot: \
                     {astro_ssd_prompt_cache_error}"
                ))
            })?;

        let mut astro_ssd_prompt_cache_diagnostics = astro_ssd_prompt_cache_request_diagnostics(
            astro_ssd_prompt_cache.model_contract.block_token_count(),
            retained_lookup.diagnostics(),
            adopted_token_count,
            None,
        );
        astro_ssd_prompt_cache_diagnostics.lookup_outcome =
            WorkerAstroSsdPromptCacheLookupOutcome::Hit;
        astro_ssd_prompt_cache_diagnostics.in_memory_prefix_adoption =
            Some(WorkerInMemoryPrefixAdoptionOutcome::Adopted);
        self.core
            .astro_ssd_prompt_cache_counters
            .record_in_memory_prefix_adoption_hit();
        Ok(AstroSsdPromptCacheRestoreOutcome {
            astro_ssd_prompt_cache_restored_token_count: u32::try_from(adopted_token_count)
                .unwrap_or(u32::MAX),
            restored_token_count: adopted_token_count,
            last_restored_astro_ssd_prompt_cache_block_key: Some(adopted_boundary_block_key),
            astro_ssd_prompt_cache_diagnostics,
        })
    }
}
