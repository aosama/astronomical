//! Chunked prompt processing and persistent-cache capture for K2.

use std::time::Instant;

use astronomical_ipc_protocol::{ExpertMemoryMode, WorkerPromptWorkReuse};

use crate::k2_horizon_mova::model::{K2HorizonMoVAExecutionError, K2HorizonMoVAModel};
use crate::{
    GeneratedToken, InferenceEngineError, PerformanceAttribution, PerformanceOperation,
    PersistentPromptCacheDiskStore,
};

use super::execution::K2HorizonMoVAActiveGeneration;

pub(super) fn prefill_next_chunk(
    model: &K2HorizonMoVAModel,
    active: &mut K2HorizonMoVAActiveGeneration,
    prompt_processing_chunk_tokens: u32,
    persistent_prompt_cache: Option<&PersistentPromptCacheDiskStore>,
) -> Result<GeneratedToken, InferenceEngineError> {
    let chunk_token_count = (prompt_processing_chunk_tokens as usize)
        .min(active.remaining_prompt_token_ids.len())
        .max(1);
    let chunk_start = active.processed_prompt_token_count as usize;
    let chunk_token_ids = active
        .remaining_prompt_token_ids
        .drain(..chunk_token_count)
        .collect::<Vec<_>>();
    let chunk_started_at = Instant::now();
    let mut performance_attribution = std::mem::replace(
        &mut active.performance_attribution,
        PerformanceAttribution::disabled(),
    );
    let hidden_states = performance_attribution
        .measure_operation(
            PerformanceOperation::PromptPrefillAdvanceSpan,
            |performance_attribution| {
                model.forward(
                    &chunk_token_ids,
                    &mut active.caches,
                    performance_attribution,
                    false,
                )
            },
        )
        .map_err(execution_error)?;
    let forward_prefill_chunk_elapsed_millis =
        u64::try_from(chunk_started_at.elapsed().as_millis()).unwrap_or(u64::MAX);
    active.processed_prompt_token_count = active
        .processed_prompt_token_count
        .saturating_add(chunk_token_ids.len() as u32);
    let capture_result = if let Some(persistent_prompt_cache) = persistent_prompt_cache {
        super::prompt_cache::capture_completed_cache_blocks(
            &model.runtime,
            persistent_prompt_cache,
            &active.prompt_token_ids,
            &active.caches,
            chunk_start,
            active.processed_prompt_token_count as usize,
            &mut active.last_published_block_key,
            &mut performance_attribution,
        )
    } else {
        Ok(())
    };
    active.performance_attribution = performance_attribution;
    capture_result?;
    let key_value_tokens = active
        .caches
        .first()
        .map(crate::k2_horizon_mova::model::K2HorizonMoVAKvState::offset_tokens)
        .map(|offset| offset.max(0) as u32)
        .unwrap_or(0);
    tracing::info!(
        chunk_token_count = chunk_token_ids.len() as u32,
        processed_prompt_token_count = active.processed_prompt_token_count,
        remaining_prompt_token_count = active.remaining_prompt_token_ids.len() as u32,
        key_value_tokens,
        forward_prefill_chunk_elapsed_millis,
        total_prefill_elapsed_millis =
            u64::try_from(active.prefill_started_at.elapsed().as_millis()).unwrap_or(u64::MAX),
        chunk_tokens_per_second = if forward_prefill_chunk_elapsed_millis == 0 {
            0.0
        } else {
            chunk_token_ids.len() as f64 * 1_000.0 / forward_prefill_chunk_elapsed_millis as f64
        },
        "K2 Horizon MoVA prefill chunk completed"
    );
    if active.remaining_prompt_token_ids.is_empty() {
        let sample_started_at = Instant::now();
        let logits = model
            .logits_for_last_token(&hidden_states)
            .map_err(execution_error)?;
        let first_token = model
            .sample_token(&logits, &active.request, &mut active.random_state)
            .map_err(execution_error)?;
        tracing::info!(
            first_token_sample_elapsed_millis =
                u64::try_from(sample_started_at.elapsed().as_millis()).unwrap_or(u64::MAX),
            "K2 Horizon MoVA sampled the first token after prefill"
        );
        active.next_input_token_ids = vec![first_token];
    }
    Ok(GeneratedToken::PrefillProgress {
        processed_token_count: chunk_token_ids.len() as u32,
        elapsed_millis: forward_prefill_chunk_elapsed_millis.max(1),
        forward_prefill_chunk_elapsed_millis,
        completed_prefill_chunk_tokens: chunk_token_ids.len() as u32,
        mlx_memory_telemetry: None,
        expert_residency_telemetry: None,
        speculative_prefill_draft_memory_telemetry: None,
        expert_memory_mode: Some(ExpertMemoryMode::Resident),
        prompt_work_reuse: WorkerPromptWorkReuse {
            target_eligible_token_count: active.prompt_token_ids.len() as u64,
            target_restored_token_count: u64::from(active.cached_token_count),
            drafter_eligible_token_count: 0,
            drafter_restored_token_count: 0,
        },
        persistent_prompt_cache_diagnostics: None,
    })
}

pub(super) fn execution_error(error: K2HorizonMoVAExecutionError) -> InferenceEngineError {
    InferenceEngineError::Fatal {
        reason: error.to_string(),
    }
}
