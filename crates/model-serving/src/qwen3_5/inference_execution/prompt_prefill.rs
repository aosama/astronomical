use std::time::Instant;

use astronomical_ipc_protocol::RequestId;
use astronomical_runtime_integration::MlxRuntimeError;

use crate::{
    AdaptiveRamGrowthContext, InferenceEngineError, PerformanceCounter, Qwen3_5ExecutionError,
};

use super::engine_request::{Qwen3_5EngineRequest, Qwen3_5PrefillRequestCheckpoint};
use super::memory_admission::AdaptiveRamGrowthMemoryAdmissionError;
use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};

pub(super) enum PromptPrefillChunckAttemptError {
    AdaptiveMemoryLimitExceeded {
        reason: String,
    },
    ActiveMemoryLimitExceeded {
        active_memory_bytes: usize,
        attempted_allocation_bytes: usize,
        allowed_active_memory_bytes: usize,
        prefill_request_checkpoint: Qwen3_5PrefillRequestCheckpoint,
    },
    Engine(InferenceEngineError),
}

impl From<InferenceEngineError> for PromptPrefillChunckAttemptError {
    fn from(inference_engine_error: InferenceEngineError) -> Self {
        Self::Engine(inference_engine_error)
    }
}

impl From<AdaptiveRamGrowthMemoryAdmissionError> for PromptPrefillChunckAttemptError {
    fn from(admission_error: AdaptiveRamGrowthMemoryAdmissionError) -> Self {
        match admission_error {
            AdaptiveRamGrowthMemoryAdmissionError::InsufficientCapacity { reason } => {
                Self::AdaptiveMemoryLimitExceeded { reason }
            }
            AdaptiveRamGrowthMemoryAdmissionError::Engine(inference_engine_error) => {
                Self::Engine(inference_engine_error)
            }
        }
    }
}

impl Qwen3_5EngineState {
    pub(super) fn execute_prompt_prefill_chunck(
        &self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
        prefill_start: usize,
        prefill_end: usize,
    ) -> Result<(usize, u64, AdaptiveRamGrowthContext), PromptPrefillChunckAttemptError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let prefill_token_count = prefill_end - prefill_start;
        active_request
            .performance_attribution
            .record_counter(PerformanceCounter::PrefillChunckCount, 1);
        let prefill_token_ids = &active_request.input_token_ids[prefill_start..prefill_end];
        let mtp_full_attention_growth_bytes = match active_request.mtp_request_state.as_ref() {
            Some(mtp_request_state) if active_request.visual_embeddings.is_none() => {
                let mtp_full_attention_bytes_per_layer_token = model
                    .config()
                    .full_attention_key_value_state_bytes_per_layer_token()
                    .ok_or_else(|| {
                        fatal_engine_error("MTP full-attention bytes per layer token overflowed")
                    })?;
                mtp_request_state
                    .projected_capacity_growth_bytes(
                        mtp_full_attention_bytes_per_layer_token,
                        prefill_token_count,
                    )
                    .map_err(qwen3_5_runtime_error)?
            }
            _ => 0,
        };
        let adaptive_ram_growth_context = AdaptiveRamGrowthContext::prefill(
            prefill_token_count,
            self.prefill_chunck_sizer
                .prompt_processing_context_identifier(prefill_start),
            active_request.visual_embeddings.is_some(),
            active_request.mtp_request_state.is_some(),
            model.sparse_experts_are_paged(),
        );
        let active_memory_bytes_before_growth = self.measure_adaptive_ram_growth_memory_admission(
            adaptive_ram_growth_context,
            &mut active_request.performance_attribution,
            &active_request.request_decoder_state,
            mtp_full_attention_growth_bytes,
        )?;
        let prefill_request_checkpoint = active_request
            .prefill_request_checkpoint()
            .map_err(qwen3_5_runtime_error)?;
        let forward_chunck_started_at = Instant::now();
        if let Some(visual_embeddings) = active_request.visual_embeddings.as_ref() {
            let consumed_visual_embedding_count = match model
                .prefill_chunck_with_visual_embeddings_and_performance_attribution(
                    prefill_token_ids,
                    active_request.next_position_tokens,
                    visual_embeddings,
                    active_request.consumed_visual_embedding_count,
                    &mut active_request.request_decoder_state,
                    active_request.image_pad_token_id,
                    &mut active_request.performance_attribution,
                ) {
                Ok(consumed_visual_embedding_count) => consumed_visual_embedding_count,
                Err(qwen3_5_execution_error) => {
                    return Err(prefill_execution_error(
                        qwen3_5_execution_error,
                        prefill_request_checkpoint,
                    ));
                }
            };
            active_request.consumed_visual_embedding_count += consumed_visual_embedding_count;
        } else if active_request.mtp_request_state.is_some() {
            let target_prefill_output = match model
                .forward_chunk_with_pre_final_normalization_hidden_states_and_performance_attribution(
                    prefill_token_ids,
                    active_request.next_position_tokens,
                    &mut active_request.request_decoder_state,
                    &mut active_request.performance_attribution,
                )
            {
                Ok(target_prefill_output) => target_prefill_output,
                Err(qwen3_5_execution_error) => {
                    return Err(prefill_execution_error(
                        qwen3_5_execution_error,
                        prefill_request_checkpoint,
                    ));
                }
            };
            let shifted_prompt_token_ids =
                &active_request.input_token_ids[prefill_start + 1..prefill_end + 1];
            let mtp_prefill_result = model
                .prefill_mtp_history_from_token_ids_with_performance_attribution(
                    target_prefill_output.pre_final_normalization_hidden_states(),
                    shifted_prompt_token_ids,
                    active_request
                        .mtp_request_state
                        .as_mut()
                        .ok_or_else(|| fatal_engine_error("MTP request state disappeared"))?,
                    &mut active_request.performance_attribution,
                );
            if let Err(mtp_prefill_error) = mtp_prefill_result {
                tracing::warn!(
                    request_id = request_id.value(),
                    error = %mtp_prefill_error,
                    "MTP prompt-history prefill failed; continuing target-only"
                );
                active_request.mtp_request_state = None;
                active_request.mtp_target_hidden_states = None;
            }
        } else {
            if let Err(qwen3_5_execution_error) = model.prefill_chunck_with_performance_attribution(
                prefill_token_ids,
                active_request.next_position_tokens,
                &mut active_request.request_decoder_state,
                &mut active_request.performance_attribution,
            ) {
                return Err(prefill_execution_error(
                    qwen3_5_execution_error,
                    prefill_request_checkpoint,
                ));
            }
        }
        Ok((
            active_memory_bytes_before_growth,
            forward_chunck_started_at.elapsed().as_millis() as u64,
            adaptive_ram_growth_context
                .with_sparse_experts_are_paged(model.sparse_experts_are_paged()),
        ))
    }
}

fn prefill_execution_error(
    qwen3_5_execution_error: Qwen3_5ExecutionError,
    prefill_request_checkpoint: Qwen3_5PrefillRequestCheckpoint,
) -> PromptPrefillChunckAttemptError {
    match qwen3_5_execution_error {
        Qwen3_5ExecutionError::Runtime(MlxRuntimeError::ActiveMemoryLimitExceeded {
            active_memory_bytes,
            attempted_allocation_bytes,
            allowed_active_memory_bytes,
        }) => PromptPrefillChunckAttemptError::ActiveMemoryLimitExceeded {
            active_memory_bytes,
            attempted_allocation_bytes,
            allowed_active_memory_bytes,
            prefill_request_checkpoint,
        },
        other_qwen3_5_execution_error => {
            PromptPrefillChunckAttemptError::Engine(other_qwen3_5_execution_error.into())
        }
    }
}
