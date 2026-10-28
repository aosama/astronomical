//! The `MlxInferenceExecution` engine implementation for the owner-thread
//! folded-ternary execution: load, chunked prefill, decode, correction
//! injection, and live memory-limit adjustment. Split from the execution
//! state vocabulary in the parent so each file stays inside the source-size
//! budget; as a child module it reaches the parent's private fields directly.

use std::time::Instant;

use astronomical_ipc_protocol::{ExpertMemoryMode, RequestId, WorkerPromptWorkReuse};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use mlx_c_rust::MlxArray;

use super::{
    PrismHadamardQwen35ActiveGeneration, PrismHadamardQwen35InferenceExecution,
    record_emitted_token,
    sampling::{
        record_penalized_token_id, sample_token, successor_may_be_speculated, synchronize_token_id,
    },
    successor_stays_pending,
};
use crate::EngineGenerationStart;
use crate::GeneratedToken;
use crate::GenerationFinalization;
use crate::InferenceEngineError;
use crate::MlxInferenceExecution;
use crate::MlxMemoryLimitAdjustment;
use crate::MlxMemoryTelemetry;
use crate::PerformanceAttribution;
use crate::PerformanceOperation;
use crate::Qwen3_5ThinkingBudgetState;
use crate::inference_engine::load_result::EngineLoadResult;
use crate::memory::telemetry::MlxActiveMemoryBreakdown;
use crate::persistent_prompt_cache_boundary_clamped_prefill_chunk_end;
use crate::prism_hadamard_qwen35::model::PrismHadamardTextModel;
use crate::prism_hadamard_qwen35::model::PrismStateSlotLayout;
use crate::prism_hadamard_qwen35::model::StraightGenerationState;
use crate::sampling_seed;
use crate::sampling_seed::current_time_millis_since_unix_epoch;
use astronomical_ipc_protocol::WorkerEvent;

impl MlxInferenceExecution for PrismHadamardQwen35InferenceExecution {
    type Request = crate::Qwen3_5InferenceRequest;

    fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        let pending = self
            .pending_startup
            .take()
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "folded-ternary startup was already consumed".to_owned(),
            })?;
        let mut performance_attribution = pending.performance_attribution;
        let memory_limits = MlxMemoryLimits::new(
            pending.effective_mlx_memory_ceiling_bytes,
            pending.allocator_cache_memory_limit_bytes,
        )
        .map_err(|error| InferenceEngineError::Fatal {
            reason: format!("folded-ternary MLX limits failed: {error}"),
        })?;
        let runtime = performance_attribution
            .measure_operation(PerformanceOperation::MlxRuntimeInitialization, |_| {
                MlxRuntime::initialize(memory_limits)
            })
            .map_err(|error| InferenceEngineError::Fatal {
                reason: format!("folded-ternary MLX runtime failed: {error}"),
            })?;
        let model = performance_attribution
            .measure_operation(PerformanceOperation::ModelSafetensorsMapping, |_| {
                PrismHadamardTextModel::new(runtime, pending.validated_pack)
            })
            .map_err(gpu_error)?;
        // The dense resident family always owns its whole weight payload; the
        // safe idle minimum is the weights plus decode working set headroom.
        let weights_payload_bytes = model.pack().total_payload_bytes();
        let minimum_ceiling_bytes =
            crate::memory::dense_resident_safe_minimum_mlx_memory_ceiling_bytes(
                weights_payload_bytes,
            );
        self.minimum_ceiling_bytes = minimum_ceiling_bytes;
        self.performance_attribution_log = pending.performance_attribution_log;
        self.attribution_model_id = Some(pending.attribution_model_id.clone());
        self.attribution_model_revision = Some(pending.attribution_model_revision.clone());
        self.model = Some(model);
        if let Some(prompt_cache_disk_store_config) = pending.prompt_cache_disk_store_config {
            let loaded_model = self
                .model
                .as_ref()
                .ok_or_else(|| InferenceEngineError::Fatal {
                    reason: "folded-ternary model is not loaded".to_owned(),
                })?;
            self.persistent_prompt_cache =
                Some(super::super::prompt_cache::open_prompt_cache_store(
                    pending.effective_mlx_memory_ceiling_bytes,
                    loaded_model,
                    &pending.attribution_model_id,
                    &pending.attribution_model_revision,
                    pending.configured_prompt_cache_block_token_count,
                    pending.prompt_cache_common_prefix_stride_blocks,
                    prompt_cache_disk_store_config,
                    &mut performance_attribution,
                )?);
        }
        self.record_model_loading_performance_attribution(performance_attribution);
        Ok(EngineLoadResult::new()
            .with_expert_memory_mode(Some(ExpertMemoryMode::Resident))
            .with_minimum_mlx_memory_ceiling_bytes(minimum_ceiling_bytes))
    }

    fn start_generation(
        &mut self,
        mut inference_request: Self::Request,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        if self.active.is_some() {
            return Err(InferenceEngineError::EngineBusy);
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "folded-ternary model is not loaded".to_owned(),
            })?;
        if inference_request.input_token_ids().is_empty() {
            return Err(InferenceEngineError::InvalidRequest {
                reason: "folded-ternary prompt must contain tokens".to_owned(),
            });
        }
        let sampling_seed = sampling_seed::resolve_sampling_seed(
            match inference_request.sampling_strategy() {
                crate::Qwen3_5SamplingStrategy::HighestLogit => None,
                crate::Qwen3_5SamplingStrategy::TopKTopP { seed, .. } => seed,
            },
            current_time_millis_since_unix_epoch,
        );
        let random_state = model.runtime.random_key(sampling_seed).map_err(|error| {
            InferenceEngineError::Fatal {
                reason: format!("folded-ternary random key failed: {error}"),
            }
        })?;
        let thinking_budget = Qwen3_5ThinkingBudgetState::new(
            inference_request.generation_starts_inside_thinking_block(),
            inference_request.thinking_budget(),
            inference_request
                .forced_thinking_transition_token_ids()
                .to_vec(),
            inference_request.natural_reasoning_end_token_ids().to_vec(),
        )
        .map_err(|budget_error| InferenceEngineError::Fatal {
            reason: format!("folded-ternary thinking budget is invalid: {budget_error}"),
        })?;
        let penalized_token_ids = inference_request.initial_penalized_token_ids();
        let request_id = inference_request.request_id();
        let structured_generation = inference_request.take_structured_generation();
        let mut performance_attribution = inference_request.take_performance_attribution();
        let prompt_token_ids = inference_request.input_token_ids().to_vec();
        let mut cached_token_count: u32 = 0;
        let mut last_published_block_key = None;
        let mut generation_state = StraightGenerationState::empty();
        if let Some(persistent_prompt_cache) = self.persistent_prompt_cache.clone() {
            let restore_outcome = super::super::prompt_cache::lookup_and_restore_prompt_prefix(
                model,
                &persistent_prompt_cache,
                &prompt_token_ids,
                &mut generation_state,
                &mut self.persistent_prompt_cache_counters,
                &mut performance_attribution,
            )?;
            cached_token_count = restore_outcome.restored_token_count;
            last_published_block_key =
                restore_outcome.last_restored_persistent_prompt_cache_block_key;
        }
        self.active = Some(PrismHadamardQwen35ActiveGeneration {
            request_id,
            remaining_output_tokens: u32::from(inference_request.max_output_tokens()),
            prompt_token_ids: prompt_token_ids.clone(),
            remaining_prompt_token_ids: prompt_token_ids[cached_token_count as usize..].to_vec(),
            processed_prompt_token_count: cached_token_count,
            cached_token_count,
            generation_state,
            pending_sampled_token: None,
            penalized_token_ids,
            random_state,
            request: inference_request,
            thinking_budget,
            structured_generation,
            last_published_block_key,
            performance_attribution,
        });
        Ok(EngineGenerationStart::with_expert_memory_mode(
            cached_token_count,
            ExpertMemoryMode::Resident,
        ))
    }

    #[expect(clippy::too_many_lines)]
    fn decode_next_token(
        &mut self,
        request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        let prompt_processing_chunk_tokens = self.prompt_processing_chunk_tokens;
        let has_remaining_prompt = self.active.as_ref().is_some_and(|active_generation| {
            !active_generation.remaining_prompt_token_ids.is_empty()
        });
        if has_remaining_prompt {
            return self.prefill_next_chunk(request_id, prompt_processing_chunk_tokens);
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "folded-ternary model is not loaded".to_owned(),
            })?;
        // Sampled before the active-generation borrow; the one-step-old read
        // still moves with real decode work and keeps the status surface live.
        let mlx_memory_telemetry = self.collect_current_mlx_memory_telemetry();
        let active = self
            .active
            .as_mut()
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "folded-ternary has no active generation".to_owned(),
            })?;
        if active.request_id != request_id {
            return Err(InferenceEngineError::Fatal {
                reason: "folded-ternary generation request correlation mismatch".to_owned(),
            });
        }
        let forced_token_id = active
            .thinking_budget
            .next_forced_transition_token_id()
            .map_err(|budget_error| InferenceEngineError::Fatal {
                reason: format!("folded-ternary thinking budget failed: {budget_error}"),
            })?;
        let eos_token_ids = &model.pack().config().text().eos_token_ids;
        let (selected_token_id, speculated_successor) = match forced_token_id {
            Some(forced_token_id) => {
                // A forced transition replaces whatever was sampled; the
                // decoder state is still exactly the state after the last
                // emitted token, so the forced token's forward runs below.
                active.pending_sampled_token = None;
                (forced_token_id, None)
            }
            None => {
                let Some(sampled_token) = active.pending_sampled_token.take() else {
                    return Ok(GeneratedToken::EndOfSequence);
                };
                // Build and submit the successor before reading this token's
                // value whenever nothing on the host can change the
                // successor's sampling: the device executes while the host
                // synchronizes, and the only waste is a successor behind a
                // token that turns out to end the sequence.
                let speculated_successor =
                    if successor_may_be_speculated(active) && active.remaining_output_tokens > 1 {
                        Some(submit_successor(model, active, &sampled_token)?)
                    } else {
                        None
                    };
                let selected_token_id = active
                    .measure_operation(PerformanceOperation::DecodeSamplingSpan, |_| {
                        synchronize_token_id(&sampled_token)
                    })?;
                if eos_token_ids.contains(&selected_token_id) {
                    return Ok(GeneratedToken::EndOfSequence);
                }
                (selected_token_id, speculated_successor)
            }
        };
        let was_already_visible = !active.thinking_budget.is_inside_thinking();
        let is_reasoning_token = active
            .thinking_budget
            .observe_committed_token(selected_token_id)
            .map_err(|budget_error| InferenceEngineError::Fatal {
                reason: format!("folded-ternary thinking budget failed: {budget_error}"),
            })?;
        if was_already_visible
            && !is_reasoning_token
            && let Some(structured_generation) = active.structured_generation.as_mut()
        {
            structured_generation.accept_visible_token(selected_token_id);
        }
        let output_budget_exhausted = record_emitted_token(&mut active.remaining_output_tokens);
        if output_budget_exhausted {
            active.pending_sampled_token = None;
            return Ok(GeneratedToken::TokenId {
                token_id: selected_token_id,
                is_reasoning_token,
                expert_memory_mode: Some(ExpertMemoryMode::Resident),
                mlx_memory_telemetry,
                first_decode_forward_elapsed_millis: None,
                generation_finalization: None,
            });
        }
        record_penalized_token_id(active, selected_token_id);
        let successor = match speculated_successor {
            Some(successor) => successor,
            None => {
                let token_indices = model
                    .runtime
                    .array_from_i32(&[selected_token_id as i32], &[1, 1])
                    .map_err(gpu_error)?;
                submit_successor(model, active, &token_indices)?
            }
        };
        active.pending_sampled_token =
            successor_stays_pending(active.remaining_output_tokens).then_some(successor);
        Ok(GeneratedToken::TokenId {
            token_id: selected_token_id,
            is_reasoning_token,
            expert_memory_mode: Some(ExpertMemoryMode::Resident),
            mlx_memory_telemetry,
            first_decode_forward_elapsed_millis: None,
            generation_finalization: None,
        })
    }

    fn collect_mlx_memory_telemetry(
        &self,
    ) -> Result<Option<MlxMemoryTelemetry>, InferenceEngineError> {
        Ok(self.collect_current_mlx_memory_telemetry())
    }

    fn inject_input_tokens(
        &mut self,
        request_id: RequestId,
        input_token_ids: Vec<u32>,
    ) -> Result<(), InferenceEngineError> {
        super::input_injection::inject_input_tokens(self, request_id, input_token_ids)
    }

    fn cancel_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError> {
        if self
            .active
            .as_ref()
            .is_some_and(|active_generation| active_generation.request_id != request_id)
        {
            return Err(InferenceEngineError::Fatal {
                reason: "folded-ternary cancellation request correlation mismatch".to_owned(),
            });
        }
        self.take_active_and_record_generation_attribution();
        let mlx_memory_telemetry = self.collect_current_mlx_memory_telemetry();
        Ok(GenerationFinalization::new(
            Some(ExpertMemoryMode::Resident),
            mlx_memory_telemetry,
            None,
        ))
    }

    fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, InferenceEngineError> {
        self.ceiling_bytes = requested_mlx_memory_ceiling_bytes.max(1);
        self.allocator_cache_bytes = self.allocator_cache_bytes.min(self.ceiling_bytes);
        if let Some(model) = self.model.as_mut() {
            let memory_limits = MlxMemoryLimits::new(
                usize::try_from(self.ceiling_bytes).unwrap_or(usize::MAX),
                usize::try_from(self.allocator_cache_bytes).unwrap_or(usize::MAX),
            )
            .map_err(|error| InferenceEngineError::Fatal {
                reason: format!("folded-ternary MLX limits failed: {error}"),
            })?;
            model
                .runtime
                .update_memory_limits(memory_limits)
                .map_err(|error| InferenceEngineError::Fatal {
                    reason: format!(
                        "folded-ternary could not apply the MLX memory ceiling: {error}"
                    ),
                })?;
        }
        let mlx_memory_telemetry = self.collect_current_mlx_memory_telemetry();
        Ok(MlxMemoryLimitAdjustment::new(
            self.ceiling_bytes,
            self.allocator_cache_bytes,
            self.minimum_ceiling_bytes,
            ExpertMemoryMode::Resident,
            mlx_memory_telemetry,
        ))
    }

    fn collect_persistent_prompt_cache_stats(
        &self,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        Ok(self.persistent_prompt_cache_stats_event())
    }

    fn clear_persistent_prompt_cache(
        &mut self,
        model_id: Option<String>,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        let Some(persistent_prompt_cache) = self.persistent_prompt_cache.as_ref() else {
            return Ok(None);
        };
        let clear_outcome = persistent_prompt_cache
            .clear_prompt_cache(model_id.as_deref())
            .map_err(|clear_error| InferenceEngineError::Fatal {
                reason: format!("folded-ternary prompt cache could not be cleared: {clear_error}"),
            })?;
        Ok(Some(WorkerEvent::PromptCacheCleared {
            model_id: clear_outcome.model_id,
            blocks_removed: clear_outcome.blocks_removed,
            bytes_freed: clear_outcome.bytes_freed,
        }))
    }
}

impl PrismHadamardQwen35InferenceExecution {
    /// Samples the current MLX memory footprint for the status surface.
    ///
    /// The dense folded-ternary model owns its whole weight payload, so the
    /// breakdown attributes active memory to the model core first and the
    /// live generation state second; there is no expert owner.
    pub(super) fn collect_current_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        let model = self.model.as_ref()?;
        let memory_snapshot = match model.runtime.memory_snapshot() {
            Ok(memory_snapshot) => memory_snapshot,
            Err(memory_snapshot_error) => {
                tracing::warn!(
                    error = %memory_snapshot_error,
                    "folded-ternary execution could not sample current MLX memory"
                );
                return None;
            }
        };
        let active_memory_bytes = u64::try_from(memory_snapshot.active_memory_bytes()).ok()?;
        let allocator_cache_memory_bytes =
            u64::try_from(memory_snapshot.allocator_cache_memory_bytes()).ok()?;
        let peak_memory_bytes = u64::try_from(memory_snapshot.peak_memory_bytes()).ok()?;
        let model_core_payload_bytes = model.pack().total_payload_bytes();
        let context_state_payload_bytes = self.active.as_ref().map_or(0, |active_generation| {
            active_generation.generation_state.payload_byte_count()
        });
        let active_memory_breakdown = MlxActiveMemoryBreakdown::reconcile(
            active_memory_bytes,
            0,
            model_core_payload_bytes,
            context_state_payload_bytes,
        );
        Some(MlxMemoryTelemetry::new(
            active_memory_bytes,
            allocator_cache_memory_bytes,
            peak_memory_bytes,
            active_memory_breakdown,
        ))
    }

    /// Advances one bounded prompt-processing chunk through the straight
    /// forward pass and reports prefill progress to the worker.
    #[expect(clippy::too_many_lines)]
    fn prefill_next_chunk(
        &mut self,
        request_id: RequestId,
        prompt_processing_chunk_tokens: u32,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "folded-ternary model is not loaded".to_owned(),
            })?;
        // Sampled before the active-generation borrow; chunk boundaries are
        // far enough apart that one-chunk-old memory reads stay honest.
        let mlx_memory_telemetry = self.collect_current_mlx_memory_telemetry();
        let active = self
            .active
            .as_mut()
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "folded-ternary has no active generation".to_owned(),
            })?;
        if active.request_id != request_id {
            return Err(InferenceEngineError::Fatal {
                reason: "folded-ternary prefill request correlation mismatch".to_owned(),
            });
        }
        let persistent_prompt_cache = self.persistent_prompt_cache.clone();
        let chunk_start = active.processed_prompt_token_count as usize;
        let requested_chunk_end = chunk_start
            + active
                .remaining_prompt_token_ids
                .len()
                .min(prompt_processing_chunk_tokens as usize);
        // A durable boundary snapshot is only valid at a forward edge, so the
        // chunk stops at the first block boundary it would cross; at most one
        // durable publication per forward keeps the boundary snapshot honest.
        let clamped_chunk_end = match persistent_prompt_cache.as_ref() {
            Some(persistent_prompt_cache) => {
                let block_token_count = persistent_prompt_cache
                    .model_contract_ref()
                    .block_token_count();
                persistent_prompt_cache_boundary_clamped_prefill_chunk_end(
                    chunk_start,
                    requested_chunk_end,
                    block_token_count,
                )
            }
            None => requested_chunk_end,
        };
        let chunk_length = clamped_chunk_end - chunk_start;
        let chunk_token_ids: Vec<i32> = active
            .remaining_prompt_token_ids
            .drain(..chunk_length)
            .map(|token_id| token_id as i32)
            .collect();
        let chunk_started_at = Instant::now();
        let token_indices = model
            .runtime
            .array_from_i32(&chunk_token_ids, &[1, chunk_length as i32])
            .map_err(gpu_error)?;
        let hidden_states = active
            .measure_operation(
                PerformanceOperation::PromptPrefillAdvanceSpan,
                |active_generation| {
                    model.forward_hidden_states(
                        &token_indices,
                        &mut active_generation.generation_state,
                    )
                },
            )
            .map_err(gpu_error)?;
        let final_logits = if active.remaining_prompt_token_ids.is_empty() {
            Some(
                active
                    .measure_operation(PerformanceOperation::FinalLogitsGraphConstruction, |_| {
                        model.logits_for_last_hidden_state(&hidden_states)
                    })
                    .map_err(gpu_error)?,
            )
        } else {
            None
        };
        if let Some(final_logits) = final_logits.as_ref() {
            active
                .measure_operation(
                    PerformanceOperation::PrefillStateAsyncEvaluationSubmission,
                    |_| {
                        model
                            .runtime
                            .evaluate_arrays(&[&hidden_states, final_logits])
                    },
                )
                .map_err(gpu_error)?;
        } else {
            active
                .measure_operation(
                    PerformanceOperation::PrefillStateAsyncEvaluationSubmission,
                    |_| model.runtime.evaluate_arrays(&[&hidden_states]),
                )
                .map_err(gpu_error)?;
        }
        let elapsed_millis = chunk_started_at
            .elapsed()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX);
        if let Some(final_logits) = final_logits.as_ref() {
            let first_token = active.measure_operation(
                PerformanceOperation::DecodeSamplingSpan,
                |active_generation| sample_token(model, final_logits, active_generation),
            )?;
            active
                .measure_operation(
                    PerformanceOperation::DecodeAsyncEvaluationSubmission,
                    |_| model.runtime.async_eval_arrays(&[&first_token]),
                )
                .map_err(gpu_error)?;
            active.pending_sampled_token =
                successor_stays_pending(active.remaining_output_tokens).then_some(first_token);
        }
        // Capture runs after the chunk's state graph was submitted: the
        // published tensors chain onto the lazy arrays and materialize during
        // publication, so no extra device synchronization happens here.
        active.processed_prompt_token_count += chunk_length as u32;
        if let Some(persistent_prompt_cache) = persistent_prompt_cache.clone() {
            let slot_layout = PrismStateSlotLayout::for_model(model);
            let mut capture_attribution = std::mem::replace(
                &mut active.performance_attribution,
                PerformanceAttribution::disabled(),
            );
            let capture_result = super::super::prompt_cache::capture_completed_cache_block(
                &model.runtime,
                &persistent_prompt_cache,
                active,
                &slot_layout,
                &mut capture_attribution,
            );
            active.performance_attribution = capture_attribution;
            capture_result?;
        }
        // The worker accumulates per-chunk deltas into the request-wide
        // prefill cursor, so this must stay the chunk's own token count — a
        // cumulative count double-counts every earlier chunk and pins the
        // status surface at 100% while most of the prompt is still processing.
        Ok(GeneratedToken::PrefillProgress {
            processed_token_count: chunk_length as u32,
            elapsed_millis,
            forward_prefill_chunk_elapsed_millis: elapsed_millis,
            completed_prefill_chunk_tokens: chunk_length as u32,
            mlx_memory_telemetry,
            expert_residency_telemetry: None,
            expert_memory_mode: Some(ExpertMemoryMode::Resident),
            prompt_work_reuse: WorkerPromptWorkReuse {
                target_eligible_token_count: active.prompt_token_ids.len() as u64,
                target_restored_token_count: u64::from(active.cached_token_count),
            },
            persistent_prompt_cache_diagnostics: None,
        })
    }
}

/// Builds the forward for `token_indices` (`[1, 1]`, host or device
/// resident), samples its successor lazily, and submits the whole step
/// asynchronously together with every retained state array. Returns the
/// lazy successor; nothing here waits for the graphics processor.
pub(super) fn submit_successor(
    model: &PrismHadamardTextModel,
    active: &mut PrismHadamardQwen35ActiveGeneration,
    token_indices: &MlxArray,
) -> Result<MlxArray, InferenceEngineError> {
    let logits = active
        .measure_operation(
            PerformanceOperation::DecodeAdvanceSpan,
            |active_generation| {
                model.forward_pass(token_indices, &mut active_generation.generation_state)
            },
        )
        .map_err(gpu_error)?;
    let successor = sample_token(model, &logits, active)?;
    active
        .measure_operation(
            PerformanceOperation::DecodeAsyncEvaluationSubmission,
            |active_generation| {
                model.runtime.async_eval_arrays(
                    &active_generation
                        .generation_state
                        .evaluation_arrays(&successor),
                )
            },
        )
        .map_err(gpu_error)?;
    Ok(successor)
}

/// Maps any boundary failure onto the engine error surface, keeping the
/// cause's own rendering as the fatal reason.
pub(super) fn gpu_error<E: std::fmt::Display>(error: E) -> InferenceEngineError {
    InferenceEngineError::Fatal {
        reason: error.to_string(),
    }
}
