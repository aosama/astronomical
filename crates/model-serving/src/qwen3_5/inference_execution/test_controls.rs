use crate::{ExpertWeightMemoryCacheStatistics, InferenceEngineError};
use astronomical_ipc_protocol::{ExpertMemoryMode, RequestId};

use crate::MlxInferenceEngine;

use super::{
    Qwen3_5EngineState, Qwen3_5InferenceExecution, Qwen3_5SpeculativePrefillFailureStageForTests,
    fatal_engine_error, qwen3_5_runtime_error,
};

const TEST_MTP_FULL_ATTENTION_GROWTH_TOKENS: i32 = 1;

// These owner-thread operations intentionally live beside their asynchronous
// public wrappers. Keeping qualification-only state mutation out of the engine
// owner prevents the production lifecycle module from becoming a test-control bag.
impl Qwen3_5EngineState {
    fn expert_memory_mode_for_tests(&self) -> Option<ExpertMemoryMode> {
        self.model
            .as_ref()
            .map(|loaded_model| loaded_model.expert_memory_mode())
    }

    fn expert_weight_memory_cache_statistics_for_tests(
        &self,
    ) -> Result<ExpertWeightMemoryCacheStatistics, InferenceEngineError> {
        self.model
            .as_ref()
            .map(|loaded_model| loaded_model.expert_weight_memory_cache_statistics())
            .ok_or_else(|| fatal_engine_error("cannot inspect expert statistics before loading"))
    }

    fn remove_resident_expert_source_files_for_tests(
        &mut self,
    ) -> Result<(), InferenceEngineError> {
        let loaded_model = self
            .model
            .as_mut()
            .ok_or_else(|| fatal_engine_error("cannot remove expert sources before loading"))?;
        let expert_pager = loaded_model.expert_pager.as_mut().ok_or_else(|| {
            fatal_engine_error("cannot remove resident expert sources from a dense model")
        })?;
        expert_pager.remove_resident_expert_source_files_for_tests();
        Ok(())
    }

    fn native_expert_retention_growth_is_enabled_for_tests(
        &self,
    ) -> Result<bool, InferenceEngineError> {
        let loaded_model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("cannot inspect expert retention before loading"))?;
        let expert_pager = loaded_model.expert_pager.as_ref().ok_or_else(|| {
            fatal_engine_error("cannot inspect native expert retention for a dense model")
        })?;
        Ok(expert_pager.native_expert_retention_growth_is_enabled_for_tests())
    }

    fn execute_resident_mtp_draft_for_tests(
        &self,
        next_token_id: u32,
    ) -> Result<u32, InferenceEngineError> {
        let loaded_model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("cannot execute resident MTP before loading"))?;
        if loaded_model.expert_memory_mode() != ExpertMemoryMode::Resident {
            return Err(fatal_engine_error(
                "cannot execute the resident MTP qualification while experts are paged",
            ));
        }
        let next_token_indices = loaded_model
            .runtime()
            .array_from_u32(&[next_token_id], &[1, 1])
            .map_err(qwen3_5_runtime_error)?;
        let hidden_states_for_mtp_fusion = loaded_model
            .embedding_lookup(&next_token_indices)
            .map_err(InferenceEngineError::from)?;
        let mut mtp_request_state = crate::Qwen3_5MtpRequestState::empty_with_growth_tokens(
            TEST_MTP_FULL_ATTENTION_GROWTH_TOKENS,
        )
        .map_err(qwen3_5_runtime_error)?;
        let mtp_forward_output = loaded_model
            .forward_mtp_draft(
                &hidden_states_for_mtp_fusion,
                &next_token_indices,
                &mut mtp_request_state,
            )
            .map_err(InferenceEngineError::from)?;
        loaded_model
            .greedy_token_id(mtp_forward_output.draft_logits())
            .map_err(InferenceEngineError::from)
    }

    fn prompt_work_reuse_for_tests(
        &self,
        request_id: RequestId,
    ) -> Result<astronomical_ipc_protocol::WorkerPromptWorkReuse, InferenceEngineError> {
        let active_request = self.active_request.as_ref().ok_or_else(|| {
            fatal_engine_error("cannot inspect prompt work reuse without an active request")
        })?;
        if active_request.request_id != request_id {
            return Err(fatal_engine_error(
                "cannot inspect prompt work reuse for a different request",
            ));
        }
        Ok(active_request.prompt_work_reuse.clone())
    }

    fn speculative_prefill_selected_token_positions_for_tests(
        &self,
        request_id: RequestId,
    ) -> Result<Option<Vec<usize>>, InferenceEngineError> {
        let active_request = self.active_request.as_ref().ok_or_else(|| {
            fatal_engine_error(
                "cannot inspect speculative-prefill selected positions without an active request",
            )
        })?;
        if active_request.request_id != request_id {
            return Err(fatal_engine_error(
                "cannot inspect speculative-prefill selected positions for a different request",
            ));
        }
        Ok(active_request
            .speculative_prefill_selected_token_positions
            .clone())
    }

    fn force_next_prefill_capacity_rejection_for_tests(
        &mut self,
        request_id: RequestId,
    ) -> Result<(), InferenceEngineError> {
        let active_request = self.active_request.as_mut().ok_or_else(|| {
            fatal_engine_error("cannot force prefill rejection without an active request")
        })?;
        if active_request.request_id != request_id {
            return Err(fatal_engine_error(
                "cannot force prefill rejection for a different request",
            ));
        }
        active_request.force_next_prefill_capacity_rejection_for_tests = true;
        Ok(())
    }

    fn force_next_speculative_prefill_draft_prefix_restore_failure_for_tests(
        &mut self,
        request_id: RequestId,
    ) -> Result<(), InferenceEngineError> {
        let active_request = self.active_request.as_mut().ok_or_else(|| {
            fatal_engine_error(
                "cannot force speculative-prefill draft-prefix restore failure without an active request",
            )
        })?;
        if active_request.request_id != request_id {
            return Err(fatal_engine_error(
                "cannot force speculative-prefill draft-prefix restore failure for a different request",
            ));
        }
        active_request.force_next_speculative_prefill_draft_prefix_restore_failure_for_tests = true;
        Ok(())
    }

    fn force_next_speculative_prefill_failure_for_tests(
        &mut self,
        request_id: RequestId,
        failure_stage: Qwen3_5SpeculativePrefillFailureStageForTests,
    ) -> Result<(), InferenceEngineError> {
        let active_request = self.active_request.as_mut().ok_or_else(|| {
            fatal_engine_error(
                "cannot force a speculative-prefill failure without an active request",
            )
        })?;
        if active_request.request_id != request_id {
            return Err(fatal_engine_error(
                "cannot force a speculative-prefill failure for a different request",
            ));
        }
        active_request.forced_speculative_prefill_failure_stage_for_tests = Some(failure_stage);
        Ok(())
    }

    fn force_next_mtp_draft_rejection_for_tests(
        &mut self,
        request_id: RequestId,
    ) -> Result<(), InferenceEngineError> {
        let active_request = self.active_request.as_mut().ok_or_else(|| {
            fatal_engine_error("cannot force MTP rejection without an active request")
        })?;
        if active_request.request_id != request_id {
            return Err(fatal_engine_error(
                "cannot force MTP rejection for a different request",
            ));
        }
        if let Some(optional_prediction_session) = active_request.optional_prediction_session_mut()
        {
            optional_prediction_session.force_next_draft_rejection_for_tests();
            Ok(())
        } else {
            Err(fatal_engine_error(
                "cannot force MTP rejection for a target-only request",
            ))
        }
    }
}

impl MlxInferenceEngine<Qwen3_5InferenceExecution> {
    /// Returns the currently loaded model's truthful expert-memory mode.
    #[doc(hidden)]
    pub async fn expert_memory_mode_for_tests(
        &self,
    ) -> Result<Option<ExpertMemoryMode>, InferenceEngineError> {
        let (expert_memory_mode_sender, expert_memory_mode_receiver) =
            std::sync::mpsc::sync_channel(1);
        self.run_owner_test_operation(move |qwen_inference_execution| {
            expert_memory_mode_sender
                .send(qwen_inference_execution.expert_memory_mode_for_tests())
                .map_err(|_| fatal_engine_error("expert-memory mode receiver stopped unexpectedly"))
        })
        .await?;
        expert_memory_mode_receiver.recv().map_err(|_| {
            fatal_engine_error("expert-memory mode owner operation returned no response")
        })
    }

    /// Returns mode-neutral expert ownership plus cumulative paging statistics.
    #[doc(hidden)]
    pub async fn expert_weight_memory_cache_statistics_for_tests(
        &self,
    ) -> Result<ExpertWeightMemoryCacheStatistics, InferenceEngineError> {
        let (expert_statistics_sender, expert_statistics_receiver) =
            std::sync::mpsc::sync_channel(1);
        self.run_owner_test_operation(move |qwen_inference_execution| {
            let expert_statistics =
                qwen_inference_execution.expert_weight_memory_cache_statistics_for_tests()?;
            expert_statistics_sender
                .send(expert_statistics)
                .map_err(|_| fatal_engine_error("expert statistics receiver stopped unexpectedly"))
        })
        .await?;
        expert_statistics_receiver.recv().map_err(|_| {
            fatal_engine_error("expert statistics owner operation returned no response")
        })
    }

    /// Removes resident-promotion source descriptors to qualify typed failure recovery.
    #[doc(hidden)]
    pub async fn remove_resident_expert_source_files_for_tests(
        &self,
    ) -> Result<(), InferenceEngineError> {
        self.run_owner_test_operation(|qwen_inference_execution| {
            qwen_inference_execution.remove_resident_expert_source_files_for_tests()
        })
        .await
    }

    /// Probes whether native expert retention can grow, restoring its original enabled state.
    #[doc(hidden)]
    pub async fn native_expert_retention_growth_is_enabled_for_tests(
        &self,
    ) -> Result<bool, InferenceEngineError> {
        let (retention_state_sender, retention_state_receiver) = std::sync::mpsc::sync_channel(1);
        self.run_owner_test_operation(move |qwen_inference_execution| {
            let retention_growth_is_enabled =
                qwen_inference_execution.native_expert_retention_growth_is_enabled_for_tests()?;
            retention_state_sender
                .send(retention_growth_is_enabled)
                .map_err(|_| {
                    fatal_engine_error("expert retention state receiver stopped unexpectedly")
                })
        })
        .await?;
        retention_state_receiver.recv().map_err(|_| {
            fatal_engine_error("expert retention state owner operation returned no response")
        })
    }

    /// Executes one real MTP head forward from a supplied fixture token on the owner thread.
    #[doc(hidden)]
    pub async fn execute_resident_mtp_draft_for_tests(
        &self,
        next_token_id: u32,
    ) -> Result<u32, InferenceEngineError> {
        let (draft_token_sender, draft_token_receiver) = std::sync::mpsc::sync_channel(1);
        self.run_owner_test_operation(move |qwen_inference_execution| {
            let draft_token_id =
                qwen_inference_execution.execute_resident_mtp_draft_for_tests(next_token_id)?;
            draft_token_sender.send(draft_token_id).map_err(|_| {
                fatal_engine_error("resident MTP draft token receiver stopped unexpectedly")
            })
        })
        .await?;
        draft_token_receiver
            .recv()
            .map_err(|_| fatal_engine_error("resident MTP owner operation returned no draft token"))
    }

    /// Returns truthful per-model prompt work for an active qualification request.
    #[doc(hidden)]
    pub async fn prompt_work_reuse_for_tests(
        &self,
        request_id: RequestId,
    ) -> Result<astronomical_ipc_protocol::WorkerPromptWorkReuse, InferenceEngineError> {
        let (prompt_work_reuse_sender, prompt_work_reuse_receiver) =
            std::sync::mpsc::sync_channel(1);
        self.run_owner_test_operation(move |qwen_inference_execution| {
            let prompt_work_reuse =
                qwen_inference_execution.prompt_work_reuse_for_tests(request_id)?;
            prompt_work_reuse_sender
                .send(prompt_work_reuse)
                .map_err(|_| fatal_engine_error("prompt work reuse receiver stopped unexpectedly"))
        })
        .await?;
        prompt_work_reuse_receiver.recv().map_err(|_| {
            fatal_engine_error("prompt work reuse owner operation returned no response")
        })
    }

    /// Returns the target conversation positions selected for an active qualification request.
    #[doc(hidden)]
    pub async fn speculative_prefill_selected_token_positions_for_tests(
        &self,
        request_id: RequestId,
    ) -> Result<Option<Vec<usize>>, InferenceEngineError> {
        let (selected_token_positions_sender, selected_token_positions_receiver) =
            std::sync::mpsc::sync_channel(1);
        self.run_owner_test_operation(move |qwen_inference_execution| {
            let selected_token_positions = qwen_inference_execution
                .speculative_prefill_selected_token_positions_for_tests(request_id)?;
            selected_token_positions_sender
                .send(selected_token_positions)
                .map_err(|_| {
                    fatal_engine_error(
                        "selected speculative-prefill positions receiver stopped unexpectedly",
                    )
                })
        })
        .await?;
        selected_token_positions_receiver.recv().map_err(|_| {
            fatal_engine_error(
                "selected speculative-prefill positions owner operation returned no response",
            )
        })
    }

    /// Rejects one completed prefill attempt so qualification can verify full retry rollback.
    pub async fn force_next_prefill_capacity_rejection_for_tests(
        &self,
        request_id: RequestId,
    ) -> Result<(), InferenceEngineError> {
        self.run_owner_test_operation(move |qwen_inference_execution| {
            qwen_inference_execution.force_next_prefill_capacity_rejection_for_tests(request_id)
        })
        .await
    }

    /// Forces one draft-prefix restore failure so qualification can verify uncached retry.
    pub async fn force_next_speculative_prefill_draft_prefix_restore_failure_for_tests(
        &self,
        request_id: RequestId,
    ) -> Result<(), InferenceEngineError> {
        self.run_owner_test_operation(move |qwen_inference_execution| {
            qwen_inference_execution
                .force_next_speculative_prefill_draft_prefix_restore_failure_for_tests(request_id)
        })
        .await
    }

    /// Forces one configured SpecPrefill stage to fail through its production error boundary.
    #[doc(hidden)]
    pub async fn force_next_speculative_prefill_failure_for_tests(
        &self,
        request_id: RequestId,
        failure_stage: Qwen3_5SpeculativePrefillFailureStageForTests,
    ) -> Result<(), InferenceEngineError> {
        self.run_owner_test_operation(move |qwen_inference_execution| {
            qwen_inference_execution
                .force_next_speculative_prefill_failure_for_tests(request_id, failure_stage)
        })
        .await
    }

    /// Arms one deterministic draft rejection through the production acceptance path.
    pub async fn force_next_mtp_draft_rejection_for_tests(
        &self,
        request_id: RequestId,
    ) -> Result<(), InferenceEngineError> {
        self.run_owner_test_operation(move |qwen_inference_execution| {
            qwen_inference_execution.force_next_mtp_draft_rejection_for_tests(request_id)
        })
        .await
    }

    /// Disables adaptive growth guarding and its memory sampling for benchmarks.
    #[doc(hidden)]
    pub async fn disable_adaptive_ram_growth_memory_guard_for_tests(
        &self,
    ) -> Result<(), InferenceEngineError> {
        self.run_owner_test_operation(|qwen_inference_execution| {
            qwen_inference_execution.adaptive_ram_growth_guard_enabled = false;
            Ok(())
        })
        .await
    }

    /// Resets the process-global MLX peak counter on the engine owner thread.
    #[doc(hidden)]
    pub async fn reset_mlx_peak_memory_for_tests(&self) -> Result<(), InferenceEngineError> {
        self.run_owner_test_operation(|qwen_inference_execution| {
            qwen_inference_execution
                .model
                .as_ref()
                .ok_or_else(|| {
                    fatal_engine_error("cannot reset MLX peak memory before the model is loaded")
                })?
                .runtime()
                .reset_peak_memory()
                .map_err(qwen3_5_runtime_error)
        })
        .await
    }
}
