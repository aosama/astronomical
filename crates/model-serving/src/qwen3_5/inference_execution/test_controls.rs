use crate::InferenceEngineError;
use astronomical_ipc_protocol::RequestId;

use crate::MlxInferenceEngine;

use super::{
    Qwen3_5EngineState, Qwen3_5InferenceExecution, Qwen3_5SpeculativePrefillFailureStageForTests,
    fatal_engine_error, qwen3_5_runtime_error,
};

// These owner-thread operations intentionally live beside their asynchronous
// public wrappers. Keeping qualification-only state mutation out of the engine
// owner prevents the production lifecycle module from becoming a test-control bag.
impl Qwen3_5EngineState {
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
