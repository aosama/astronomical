use crate::{AstroExpertCacheStatistics, InferenceEngineError};
use ipc_protocol::{ExpertMemoryMode, RequestId};

use crate::MlxInferenceEngine;

use super::{
    Qwen3_5EngineState, Qwen3_5InferenceExecution, fatal_engine_error, qwen3_5_runtime_error,
};

// These owner-thread operations intentionally live beside their asynchronous
// public wrappers. Keeping acceptance-only state mutation out of the engine
// owner prevents the production lifecycle module from becoming a test-control bag.
impl Qwen3_5EngineState {
    fn expert_memory_mode_for_tests(&self) -> Option<ExpertMemoryMode> {
        self.model
            .as_ref()
            .map(|loaded_model| loaded_model.expert_memory_mode())
    }

    fn astro_expert_cache_statistics_for_tests(
        &self,
    ) -> Result<AstroExpertCacheStatistics, InferenceEngineError> {
        self.model
            .as_ref()
            .map(|loaded_model| loaded_model.astro_expert_cache_statistics())
            .ok_or_else(|| fatal_engine_error("cannot inspect expert statistics before loading"))
    }

    fn complete_expert_payload_byte_count_for_tests(&self) -> Result<u64, InferenceEngineError> {
        let loaded_model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("cannot inspect expert payload before loading"))?;
        loaded_model
            .resident_expert_weights
            .as_ref()
            .map(crate::qwen3_5_resident::Qwen3_5ResidentExpertWeights::payload_byte_count)
            .ok_or_else(|| {
                fatal_engine_error("cannot inspect complete expert payload for a dense model")
            })
    }

    fn prompt_work_reuse_for_tests(
        &self,
        request_id: RequestId,
    ) -> Result<ipc_protocol::WorkerPromptWorkReuse, InferenceEngineError> {
        let active_request = self.active_request.as_ref().ok_or_else(|| {
            fatal_engine_error("cannot inspect prompt work reuse without an active request")
        })?;
        if active_request.request_id != request_id {
            return Err(fatal_engine_error(
                "cannot inspect prompt work reuse for a different request",
            ));
        }
        Ok(active_request.prompt_work_reuse)
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

    /// The memory governor's view after the last measured forward, with its
    /// accounts compared against what MLX reported.
    #[doc(hidden)]
    pub async fn memory_ledger_reading_for_tests(
        &self,
    ) -> Result<Option<memory_policy::ledger::LedgerReading>, InferenceEngineError> {
        let (ledger_reading_sender, ledger_reading_receiver) = std::sync::mpsc::sync_channel(1);
        self.run_owner_test_operation(move |qwen_inference_execution| {
            let latest_ledger_reading = qwen_inference_execution
                .model
                .as_ref()
                .ok_or_else(|| fatal_engine_error("no model is loaded to read the ledger of"))?
                .latest_memory_ledger_reading();
            ledger_reading_sender
                .send(latest_ledger_reading)
                .map_err(|_| fatal_engine_error("ledger reading receiver stopped unexpectedly"))
        })
        .await?;
        ledger_reading_receiver
            .recv()
            .map_err(|_| fatal_engine_error("ledger reading owner operation returned no response"))
    }

    /// Returns mode-neutral expert ownership plus cumulative paging statistics.
    #[doc(hidden)]
    pub async fn astro_expert_cache_statistics_for_tests(
        &self,
    ) -> Result<AstroExpertCacheStatistics, InferenceEngineError> {
        let (expert_statistics_sender, expert_statistics_receiver) =
            std::sync::mpsc::sync_channel(1);
        self.run_owner_test_operation(move |qwen_inference_execution| {
            let expert_statistics =
                qwen_inference_execution.astro_expert_cache_statistics_for_tests()?;
            expert_statistics_sender
                .send(expert_statistics)
                .map_err(|_| fatal_engine_error("expert statistics receiver stopped unexpectedly"))
        })
        .await?;
        expert_statistics_receiver.recv().map_err(|_| {
            fatal_engine_error("expert statistics owner operation returned no response")
        })
    }

    /// Returns the exact complete expert payload used by residency admission.
    #[doc(hidden)]
    pub async fn complete_expert_payload_byte_count_for_tests(
        &self,
    ) -> Result<u64, InferenceEngineError> {
        let (complete_expert_payload_sender, complete_expert_payload_receiver) =
            std::sync::mpsc::sync_channel(1);
        self.run_owner_test_operation(move |qwen_inference_execution| {
            let complete_expert_payload_bytes =
                qwen_inference_execution.complete_expert_payload_byte_count_for_tests()?;
            complete_expert_payload_sender
                .send(complete_expert_payload_bytes)
                .map_err(|_| {
                    fatal_engine_error("complete expert payload receiver stopped unexpectedly")
                })
        })
        .await?;
        complete_expert_payload_receiver.recv().map_err(|_| {
            fatal_engine_error("complete expert payload owner operation returned no response")
        })
    }

    /// Returns truthful per-model prompt work for an active acceptance request.
    #[doc(hidden)]
    pub async fn prompt_work_reuse_for_tests(
        &self,
        request_id: RequestId,
    ) -> Result<ipc_protocol::WorkerPromptWorkReuse, InferenceEngineError> {
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
