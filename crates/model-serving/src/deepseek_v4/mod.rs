//! Structural placeholder for the future DeepSeek-V4 model family.

use astronomical_ipc_protocol::{ChatGenerationFailureReason, WorkerEvent};

use crate::{
    ModelGeneratedTokenTranslation, ModelGenerationOutputError, ModelGenerationProcessor,
    PreparedInferenceRequest, PreparedModelGeneration,
};

const DEEPSEEK_V4_UNAVAILABLE_REASON: &str =
    "DeepSeek-V4 model execution is not implemented in this build";

/// Prepared request placeholder used only to close the family enum boundary.
#[derive(Debug)]
pub struct DeepSeekV4UnavailableInferenceRequest;

impl PreparedInferenceRequest for DeepSeekV4UnavailableInferenceRequest {
    fn prompt_token_count(&self) -> usize {
        0
    }
}

/// Request output placeholder used only to close the family enum boundary.
#[derive(Debug)]
pub struct DeepSeekV4UnavailableRequestOutput;

/// Engine placeholder used only to close the family enum boundary.
#[derive(Debug, Default)]
pub struct DeepSeekV4UnavailableInferenceEngine;

/// Processor placeholder that never accepts a generation request.
#[derive(Debug, Default)]
pub struct DeepSeekV4UnavailableGenerationProcessor;

impl ModelGenerationProcessor for DeepSeekV4UnavailableGenerationProcessor {
    type InferenceRequest = DeepSeekV4UnavailableInferenceRequest;
    type RequestOutput = DeepSeekV4UnavailableRequestOutput;

    fn ready_event(
        &self,
        _mtp_runtime_state: astronomical_ipc_protocol::MtpRuntimeState,
        _mtp_unavailable_reason: Option<String>,
        _mtp_depth_status: astronomical_ipc_protocol::MtpDepthStatus,
        _speculative_prefill_runtime_state:
            astronomical_ipc_protocol::SpeculativePrefillRuntimeState,
        _speculative_prefill_unavailable_reason: Option<String>,
        _speculative_prefill_draft_model_id: Option<String>,
        _speculative_prefill_draft_model_revision: Option<String>,
    ) -> WorkerEvent {
        WorkerEvent::ModelSwapFailed {
            loaded_model_remains_ready: false,
            model_load_failure_reason: DEEPSEEK_V4_UNAVAILABLE_REASON.to_owned(),
        }
    }

    fn prepare_chat_generation(
        &self,
        _chat_generation_command: &astronomical_ipc_protocol::ChatGenerationCommand,
    ) -> Result<
        PreparedModelGeneration<Self::InferenceRequest, Self::RequestOutput>,
        ChatGenerationFailureReason,
    > {
        Err(ChatGenerationFailureReason::invalid_request(
            DEEPSEEK_V4_UNAVAILABLE_REASON,
        ))
    }

    fn is_end_of_sequence_token(&self, _generated_token_id: u32) -> bool {
        false
    }

    fn translate_generated_token(
        &self,
        _request_output: &mut Self::RequestOutput,
        _generated_token_id: u32,
    ) -> Result<ModelGeneratedTokenTranslation, ModelGenerationOutputError> {
        Err(ModelGenerationOutputError::Fatal {
            reason: DEEPSEEK_V4_UNAVAILABLE_REASON.to_owned(),
        })
    }

    fn finish_request_output(
        &self,
        _request_output: &mut Self::RequestOutput,
    ) -> Result<Vec<astronomical_ipc_protocol::ChatGenerationOutput>, ModelGenerationOutputError>
    {
        Err(ModelGenerationOutputError::Fatal {
            reason: DEEPSEEK_V4_UNAVAILABLE_REASON.to_owned(),
        })
    }
}

/// Returns the bounded reason used by the structural DeepSeek placeholder.
#[must_use]
pub const fn deepseek_v4_unavailable_reason() -> &'static str {
    DEEPSEEK_V4_UNAVAILABLE_REASON
}
