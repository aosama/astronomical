use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationFailureReason, ChatGenerationOutput, MtpRuntimeState,
    SpeculativePrefillRuntimeState, WorkerEvent,
};

use crate::{
    DeepSeekV4UnavailableGenerationProcessor, LagunaGenerationProcessor,
    ModelGeneratedTokenTranslation, ModelGenerationOutputError, ModelGenerationProcessor,
    PreparedModelGeneration, Qwen3_5GenerationProcessor,
};

use super::{ModelFamilyInferenceRequest, ModelFamilyRequestOutput};

/// Family-tagged prompt and output processor used by EngineBackedWorker.
pub enum ModelFamilyGenerationProcessor {
    Qwen3_5(Qwen3_5GenerationProcessor),
    Laguna(LagunaGenerationProcessor),
    DeepSeekV4(DeepSeekV4UnavailableGenerationProcessor),
}

impl ModelGenerationProcessor for ModelFamilyGenerationProcessor {
    type InferenceRequest = ModelFamilyInferenceRequest;
    type RequestOutput = ModelFamilyRequestOutput;

    fn ready_event(
        &self,
        mtp_runtime_state: MtpRuntimeState,
        mtp_unavailable_reason: Option<String>,
        speculative_prefill_runtime_state: SpeculativePrefillRuntimeState,
        speculative_prefill_unavailable_reason: Option<String>,
        speculative_prefill_draft_model_id: Option<String>,
        speculative_prefill_draft_model_revision: Option<String>,
    ) -> WorkerEvent {
        match self {
            Self::Qwen3_5(processor) => processor.ready_event(
                mtp_runtime_state,
                mtp_unavailable_reason,
                speculative_prefill_runtime_state,
                speculative_prefill_unavailable_reason,
                speculative_prefill_draft_model_id,
                speculative_prefill_draft_model_revision,
            ),
            Self::Laguna(processor) => processor.ready_event(
                mtp_runtime_state,
                mtp_unavailable_reason,
                speculative_prefill_runtime_state,
                speculative_prefill_unavailable_reason,
                speculative_prefill_draft_model_id,
                speculative_prefill_draft_model_revision,
            ),
            Self::DeepSeekV4(processor) => processor.ready_event(
                mtp_runtime_state,
                mtp_unavailable_reason,
                speculative_prefill_runtime_state,
                speculative_prefill_unavailable_reason,
                speculative_prefill_draft_model_id,
                speculative_prefill_draft_model_revision,
            ),
        }
    }

    fn prepare_chat_generation(
        &self,
        chat_generation_command: &ChatGenerationCommand,
    ) -> Result<
        PreparedModelGeneration<Self::InferenceRequest, Self::RequestOutput>,
        ChatGenerationFailureReason,
    > {
        match self {
            Self::Qwen3_5(processor) => processor
                .prepare_chat_generation(chat_generation_command)
                .map(|prepared_generation| {
                    PreparedModelGeneration::new(
                        ModelFamilyInferenceRequest::Qwen3_5(prepared_generation.inference_request),
                        ModelFamilyRequestOutput::Qwen3_5(prepared_generation.request_output),
                    )
                }),
            Self::Laguna(processor) => processor
                .prepare_chat_generation(chat_generation_command)
                .map(|prepared_generation| {
                    PreparedModelGeneration::new(
                        ModelFamilyInferenceRequest::Laguna(prepared_generation.inference_request),
                        ModelFamilyRequestOutput::Laguna(prepared_generation.request_output),
                    )
                }),
            Self::DeepSeekV4(processor) => processor
                .prepare_chat_generation(chat_generation_command)
                .map(|prepared_generation| {
                    PreparedModelGeneration::new(
                        ModelFamilyInferenceRequest::DeepSeekV4(
                            prepared_generation.inference_request,
                        ),
                        ModelFamilyRequestOutput::DeepSeekV4(prepared_generation.request_output),
                    )
                }),
        }
    }

    fn is_end_of_sequence_token(&self, generated_token_id: u32) -> bool {
        match self {
            Self::Qwen3_5(processor) => processor.is_end_of_sequence_token(generated_token_id),
            Self::Laguna(processor) => processor.is_end_of_sequence_token(generated_token_id),
            Self::DeepSeekV4(processor) => processor.is_end_of_sequence_token(generated_token_id),
        }
    }

    fn translate_generated_token(
        &self,
        request_output: &mut Self::RequestOutput,
        generated_token_id: u32,
    ) -> Result<ModelGeneratedTokenTranslation, ModelGenerationOutputError> {
        match (self, request_output) {
            (Self::Qwen3_5(processor), ModelFamilyRequestOutput::Qwen3_5(request_output)) => {
                processor.translate_generated_token(request_output, generated_token_id)
            }
            (Self::Laguna(processor), ModelFamilyRequestOutput::Laguna(request_output)) => {
                processor.translate_generated_token(request_output, generated_token_id)
            }
            (Self::DeepSeekV4(processor), ModelFamilyRequestOutput::DeepSeekV4(request_output)) => {
                processor.translate_generated_token(request_output, generated_token_id)
            }
            _ => Err(ModelGenerationOutputError::Fatal {
                reason: "model-family processor and request output do not match".to_owned(),
            }),
        }
    }

    fn finish_request_output(
        &self,
        request_output: &mut Self::RequestOutput,
    ) -> Result<Vec<ChatGenerationOutput>, ModelGenerationOutputError> {
        match (self, request_output) {
            (Self::Qwen3_5(processor), ModelFamilyRequestOutput::Qwen3_5(request_output)) => {
                processor.finish_request_output(request_output)
            }
            (Self::Laguna(processor), ModelFamilyRequestOutput::Laguna(request_output)) => {
                processor.finish_request_output(request_output)
            }
            (Self::DeepSeekV4(processor), ModelFamilyRequestOutput::DeepSeekV4(request_output)) => {
                processor.finish_request_output(request_output)
            }
            _ => Err(ModelGenerationOutputError::Fatal {
                reason: "model-family processor and request output do not match".to_owned(),
            }),
        }
    }
}
