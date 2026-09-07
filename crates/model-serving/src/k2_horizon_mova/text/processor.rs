//! Chat preparation and generated-token translation for K2 Horizon MoVA.

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationFailureReason, ChatGenerationOutput,
    ChatModelCapabilities, MtpDepthStatus, MtpRuntimeState, SpeculativePrefillRuntimeState,
    WorkerEvent,
};

use crate::k2_horizon_mova::artifacts::ValidatedK2HorizonMoVAArtifact;
use crate::{
    ModelGeneratedTokenTranslation, ModelGenerationOutputError, ModelGenerationProcessor,
    PreparedModelGeneration,
};

use super::thinking_budget::resolve_k2_horizon_mova_thinking_budget;
use super::{
    K2HorizonMoVAInferenceRequest, K2HorizonMoVAPromptRenderer, K2HorizonMoVARequestOutput,
    K2HorizonMoVATokenizer, K2HorizonMoVATokenizerError,
};

const DEFAULT_TEMPERATURE_THOUSANDTHS: u16 = 1_000;
const DEFAULT_TOP_P_THOUSANDTHS: u16 = 950;

/// Family processor used by EngineBackedWorker.
#[derive(Clone, Debug)]
pub struct K2HorizonMoVAGenerationProcessor {
    model_id: String,
    tokenizer: K2HorizonMoVATokenizer,
    prompt_renderer: K2HorizonMoVAPromptRenderer,
    maximum_context_tokens: u32,
    maximum_output_tokens: u32,
}

impl K2HorizonMoVAGenerationProcessor {
    pub fn from_validated_artifact(
        validated_artifact: &ValidatedK2HorizonMoVAArtifact,
        maximum_context_tokens: Option<u32>,
        maximum_output_tokens: Option<u32>,
    ) -> Result<Self, K2HorizonMoVATokenizerError> {
        let config = validated_artifact.config();
        let tokenizer =
            K2HorizonMoVATokenizer::from_json_bytes(validated_artifact.tokenizer_bytes(), config)?;
        let context_window = maximum_context_tokens
            .unwrap_or(config.max_position_embeddings())
            .min(config.max_position_embeddings());
        let maximum_output_tokens = maximum_output_tokens
            .unwrap_or(u32::from(u16::MAX))
            .min(context_window.saturating_sub(1));
        Ok(Self {
            model_id: validated_artifact.model_id().to_owned(),
            tokenizer,
            prompt_renderer: K2HorizonMoVAPromptRenderer::new(),
            maximum_context_tokens: context_window,
            maximum_output_tokens,
        })
    }
}

impl ModelGenerationProcessor for K2HorizonMoVAGenerationProcessor {
    type InferenceRequest = K2HorizonMoVAInferenceRequest;
    type RequestOutput = K2HorizonMoVARequestOutput;

    fn ready_event(
        &self,
        mtp_runtime_state: MtpRuntimeState,
        mtp_unavailable_reason: Option<String>,
        mtp_depth_status: MtpDepthStatus,
        speculative_prefill_runtime_state: SpeculativePrefillRuntimeState,
        speculative_prefill_unavailable_reason: Option<String>,
        speculative_prefill_draft_model_id: Option<String>,
        speculative_prefill_draft_model_revision: Option<String>,
    ) -> WorkerEvent {
        WorkerEvent::Ready {
            model_id: self.model_id.clone(),
            capabilities: ChatModelCapabilities {
                supports_reasoning: true,
                supports_tool_calls: true,
                has_vision: false,
                max_input_tokens: self.maximum_context_tokens.saturating_sub(1),
                max_output_tokens: self.maximum_output_tokens,
                context_window: self.maximum_context_tokens,
            }
            .into(),
            mtp_runtime_state,
            mtp_unavailable_reason,
            mtp_depth_status,
            speculative_prefill_runtime_state,
            speculative_prefill_unavailable_reason,
            speculative_prefill_draft_model_id,
            speculative_prefill_draft_model_revision,
        }
    }

    fn prepare_chat_generation(
        &self,
        chat_generation_command: &ChatGenerationCommand,
    ) -> Result<
        PreparedModelGeneration<Self::InferenceRequest, Self::RequestOutput>,
        ChatGenerationFailureReason,
    > {
        let prompt = self.prompt_renderer.render(
            &chat_generation_command.messages,
            &chat_generation_command.tools,
            &chat_generation_command.tool_choice,
        );
        let prompt_token_ids = self.tokenizer.encode_prompt(&prompt).map_err(|_| {
            ChatGenerationFailureReason::invalid_request("K2 Horizon MoVA prompt encoding failed")
        })?;
        let requested_output_tokens = u32::from(chat_generation_command.settings.max_output_tokens)
            .min(self.maximum_output_tokens);
        let total_tokens = prompt_token_ids
            .len()
            .saturating_add(requested_output_tokens as usize);
        if total_tokens as u32 > self.maximum_context_tokens {
            return Err(ChatGenerationFailureReason::ContextLengthExceeded {
                actual_total_context_tokens: total_tokens as u32,
                maximum_context_tokens: self.maximum_context_tokens,
            });
        }
        let temperature_thousandths = chat_generation_command
            .settings
            .temperature_thousandths
            .filter(|temperature| *temperature > 0)
            .unwrap_or(DEFAULT_TEMPERATURE_THOUSANDTHS);
        let top_p_thousandths = chat_generation_command
            .settings
            .top_p_thousandths
            .unwrap_or(DEFAULT_TOP_P_THOUSANDTHS);
        let thinking_budget = resolve_k2_horizon_mova_thinking_budget(
            chat_generation_command.settings.thinking_budget,
            requested_output_tokens,
            self.tokenizer.think_close_token_ids().len(),
        );
        Ok(PreparedModelGeneration::new(
            K2HorizonMoVAInferenceRequest::with_thinking_budget(
                prompt_token_ids,
                requested_output_tokens,
                temperature_thousandths,
                top_p_thousandths,
                chat_generation_command.settings.seed,
                thinking_budget,
                self.tokenizer.think_close_token_ids().to_vec(),
                self.tokenizer.natural_reasoning_end_token_ids().to_vec(),
            ),
            K2HorizonMoVARequestOutput::new_with_declared_tool_names(
                &self.tokenizer,
                chat_generation_command
                    .tools
                    .iter()
                    .map(|tool| tool.name.clone())
                    .collect(),
            ),
        ))
    }

    fn is_end_of_sequence_token(&self, generated_token_id: u32) -> bool {
        self.tokenizer.is_end_of_sequence_token(generated_token_id)
    }

    fn translate_generated_token(
        &self,
        request_output: &mut Self::RequestOutput,
        generated_token_id: u32,
    ) -> Result<ModelGeneratedTokenTranslation, ModelGenerationOutputError> {
        Ok(ModelGeneratedTokenTranslation::from_outputs(
            request_output.push_token(generated_token_id)?,
        ))
    }

    fn finish_request_output(
        &self,
        request_output: &mut Self::RequestOutput,
    ) -> Result<Vec<ChatGenerationOutput>, ModelGenerationOutputError> {
        Ok(request_output.finish())
    }
}
