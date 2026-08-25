use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationFailureReason, ChatGenerationOutput, ChatMessage,
    ChatModelCapabilities, MtpRuntimeState, SpeculativePrefillRuntimeState, WorkerEvent,
};

use crate::{
    ModelGeneratedTokenTranslation, ModelGenerationOutputError, ModelGenerationProcessor,
    PerformanceAttribution, PreparedModelGeneration, Qwen3_5InferenceRequest,
};

use super::{
    Qwen3_5OutputEvent, Qwen3_5RequestOutput, Qwen3_5RequestOutputError, Qwen3_5Tokenizer,
    Qwen3_5TokenizerError, ValidatedQwen3_5Artifact,
};

/// Model-specific structured-chat preparation and output translation for Qwen3.5.
#[derive(Clone, Debug)]
pub struct Qwen3_5GenerationProcessor {
    enable_thinking: bool,
    tokenizer: Qwen3_5Tokenizer,
    context_window: u32,
    max_input_tokens: u32,
    max_output_tokens: u32,
    supports_image_input: bool,
    model_id: String,
    performance_attribution_enabled: bool,
}

impl Qwen3_5GenerationProcessor {
    /// Creates a processor from a complete validated local Qwen3.5 artifact.
    pub fn from_validated_artifact(
        validated_artifact: &ValidatedQwen3_5Artifact,
        enable_thinking: bool,
        performance_attribution_enabled: bool,
    ) -> Result<Self, Qwen3_5TokenizerError> {
        Self::from_validated_artifact_with_effective_policy(
            validated_artifact,
            validated_artifact.model_id().to_owned(),
            validated_artifact.config().maximum_position_count(),
            enable_thinking,
            performance_attribution_enabled,
        )
    }

    pub fn from_validated_artifact_with_effective_policy(
        validated_artifact: &ValidatedQwen3_5Artifact,
        requested_model_id: String,
        context_window: u32,
        enable_thinking: bool,
        performance_attribution_enabled: bool,
    ) -> Result<Self, Qwen3_5TokenizerError> {
        let tokenizer = Qwen3_5Tokenizer::from_validated_artifact_with_maximum_context_tokens(
            validated_artifact,
            context_window,
        )?;
        let max_output_tokens = validated_artifact.max_output_tokens();
        context_window.checked_sub(max_output_tokens).ok_or(
            Qwen3_5TokenizerError::ModelOutputBudgetExceedsContextWindow {
                context_window,
                max_output_tokens,
            },
        )?;
        let max_input_tokens = context_window.saturating_sub(1);
        Ok(Self {
            enable_thinking,
            tokenizer,
            context_window,
            max_input_tokens,
            max_output_tokens,
            supports_image_input: validated_artifact.supports_image_input(),
            model_id: requested_model_id,
            performance_attribution_enabled,
        })
    }

    /// Returns the token ID that closes the thinking block.
    #[must_use]
    pub const fn think_end_token_id(&self) -> u32 {
        self.tokenizer.think_end_token_id()
    }
}

impl ModelGenerationProcessor for Qwen3_5GenerationProcessor {
    type InferenceRequest = Qwen3_5InferenceRequest;
    type RequestOutput = Qwen3_5RequestOutput;

    fn ready_event(
        &self,
        mtp_runtime_state: MtpRuntimeState,
        mtp_unavailable_reason: Option<String>,
        mtp_depth_status: astronomical_ipc_protocol::MtpDepthStatus,
        speculative_prefill_runtime_state: SpeculativePrefillRuntimeState,
        speculative_prefill_unavailable_reason: Option<String>,
        speculative_prefill_draft_model_id: Option<String>,
        speculative_prefill_draft_model_revision: Option<String>,
    ) -> WorkerEvent {
        WorkerEvent::Ready {
            model_id: self.model_id.clone(),
            capabilities: ChatModelCapabilities {
                supports_reasoning: self.enable_thinking,
                supports_tool_calls: true,
                has_vision: self.supports_image_input,
                max_input_tokens: self.max_input_tokens,
                max_output_tokens: self.max_output_tokens,
                context_window: self.context_window,
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
        if u32::from(chat_generation_command.settings.max_output_tokens) > self.max_output_tokens {
            return Err(ChatGenerationFailureReason::invalid_request(format!(
                "requested output limit exceeds the loaded model maximum of {} tokens",
                self.max_output_tokens
            )));
        }
        if !self.supports_image_input
            && chat_generation_command.messages.iter().any(|chat_message| {
                matches!(chat_message, ChatMessage::User { images, .. } if !images.is_empty())
            })
        {
            return Err(ChatGenerationFailureReason::invalid_request(
                "the selected model supports text input only and cannot process images",
            ));
        }
        let mut performance_attribution = if self.performance_attribution_enabled {
            PerformanceAttribution::enabled()
        } else {
            PerformanceAttribution::disabled()
        };
        let request_enable_thinking = qwen3_5_request_enables_thinking(
            self.enable_thinking,
            chat_generation_command.settings.thinking_budget,
        );
        let inference_request = self
            .tokenizer
            .prepare_chat_with_performance_attribution(
                chat_generation_command,
                request_enable_thinking,
                &mut performance_attribution,
            )
            .map_err(|validation_error| {
                tracing::warn!(
                    request_id = chat_generation_command.request_id.value(),
                    error = %validation_error,
                    "rejected chat thread during Qwen3.5 prompt preparation"
                );
                translate_qwen3_5_preparation_error(validation_error)
            })?;
        tracing::info!(
            request_id = chat_generation_command.request_id.value(),
            message_count = chat_generation_command.messages.len(),
            tool_count = chat_generation_command.tools.len(),
            input_token_count = inference_request.input_token_ids().len(),
            maximum_output_tokens = inference_request.max_output_tokens(),
            "prepared bounded chat thread for inference"
        );
        let request_output = performance_attribution
            .measure_operation(
                crate::PerformanceOperation::GenerationOutputDecoderInitialization,
                |_performance_attribution| {
                    Qwen3_5RequestOutput::new(
                        &self.tokenizer,
                        &chat_generation_command.tools,
                        request_enable_thinking,
                        chat_generation_command
                            .qwen_thinking_channel_seed
                            .as_deref(),
                    )
                },
            )
            .map_err(|validation_error| {
                tracing::warn!(
                    request_id = chat_generation_command.request_id.value(),
                    error = %validation_error,
                    "rejected chat thread during Qwen3.5 output preparation"
                );
                ChatGenerationFailureReason::invalid_request(validation_error.to_string())
            })?;
        Ok(PreparedModelGeneration::new(
            inference_request.with_performance_attribution(performance_attribution),
            request_output,
        ))
    }

    fn is_end_of_sequence_token(&self, generated_token_id: u32) -> bool {
        matches!(
            generated_token_id,
            token_id if token_id == self.tokenizer.end_of_text_token_id()
                || token_id == self.tokenizer.im_end_token_id()
        )
    }

    fn translate_generated_token(
        &self,
        request_output: &mut Self::RequestOutput,
        generated_token_id: u32,
    ) -> Result<ModelGeneratedTokenTranslation, ModelGenerationOutputError> {
        let output_events = request_output
            .push_token(generated_token_id)
            .map_err(translate_request_output_error)?;
        let translation = translate_output_events(&self.tokenizer, request_output, output_events)?;
        let (mut public_outputs, model_feedback_token_ids) = translation.into_parts();
        prepend_seeded_reasoning(request_output, &mut public_outputs);
        Ok(ModelGeneratedTokenTranslation::new(
            public_outputs,
            model_feedback_token_ids,
        ))
    }

    fn finish_request_output(
        &self,
        request_output: &mut Self::RequestOutput,
    ) -> Result<Vec<ChatGenerationOutput>, ModelGenerationOutputError> {
        let output_events = request_output
            .finish()
            .map_err(translate_request_output_error)?;
        let (mut public_outputs, _model_feedback_token_ids) =
            translate_output_events(&self.tokenizer, request_output, output_events)?.into_parts();
        prepend_seeded_reasoning(request_output, &mut public_outputs);
        Ok(public_outputs)
    }
}

/// Resolves per-request thinking mode without changing the model capability flag.
#[must_use]
pub const fn qwen3_5_request_enables_thinking(
    model_supports_thinking: bool,
    thinking_budget: Option<u16>,
) -> bool {
    if !model_supports_thinking {
        return false;
    }
    match thinking_budget {
        Some(0) => false,
        Some(_) | None => true,
    }
}

/// Preserves model-native context overflow as a typed public failure signal.
pub fn translate_qwen3_5_preparation_error(
    tokenizer_error: Qwen3_5TokenizerError,
) -> ChatGenerationFailureReason {
    match tokenizer_error {
        Qwen3_5TokenizerError::TotalContextTooLarge {
            actual_total_context_tokens,
            maximum_total_context_tokens,
        } => ChatGenerationFailureReason::ContextLengthExceeded {
            actual_total_context_tokens: u32::try_from(actual_total_context_tokens)
                .unwrap_or(u32::MAX),
            maximum_context_tokens: u32::try_from(maximum_total_context_tokens).unwrap_or(u32::MAX),
        },
        other_tokenizer_error => {
            ChatGenerationFailureReason::invalid_request(other_tokenizer_error.to_string())
        }
    }
}

fn translate_output_events(
    tokenizer: &Qwen3_5Tokenizer,
    request_output: &mut Qwen3_5RequestOutput,
    output_events: Vec<Qwen3_5OutputEvent>,
) -> Result<ModelGeneratedTokenTranslation, ModelGenerationOutputError> {
    let mut public_outputs = Vec::new();
    let mut model_feedback_token_ids = Vec::new();

    for output_event in output_events {
        match output_event {
            Qwen3_5OutputEvent::ReasoningDelta(text) => {
                public_outputs.push(ChatGenerationOutput::Reasoning { text });
            }
            Qwen3_5OutputEvent::TextDelta(text) => {
                public_outputs.push(ChatGenerationOutput::Text { text });
            }
            Qwen3_5OutputEvent::ToolCall(tool_call) => {
                public_outputs.push(ChatGenerationOutput::ToolCall {
                    tool_call_index: tool_call.index,
                    function_name: tool_call.function_name,
                    arguments_json: tool_call.arguments_json,
                });
            }
            Qwen3_5OutputEvent::ModelVisibleCorrection { correction_text } => {
                let enable_thinking = request_output.enable_thinking();
                let correction_token_ids = tokenizer
                    .encode_model_visible_correction(
                        &correction_text,
                        enable_thinking,
                        request_output.thinking_channel_seed(),
                    )
                    .map_err(|tokenizer_error| ModelGenerationOutputError::Fatal {
                        reason: tokenizer_error.to_string(),
                    })?;
                model_feedback_token_ids.extend(correction_token_ids);
                request_output.reset_after_model_visible_correction(enable_thinking);
                if let Some(seeded_reasoning) = request_output.take_seeded_reasoning_output() {
                    public_outputs.push(seeded_reasoning);
                }
            }
        }
    }

    Ok(ModelGeneratedTokenTranslation::new(
        public_outputs,
        model_feedback_token_ids,
    ))
}

fn prepend_seeded_reasoning(
    request_output: &mut Qwen3_5RequestOutput,
    public_outputs: &mut Vec<ChatGenerationOutput>,
) {
    // The seed is assistant reasoning already present in the prompt, so clients must observe it
    // before the model-generated continuation even when the first generated token is terminal.
    if let Some(seeded_reasoning) = request_output.take_seeded_reasoning_output() {
        public_outputs.insert(0, seeded_reasoning);
    }
}

/// Maps request-output failures to model-neutral output errors.
///
/// Tokenizer failures during generation (such as an out-of-vocabulary token ID) indicate
/// a broken engine that must terminate the worker, not just the current request. Parser
/// failures indicate the model produced malformed structured output, which leaves the
/// engine reusable for the next request.
pub fn translate_request_output_error(
    request_output_error: Qwen3_5RequestOutputError,
) -> ModelGenerationOutputError {
    match request_output_error {
        Qwen3_5RequestOutputError::Tokenizer(tokenizer_error) => {
            ModelGenerationOutputError::Fatal {
                reason: tokenizer_error.to_string(),
            }
        }
        Qwen3_5RequestOutputError::Parser { diagnostic, .. } => {
            ModelGenerationOutputError::MalformedOutput { diagnostic }
        }
    }
}
