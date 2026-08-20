use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationFailureReason, ChatGenerationOutput, ChatMessage,
    ChatModelCapabilities, ChatToolChoice, MtpDepthStatus, MtpRuntimeState,
    SpeculativePrefillRuntimeState, WorkerEvent,
};

use crate::{
    ModelGeneratedTokenTranslation, ModelGenerationOutputError, ModelGenerationProcessor,
    PerformanceAttribution, PerformanceOperation, PreparedModelGeneration,
};

use super::{
    LagunaInferenceRequest, LagunaOutputEvent, LagunaPreparationError, LagunaPreparedGeneration,
    LagunaPromptRenderer, LagunaRequestOutput, LagunaRequestOutputError,
    LagunaTextArtifactDescriptor, LagunaTokenizer, LagunaTokenizerError,
};

/// Laguna-specific preparation and output translation, intentionally unconnected to dispatch.
#[derive(Clone, Debug)]
pub struct LagunaGenerationProcessor {
    model_id: String,
    descriptor: LagunaTextArtifactDescriptor,
    tokenizer: LagunaTokenizer,
    maximum_context_tokens: u32,
    maximum_output_tokens: u32,
    performance_attribution_enabled: bool,
}

impl LagunaGenerationProcessor {
    /// Creates a processor with disabled attribution for standalone text qualification.
    pub fn new(
        model_id: impl Into<String>,
        descriptor: LagunaTextArtifactDescriptor,
    ) -> Result<Self, LagunaTokenizerError> {
        let maximum_context_tokens = descriptor.maximum_context_tokens();
        let maximum_output_tokens = descriptor
            .sampler_config()
            .maximum_new_tokens()
            .unwrap_or(u32::from(u16::MAX))
            .min(maximum_context_tokens.saturating_sub(1));
        Self::new_with_performance_attribution(
            model_id,
            descriptor,
            maximum_context_tokens,
            maximum_output_tokens,
            false,
        )
    }

    /// Creates a processor whose request-local attribution can be enabled by runtime config.
    pub fn new_with_performance_attribution(
        model_id: impl Into<String>,
        descriptor: LagunaTextArtifactDescriptor,
        maximum_context_tokens: u32,
        maximum_output_tokens: u32,
        performance_attribution_enabled: bool,
    ) -> Result<Self, LagunaTokenizerError> {
        let tokenizer = LagunaTokenizer::from_descriptor(&descriptor)?;
        Ok(Self {
            model_id: model_id.into(),
            descriptor,
            tokenizer,
            maximum_context_tokens,
            maximum_output_tokens,
            performance_attribution_enabled,
        })
    }

    /// Creates request-local decode state for generated token text.
    #[must_use]
    pub fn incremental_decoder(&self) -> super::LagunaTokenDecoder {
        self.tokenizer.incremental_decoder()
    }

    /// Prepares the complete public Laguna journey without entering runtime dispatch.
    pub fn prepare_chat(
        &self,
        chat_command: &ChatGenerationCommand,
    ) -> Result<LagunaPreparedGeneration, LagunaPreparationError> {
        self.prepare_chat_and_output(chat_command)
            .map(|(prepared_generation, _request_output)| prepared_generation)
    }

    fn prepare_chat_and_output(
        &self,
        chat_command: &ChatGenerationCommand,
    ) -> Result<(LagunaPreparedGeneration, LagunaRequestOutput), LagunaPreparationError> {
        let mut performance_attribution = if self.performance_attribution_enabled {
            PerformanceAttribution::enabled()
        } else {
            PerformanceAttribution::disabled()
        };
        performance_attribution.measure_operation(
            PerformanceOperation::ChatCommandValidation,
            |_performance_attribution| self.validate_command(chat_command),
        )?;
        let thinking_enabled = resolve_thinking_enabled(
            chat_command.settings.thinking_budget,
            self.descriptor.generation_default_thinking_enabled(),
            self.descriptor.default_thinking_enabled(),
        );
        let effective_tools = match &chat_command.tool_choice {
            ChatToolChoice::None => &[][..],
            ChatToolChoice::Auto | ChatToolChoice::Required | ChatToolChoice::Function { .. } => {
                chat_command.tools.as_slice()
            }
        };
        let rendered_prompt = performance_attribution
            .measure_operation(
                PerformanceOperation::PromptRendering,
                |_performance_attribution| {
                    LagunaPromptRenderer::new(&self.descriptor).render(
                        &chat_command.messages,
                        effective_tools,
                        thinking_enabled,
                    )
                },
            )
            .map_err(LagunaPreparationError::PromptRendering)?;
        let prompt_token_ids = performance_attribution
            .measure_operation(
                PerformanceOperation::PromptTokenization,
                |_performance_attribution| self.tokenizer.encode_prompt(&rendered_prompt),
            )
            .map_err(LagunaPreparationError::PromptTokenization)?;
        validate_context_length(
            prompt_token_ids.len(),
            chat_command.settings.max_output_tokens,
            self.maximum_context_tokens,
        )?;
        let sampler_config = self.descriptor.sampler_config().with_request_overrides(
            chat_command.settings.temperature_thousandths,
            chat_command.settings.top_p_thousandths,
            chat_command.settings.seed,
        );
        let request_output = performance_attribution
            .measure_operation(
                PerformanceOperation::GenerationOutputDecoderInitialization,
                |_performance_attribution| {
                    LagunaRequestOutput::new(
                        &self.descriptor,
                        &self.tokenizer,
                        effective_tools,
                        thinking_enabled,
                    )
                },
            )
            .map_err(|request_output_error| match request_output_error {
                LagunaRequestOutputError::Parser { source, .. } => {
                    LagunaPreparationError::OutputParserInitialization(source)
                }
                LagunaRequestOutputError::Tokenizer(source) => {
                    LagunaPreparationError::PromptTokenization(source)
                }
            })?;
        let thinking_budget = chat_command
            .settings
            .thinking_budget
            .filter(|thinking_budget| *thinking_budget > 0 && thinking_enabled);
        let inference_request = LagunaInferenceRequest::new(
            chat_command.request_id,
            prompt_token_ids,
            chat_command.settings.max_output_tokens,
            sampler_config.clone(),
            thinking_enabled,
            thinking_budget,
            performance_attribution,
        );
        let prepared_generation = LagunaPreparedGeneration::new(
            inference_request,
            rendered_prompt,
            sampler_config,
            thinking_enabled,
            thinking_budget,
            self.descriptor.clone(),
            effective_tools.to_vec(),
        );
        Ok((prepared_generation, request_output))
    }

    fn validate_command(
        &self,
        chat_command: &ChatGenerationCommand,
    ) -> Result<(), LagunaPreparationError> {
        // Identity and image capability are checked before any user content is rendered.
        if chat_command.model != self.model_id {
            return Err(LagunaPreparationError::ModelIdMismatch {
                expected_model_id: self.model_id.clone(),
                actual_model_id: chat_command.model.clone(),
            });
        }
        if u32::from(chat_command.settings.max_output_tokens) > self.maximum_output_tokens {
            return Err(LagunaPreparationError::MaximumOutputTokensExceeded {
                requested_output_tokens: chat_command.settings.max_output_tokens,
                maximum_output_tokens: self.maximum_output_tokens,
            });
        }
        if chat_command.messages.iter().any(
            |message| matches!(message, ChatMessage::User { images, .. } if !images.is_empty()),
        ) {
            return Err(LagunaPreparationError::ImageInputUnsupported);
        }
        chat_command
            .validate()
            .map_err(LagunaPreparationError::InvalidChatCommand)
    }
}

impl ModelGenerationProcessor for LagunaGenerationProcessor {
    type InferenceRequest = LagunaInferenceRequest;
    type RequestOutput = LagunaRequestOutput;

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
        let context_window = self.maximum_context_tokens;
        let max_output_tokens = self.maximum_output_tokens;
        WorkerEvent::Ready {
            model_id: self.model_id.clone(),
            capabilities: ChatModelCapabilities {
                supports_reasoning: true,
                supports_tool_calls: true,
                has_vision: false,
                max_input_tokens: context_window.saturating_sub(1),
                max_output_tokens,
                context_window,
            },
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
        let (prepared_generation, request_output) = self
            .prepare_chat_and_output(chat_generation_command)
            .map_err(translate_preparation_error)?;
        Ok(PreparedModelGeneration::new(
            prepared_generation.into_inference_request(),
            request_output,
        ))
    }

    fn is_end_of_sequence_token(&self, generated_token_id: u32) -> bool {
        self.descriptor.is_end_token(generated_token_id)
    }

    fn translate_generated_token(
        &self,
        request_output: &mut Self::RequestOutput,
        generated_token_id: u32,
    ) -> Result<ModelGeneratedTokenTranslation, ModelGenerationOutputError> {
        let output_events = request_output
            .push_token(generated_token_id)
            .map_err(translate_request_output_error)?;
        Ok(ModelGeneratedTokenTranslation::from_outputs(
            translate_output_events(output_events),
        ))
    }

    fn finish_request_output(
        &self,
        request_output: &mut Self::RequestOutput,
    ) -> Result<Vec<ChatGenerationOutput>, ModelGenerationOutputError> {
        request_output
            .finish()
            .map(translate_output_events)
            .map_err(translate_request_output_error)
    }
}

fn resolve_thinking_enabled(
    thinking_budget: Option<u16>,
    generation_default: Option<bool>,
    template_default: bool,
) -> bool {
    match thinking_budget {
        Some(0) => false,
        Some(_) => true,
        None => generation_default.unwrap_or(template_default),
    }
}

fn validate_context_length(
    prompt_token_count: usize,
    maximum_output_tokens: u16,
    maximum_context_tokens: u32,
) -> Result<(), LagunaPreparationError> {
    let actual_context_tokens = prompt_token_count
        .checked_add(usize::from(maximum_output_tokens))
        .unwrap_or(usize::MAX);
    let maximum_context_token_count = usize::try_from(maximum_context_tokens).unwrap_or(usize::MAX);
    if actual_context_tokens > maximum_context_token_count {
        return Err(LagunaPreparationError::ContextLengthExceeded {
            actual_context_tokens,
            maximum_context_tokens,
        });
    }
    Ok(())
}

fn translate_preparation_error(
    preparation_error: LagunaPreparationError,
) -> ChatGenerationFailureReason {
    match preparation_error {
        LagunaPreparationError::ContextLengthExceeded {
            actual_context_tokens,
            maximum_context_tokens,
        } => ChatGenerationFailureReason::ContextLengthExceeded {
            actual_total_context_tokens: u32::try_from(actual_context_tokens).unwrap_or(u32::MAX),
            maximum_context_tokens,
        },
        other_error => ChatGenerationFailureReason::invalid_request(other_error.to_string()),
    }
}

fn translate_request_output_error(
    request_output_error: LagunaRequestOutputError,
) -> ModelGenerationOutputError {
    match request_output_error {
        LagunaRequestOutputError::Tokenizer(tokenizer_error) => ModelGenerationOutputError::Fatal {
            reason: tokenizer_error.to_string(),
        },
        LagunaRequestOutputError::Parser { diagnostic, .. } => {
            ModelGenerationOutputError::MalformedOutput { diagnostic }
        }
    }
}

fn translate_output_events(output_events: Vec<LagunaOutputEvent>) -> Vec<ChatGenerationOutput> {
    output_events
        .into_iter()
        .map(|output_event| match output_event {
            LagunaOutputEvent::ReasoningDelta(text) => ChatGenerationOutput::Reasoning { text },
            LagunaOutputEvent::TextDelta(text) => ChatGenerationOutput::Text { text },
            LagunaOutputEvent::ToolCall {
                index,
                function_name,
                arguments_json,
            } => ChatGenerationOutput::ToolCall {
                tool_call_index: index,
                function_name,
                arguments_json,
            },
        })
        .collect()
}
