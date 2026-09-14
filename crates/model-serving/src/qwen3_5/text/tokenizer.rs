use std::sync::Arc;

use astronomical_config::resolve_model_id;
use astronomical_ipc_protocol::ChatGenerationCommand;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::{PerformanceAttribution, PerformanceOperation, Qwen3_5InferenceRequest};

use super::{Qwen3_5ImageProcessor, Qwen3_5PromptRenderer, ValidatedQwen3_5Artifact};

use super::context_token_validation::validate_context_token_count;
use super::sampler_config::{Qwen3_5SamplerConfig, discover_sampler_config};
use super::token_decoder::Qwen3_5TokenDecoder;
use super::token_ids::{Qwen3_5TokenIds, discover_token_ids};
use super::tokenizer_error::Qwen3_5TokenizerError;

const THINKING_BUDGET_TRANSITION_TEXT: &str = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n";

/// The validated pinned tokenizer used by Qwen3.5 prompt and output processing.
#[derive(Clone, Debug)]
pub struct Qwen3_5Tokenizer {
    tokenizer: Arc<Tokenizer>,
    tokenizer_vocabulary_size: u32,
    model_vocabulary_size: u32,
    maximum_position_count: u32,
    advertised_context_window: u32,
    model_sampler_config: Qwen3_5SamplerConfig,
    token_ids: Qwen3_5TokenIds,
    model_id: String,
    image_processor: Option<Qwen3_5ImageProcessor>,
    forced_thinking_transition_token_ids: Vec<u32>,
    natural_reasoning_end_token_ids: Vec<u32>,
}

impl Qwen3_5Tokenizer {
    /// Digests the canonical token-to-identifier mapping while ignoring JSON
    /// formatting and tokenizer metadata unused by draft-model execution.
    pub fn token_identifier_mapping_digest(
        tokenizer_bytes: &[u8],
    ) -> Result<[u8; 32], Qwen3_5TokenizerError> {
        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|source| Qwen3_5TokenizerError::LoadTokenizer { source })?;
        let mut token_identifier_entries =
            tokenizer.get_vocab(true).into_iter().collect::<Vec<_>>();
        token_identifier_entries.sort_unstable_by(
            |(left_token_content, left_token_identifier),
             (right_token_content, right_token_identifier)| {
                left_token_content
                    .cmp(right_token_content)
                    .then_with(|| left_token_identifier.cmp(right_token_identifier))
            },
        );
        let mut token_identifier_mapping_hasher = Sha256::new();
        for (token_content, token_identifier) in token_identifier_entries {
            token_identifier_mapping_hasher.update((token_content.len() as u128).to_le_bytes());
            token_identifier_mapping_hasher.update(token_content.as_bytes());
            token_identifier_mapping_hasher.update(token_identifier.to_le_bytes());
        }
        Ok(token_identifier_mapping_hasher.finalize().into())
    }

    /// Loads tokenizer JSON bytes retained by artifact validation.
    pub fn from_json_bytes(
        tokenizer_bytes: &[u8],
        model_id: &str,
        model_vocabulary_size: u32,
        maximum_position_count: u32,
        image_processor: Qwen3_5ImageProcessor,
    ) -> Result<Self, Qwen3_5TokenizerError> {
        Self::from_json_bytes_with_optional_image_processor(
            tokenizer_bytes,
            model_id,
            model_vocabulary_size,
            maximum_position_count,
            maximum_position_count,
            Some(image_processor),
        )
    }

    fn from_json_bytes_with_optional_image_processor(
        tokenizer_bytes: &[u8],
        model_id: &str,
        model_vocabulary_size: u32,
        maximum_position_count: u32,
        advertised_context_window: u32,
        image_processor: Option<Qwen3_5ImageProcessor>,
    ) -> Result<Self, Qwen3_5TokenizerError> {
        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|source| Qwen3_5TokenizerError::LoadTokenizer { source })?;
        let token_ids = discover_token_ids(&tokenizer)
            .map_err(|source| Qwen3_5TokenizerError::DiscoverTokenIds { source })?;
        let tokenizer_vocabulary_size =
            u32::try_from(tokenizer.get_vocab_size(true)).map_err(|_| {
                Qwen3_5TokenizerError::TokenizerVocabularyTooLarge {
                    actual_vocabulary_size: tokenizer.get_vocab_size(true),
                }
            })?;
        for (token_content, expected_token_id) in token_ids.validation_pairs() {
            super::token_ids::validate_token_identity(
                &tokenizer,
                token_content,
                expected_token_id,
            )?;
        }
        let mut forced_thinking_transition_token_ids = tokenizer
            .encode(THINKING_BUDGET_TRANSITION_TEXT, false)
            .map_err(|source| Qwen3_5TokenizerError::EncodeThinkingBudgetTransition { source })?
            .get_ids()
            .to_vec();
        forced_thinking_transition_token_ids.push(token_ids.think_end_token_id);
        let natural_reasoning_end_token_ids = vec![
            token_ids.think_end_token_id,
            token_ids.tool_call_start_token_id,
        ];
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            tokenizer_vocabulary_size,
            model_vocabulary_size,
            maximum_position_count,
            advertised_context_window,
            model_sampler_config: discover_sampler_config(None),
            token_ids,
            model_id: model_id.to_owned(),
            image_processor,
            forced_thinking_transition_token_ids,
            natural_reasoning_end_token_ids,
        })
    }

    /// Loads only tokenizer bytes retained by the complete validated artifact.
    pub fn from_validated_artifact(
        validated_artifact: &ValidatedQwen3_5Artifact,
    ) -> Result<Self, Qwen3_5TokenizerError> {
        Self::from_validated_artifact_with_maximum_context_tokens(
            validated_artifact,
            validated_artifact.config().maximum_position_count(),
        )
    }

    /// `maximum_context_tokens` is the operator-configured, advertised context
    /// window; the hard rejection boundary always stays at the artifact's
    /// native maximum position count so configured softness never exceeds the
    /// model's real capability.
    pub(crate) fn from_validated_artifact_with_maximum_context_tokens(
        validated_artifact: &ValidatedQwen3_5Artifact,
        maximum_context_tokens: u32,
    ) -> Result<Self, Qwen3_5TokenizerError> {
        let tokenizer_bytes = validated_artifact
            .tokenizer_bytes()
            .ok_or(Qwen3_5TokenizerError::MissingValidatedTokenizer)?;
        let image_processor = validated_artifact
            .vision_config()
            .map(Qwen3_5ImageProcessor::from_vision_config);
        let mut tokenizer = Self::from_json_bytes_with_optional_image_processor(
            tokenizer_bytes,
            validated_artifact.model_id(),
            validated_artifact.config().vocabulary_size(),
            validated_artifact.config().maximum_position_count(),
            maximum_context_tokens,
            image_processor,
        )?;
        tokenizer.model_sampler_config =
            discover_sampler_config(validated_artifact.generation_config_bytes());
        Ok(tokenizer)
    }

    #[must_use]
    pub const fn model_sampler_config(&self) -> &Qwen3_5SamplerConfig {
        &self.model_sampler_config
    }

    #[must_use]
    pub const fn tokenizer_vocabulary_size(&self) -> u32 {
        self.tokenizer_vocabulary_size
    }

    #[must_use]
    pub const fn model_vocabulary_size(&self) -> u32 {
        self.model_vocabulary_size
    }

    #[must_use]
    pub const fn end_of_text_token_id(&self) -> u32 {
        self.token_ids.end_of_text_token_id
    }

    #[must_use]
    pub const fn im_start_token_id(&self) -> u32 {
        self.token_ids.im_start_token_id
    }

    #[must_use]
    pub const fn im_end_token_id(&self) -> u32 {
        self.token_ids.im_end_token_id
    }

    pub(crate) fn vocabulary_pieces(&self) -> Vec<String> {
        let mut vocabulary_pieces = vec![String::new(); self.model_vocabulary_size as usize];
        for (token_content, token_id) in self.tokenizer.get_vocab(true) {
            if let Some(vocabulary_piece) = vocabulary_pieces.get_mut(token_id as usize) {
                *vocabulary_piece = token_content;
            }
        }
        vocabulary_pieces
    }

    fn encoded_choice_token_sequences(
        &self,
        structured_generation: &astronomical_ipc_protocol::StructuredGenerationConstraint,
    ) -> Vec<Vec<u32>> {
        let astronomical_ipc_protocol::StructuredGenerationConstraint::Choice { choices } =
            structured_generation
        else {
            return Vec::new();
        };
        let mut encoded_choice_sequences = Vec::new();
        for choice in choices {
            for choice_text in [choice.clone(), format!(" {choice}")] {
                let Ok(encoding) = self.tokenizer.encode(choice_text, false) else {
                    continue;
                };
                let token_ids = encoding.get_ids().to_vec();
                if token_ids.is_empty() {
                    continue;
                }
                if !encoded_choice_sequences
                    .iter()
                    .any(|existing_token_ids| existing_token_ids == &token_ids)
                {
                    encoded_choice_sequences.push(token_ids);
                }
            }
        }
        encoded_choice_sequences
    }

    #[must_use]
    pub const fn think_start_token_id(&self) -> u32 {
        self.token_ids.think_start_token_id
    }

    #[must_use]
    pub const fn think_end_token_id(&self) -> u32 {
        self.token_ids.think_end_token_id
    }

    /// Returns the complete model-owned sequence used to leave bounded reasoning.
    #[must_use]
    pub fn forced_thinking_transition_token_ids(&self) -> &[u32] {
        &self.forced_thinking_transition_token_ids
    }

    /// Returns explicit and implicit token boundaries that end reasoning.
    #[must_use]
    pub fn natural_reasoning_end_token_ids(&self) -> &[u32] {
        &self.natural_reasoning_end_token_ids
    }

    #[must_use]
    pub const fn image_pad_token_id(&self) -> u32 {
        self.token_ids.image_pad_token_id
    }

    #[must_use]
    pub const fn tool_call_start_token_id(&self) -> u32 {
        self.token_ids.tool_call_start_token_id
    }

    #[must_use]
    pub const fn tool_call_end_token_id(&self) -> u32 {
        self.token_ids.tool_call_end_token_id
    }

    #[must_use]
    pub const fn tool_response_start_token_id(&self) -> u32 {
        self.token_ids.tool_response_start_token_id
    }

    #[must_use]
    pub const fn tool_response_end_token_id(&self) -> u32 {
        self.token_ids.tool_response_end_token_id
    }

    /// Encodes one fixed-template prompt under the initial coding-context bounds.
    pub fn encode_prompt(&self, rendered_prompt: &str) -> Result<Vec<u32>, Qwen3_5TokenizerError> {
        let encoding = self
            .tokenizer
            .encode(rendered_prompt, false)
            .map_err(|source| Qwen3_5TokenizerError::EncodePrompt { source })?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Encodes server-generated feedback that is injected into the active model context.
    /// Creates one bounded request-local monotonic UTF-8 decoder.
    #[must_use]
    pub fn incremental_decoder(&self) -> Qwen3_5TokenDecoder {
        Qwen3_5TokenDecoder::new(
            Arc::clone(&self.tokenizer),
            self.model_vocabulary_size,
            self.token_ids,
        )
    }

    /// Validates, renders, and tokenizes one structured chat request for native prefill.
    ///
    /// User messages carrying decoded images are preprocessed through the Qwen3VL image
    /// processor so that the correct number of `<|video_pad|>` tokens appear in the
    /// rendered prompt.
    pub fn prepare_chat(
        &self,
        chat_generation_command: &ChatGenerationCommand,
        enable_thinking: bool,
    ) -> Result<Qwen3_5InferenceRequest, Qwen3_5TokenizerError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        self.prepare_chat_with_performance_attribution(
            chat_generation_command,
            enable_thinking,
            &mut disabled_performance_attribution,
        )
    }

    pub(crate) fn prepare_chat_with_performance_attribution(
        &self,
        chat_generation_command: &ChatGenerationCommand,
        enable_thinking: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5InferenceRequest, Qwen3_5TokenizerError> {
        performance_attribution
            .measure_operation(
                PerformanceOperation::ChatCommandValidation,
                |_performance_attribution| chat_generation_command.validate(),
            )
            .map_err(Qwen3_5TokenizerError::InvalidChatCommand)?;
        // Resolve the requested model ID against the loaded model's leaf-only ID.
        // Clients may send "org/model-name" (e.g. "astronomical/fake-mixture-of-experts")
        // while internally the model ID is just the leaf name (e.g. "fake-mixture-of-experts").
        let resolved_requested_model_id =
            resolve_model_id(&chat_generation_command.model, &[self.model_id.as_str()]);
        if resolved_requested_model_id != self.model_id {
            return Err(Qwen3_5TokenizerError::ModelIdMismatch {
                actual_model_id: chat_generation_command.model.clone(),
            });
        }
        let effective_thinking_allowance =
            super::thinking_allowance::resolve_effective_thinking_allowance(
                chat_generation_command.settings.thinking_budget,
                enable_thinking,
                chat_generation_command.settings.max_output_tokens,
                self.forced_thinking_transition_token_ids.len(),
            );
        if let Some(effective_allowance) = effective_thinking_allowance
            && chat_generation_command
                .settings
                .thinking_budget
                .is_some_and(|budget| enable_thinking && budget > 0)
        {
            let minimum_bounded_output_tokens =
                super::thinking_allowance::effective_thinking_reservation_token_count(
                    effective_thinking_allowance,
                    self.forced_thinking_transition_token_ids.len(),
                )
                .unwrap_or(usize::MAX);
            if usize::from(chat_generation_command.settings.max_output_tokens)
                < minimum_bounded_output_tokens
            {
                return Err(Qwen3_5TokenizerError::ThinkingBudgetOutputReservation {
                    max_output_tokens: chat_generation_command.settings.max_output_tokens,
                    thinking_budget: effective_allowance,
                    transition_token_count: self.forced_thinking_transition_token_ids.len(),
                });
            }
            if Some(effective_allowance) != chat_generation_command.settings.thinking_budget {
                tracing::warn!(
                    request_id = chat_generation_command.request_id.value(),
                    requested_thinking_budget = chat_generation_command
                        .settings
                        .thinking_budget
                        .unwrap_or(0),
                    effective_thinking_budget = effective_allowance,
                    max_output_tokens = chat_generation_command.settings.max_output_tokens,
                    "clamped the thinking allowance to fit the requested output budget"
                );
            }
        }
        let prepared_chat_images = performance_attribution.measure_operation(
            PerformanceOperation::ImagePreprocessing,
            |_performance_attribution| {
                super::chat_images::prepare_chat_images(
                    &chat_generation_command.messages,
                    self.image_processor.as_ref(),
                )
            },
        )?;
        let rendered_prompt = performance_attribution
            .measure_operation(
                PerformanceOperation::PromptRendering,
                |_performance_attribution| {
                    Qwen3_5PromptRenderer::render_with_control_span(
                        &chat_generation_command.messages,
                        &chat_generation_command.tools,
                        enable_thinking,
                        &prepared_chat_images.image_token_counts_per_user_message,
                        chat_generation_command
                            .qwen_thinking_channel_seed
                            .as_deref(),
                    )
                },
            )
            .map_err(Qwen3_5TokenizerError::RenderPrompt)?;
        let (input_token_ids, ordinary_target_prefill_control_span_token_count) =
            performance_attribution.measure_operation(
                PerformanceOperation::PromptTokenization,
                |_performance_attribution| {
                    self.encode_rendered_prompt_with_control_span(&rendered_prompt)
                },
            )?;
        validate_context_token_count(
            input_token_ids.len(),
            usize::from(chat_generation_command.settings.max_output_tokens),
            self.maximum_position_count as usize,
            self.advertised_context_window as usize,
        )?;
        let model_top_k = match u16::try_from(self.model_sampler_config.model_top_k) {
            Ok(model_top_k) if model_top_k > 0 => model_top_k,
            _ => 20,
        };
        let mut inference_request = Qwen3_5InferenceRequest::new_sampling_with_top_k(
            chat_generation_command.request_id,
            input_token_ids,
            chat_generation_command.settings.max_output_tokens,
            chat_generation_command
                .settings
                .temperature_thousandths
                .unwrap_or(self.model_sampler_config.temperature_thousandths),
            model_top_k,
            chat_generation_command
                .settings
                .top_p_thousandths
                .unwrap_or(self.model_sampler_config.top_p_thousandths),
            chat_generation_command.settings.seed,
        )
        .with_ordinary_target_prefill_control_span_token_count(
            ordinary_target_prefill_control_span_token_count,
        )
        .with_image_pad_token_id(self.image_pad_token_id())
        .with_thinking_configuration(
            enable_thinking,
            effective_thinking_allowance,
            self.forced_thinking_transition_token_ids.clone(),
            self.natural_reasoning_end_token_ids.clone(),
        );
        if let Some(structured_generation) = &chat_generation_command.structured_generation {
            let structured_token_constraint =
                crate::structured_generation::StructuredTokenConstraint::compile(
                    structured_generation,
                    self.vocabulary_pieces(),
                    vec![self.end_of_text_token_id(), self.im_end_token_id()],
                    self.encoded_choice_token_sequences(structured_generation),
                )
                .map_err(|compile_reason| {
                    Qwen3_5TokenizerError::StructuredConstraintCompile {
                        reason: compile_reason,
                    }
                })?;
            inference_request =
                inference_request.with_structured_generation(structured_token_constraint);
        }
        if !prepared_chat_images.processed_visual_images.is_empty() {
            inference_request = inference_request
                .with_processed_visual_images(prepared_chat_images.processed_visual_images);
        }
        Ok(inference_request)
    }

    /// Encodes one rendered prompt while converting its system-and-tool byte
    /// boundary into the exact corresponding token count.
    pub fn encode_rendered_prompt_with_control_span(
        &self,
        rendered_prompt: &super::Qwen3_5RenderedPrompt,
    ) -> Result<(Vec<u32>, usize), Qwen3_5TokenizerError> {
        let encoding = self
            .tokenizer
            .encode(rendered_prompt.as_str(), false)
            .map_err(|source| Qwen3_5TokenizerError::EncodePrompt { source })?;
        super::prompt::ordinary_target_prefill_control_span_token_count(
            &encoding,
            rendered_prompt.ordinary_target_prefill_control_span_byte_count(),
        )
    }
}
