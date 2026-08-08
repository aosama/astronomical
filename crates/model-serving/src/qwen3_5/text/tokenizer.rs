use std::sync::Arc;

use astronomical_config::resolve_model_id;
use astronomical_ipc_protocol::{ChatGenerationCommand, ChatMessage};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::{PerformanceAttribution, PerformanceOperation, Qwen3_5InferenceRequest};

use super::{
    Qwen3_5ImageProcessor, Qwen3_5ProcessedImage, Qwen3_5PromptRenderer, ValidatedQwen3_5Artifact,
};

use super::token_ids::{Qwen3_5TokenIds, discover_token_ids};
use super::tokenizer_error::Qwen3_5TokenizerError;

/// The validated pinned tokenizer used by Qwen3.5 prompt and output processing.
#[derive(Clone, Debug)]
pub struct Qwen3_5Tokenizer {
    tokenizer: Arc<Tokenizer>,
    tokenizer_vocabulary_size: u32,
    model_vocabulary_size: u32,
    maximum_position_count: u32,
    token_ids: Qwen3_5TokenIds,
    model_id: String,
    image_processor: Option<Qwen3_5ImageProcessor>,
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

    /// Loads and certifies tokenizer JSON bytes retained by artifact validation.
    pub fn from_json_bytes(
        tokenizer_bytes: &[u8],
        model_vocabulary_size: u32,
        maximum_position_count: u32,
        image_processor: Qwen3_5ImageProcessor,
    ) -> Result<Self, Qwen3_5TokenizerError> {
        Self::from_json_bytes_with_optional_image_processor(
            tokenizer_bytes,
            model_vocabulary_size,
            maximum_position_count,
            Some(image_processor),
        )
    }

    fn from_json_bytes_with_optional_image_processor(
        tokenizer_bytes: &[u8],
        model_vocabulary_size: u32,
        maximum_position_count: u32,
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
            validate_token_identity(&tokenizer, token_content, expected_token_id)?;
        }
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            tokenizer_vocabulary_size,
            model_vocabulary_size,
            maximum_position_count,
            token_ids,
            model_id: String::new(),
            image_processor,
        })
    }

    /// Loads only tokenizer bytes retained by the complete validated artifact.
    pub fn from_validated_artifact(
        validated_artifact: &ValidatedQwen3_5Artifact,
    ) -> Result<Self, Qwen3_5TokenizerError> {
        let tokenizer_bytes = validated_artifact
            .tokenizer_bytes()
            .ok_or(Qwen3_5TokenizerError::MissingValidatedTokenizer)?;
        let image_processor = validated_artifact
            .vision_config()
            .map(Qwen3_5ImageProcessor::from_vision_config);
        let mut tokenizer = Self::from_json_bytes_with_optional_image_processor(
            tokenizer_bytes,
            validated_artifact.config().vocabulary_size(),
            validated_artifact.config().maximum_position_count(),
            image_processor,
        )?;
        tokenizer.model_id = validated_artifact.model_id().to_owned();
        Ok(tokenizer)
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

    #[must_use]
    pub const fn think_start_token_id(&self) -> u32 {
        self.token_ids.think_start_token_id
    }

    #[must_use]
    pub const fn think_end_token_id(&self) -> u32 {
        self.token_ids.think_end_token_id
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
    pub fn encode_model_visible_correction(
        &self,
        correction_text: &str,
        enable_thinking: bool,
    ) -> Result<Vec<u32>, Qwen3_5TokenizerError> {
        let rendered_correction = Qwen3_5PromptRenderer::render_model_visible_correction(
            correction_text,
            enable_thinking,
        );
        self.encode_prompt(&rendered_correction)
    }

    /// Creates one bounded request-local monotonic UTF-8 decoder.
    #[must_use]
    pub fn incremental_decoder(&self) -> Qwen3_5TokenDecoder {
        Qwen3_5TokenDecoder {
            tokenizer: Arc::clone(&self.tokenizer),
            model_vocabulary_size: self.model_vocabulary_size,
            token_ids: self.token_ids,
            generated_token_ids: Vec::new(),
            emitted_text: String::new(),
            pending_token_ids: Vec::new(),
        }
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
        // Clients may send "org/model-name" (e.g. "mlx-community/Ornith-1.0-35B-OptiQ-4bit")
        // while internally the model ID is just the leaf name (e.g. "Ornith-1.0-35B-OptiQ-4bit").
        let resolved_requested_model_id =
            resolve_model_id(&chat_generation_command.model, &[self.model_id.as_str()]);
        if resolved_requested_model_id != self.model_id {
            return Err(Qwen3_5TokenizerError::ModelIdMismatch {
                actual_model_id: chat_generation_command.model.clone(),
            });
        }
        let prepared_chat_images = performance_attribution.measure_operation(
            PerformanceOperation::ImagePreprocessing,
            |_performance_attribution| {
                prepare_chat_images(
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
        )?;
        let mut inference_request = Qwen3_5InferenceRequest::new_sampling(
            chat_generation_command.request_id,
            input_token_ids,
            chat_generation_command.settings.max_output_tokens,
            chat_generation_command
                .settings
                .temperature_thousandths
                .unwrap_or(1_000),
            chat_generation_command
                .settings
                .top_p_thousandths
                .unwrap_or(950),
            chat_generation_command.settings.seed,
        )
        .with_ordinary_target_prefill_control_span_token_count(
            ordinary_target_prefill_control_span_token_count,
        )
        .with_image_pad_token_id(self.image_pad_token_id());
        if let Some(thinking_budget) = chat_generation_command.settings.thinking_budget {
            inference_request = inference_request.with_thinking_budget(thinking_budget);
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
        let ordinary_target_prefill_control_span_byte_count =
            rendered_prompt.ordinary_target_prefill_control_span_byte_count();
        let mut ordinary_target_prefill_control_span_token_count = 0usize;
        for (token_start_byte_offset, token_end_byte_offset) in encoding.get_offsets() {
            if *token_start_byte_offset < ordinary_target_prefill_control_span_byte_count
                && *token_end_byte_offset > ordinary_target_prefill_control_span_byte_count
            {
                return Err(Qwen3_5TokenizerError::ControlSpanTokenBoundaryUnavailable);
            }
            if *token_end_byte_offset <= ordinary_target_prefill_control_span_byte_count
                && *token_start_byte_offset < ordinary_target_prefill_control_span_byte_count
            {
                ordinary_target_prefill_control_span_token_count =
                    ordinary_target_prefill_control_span_token_count.saturating_add(1);
            }
        }
        Ok((
            encoding.get_ids().to_vec(),
            ordinary_target_prefill_control_span_token_count,
        ))
    }
}

/// Validates combined input and output tokens against model-native positions.
pub fn validate_context_token_count(
    input_token_count: usize,
    maximum_output_tokens: usize,
    maximum_context_tokens: usize,
) -> Result<(), Qwen3_5TokenizerError> {
    let total_context_tokens = input_token_count.checked_add(maximum_output_tokens).ok_or(
        Qwen3_5TokenizerError::TotalContextTooLarge {
            actual_total_context_tokens: usize::MAX,
            maximum_total_context_tokens: maximum_context_tokens,
        },
    )?;
    if total_context_tokens > maximum_context_tokens {
        return Err(Qwen3_5TokenizerError::TotalContextTooLarge {
            actual_total_context_tokens: total_context_tokens,
            maximum_total_context_tokens: maximum_context_tokens,
        });
    }
    Ok(())
}

/// Request-local streaming decoder that emits only newly stable text suffixes.
///
/// Uses incremental single-token or small-batch decode calls instead of
/// re-decoding the full token prefix on every push, reducing decode complexity
/// from O(n^2) to O(n) for normal BPE byte-level tokenizers.
#[derive(Clone, Debug)]
pub struct Qwen3_5TokenDecoder {
    tokenizer: Arc<Tokenizer>,
    model_vocabulary_size: u32,
    token_ids: Qwen3_5TokenIds,
    generated_token_ids: Vec<u32>,
    emitted_text: String,
    pending_token_ids: Vec<u32>,
}

const MAX_PENDING_TOKENS: usize = 4;
const REPLACEMENT_CHARACTER: char = '\u{FFFD}';

impl Qwen3_5TokenDecoder {
    /// Decodes one generated token while suppressing both certified stop tokens.
    pub fn push_token(
        &mut self,
        generated_token_id: u32,
    ) -> Result<Option<String>, Qwen3_5TokenizerError> {
        if generated_token_id == self.token_ids.end_of_text_token_id
            || generated_token_id == self.token_ids.im_end_token_id
        {
            return Ok(None);
        }
        if generated_token_id >= self.model_vocabulary_size
            || self.tokenizer.id_to_token(generated_token_id).is_none()
        {
            return Err(Qwen3_5TokenizerError::GeneratedTokenOutOfVocabulary {
                generated_token_id,
                model_vocabulary_size: self.model_vocabulary_size,
            });
        }
        self.generated_token_ids.push(generated_token_id);
        self.pending_token_ids.push(generated_token_id);

        let candidate_text = self
            .tokenizer
            .decode(&self.pending_token_ids, true)
            .map_err(|source| Qwen3_5TokenizerError::DecodeGeneratedTokens { source })?;

        if !candidate_text.ends_with(REPLACEMENT_CHARACTER) {
            self.emitted_text.push_str(&candidate_text);
            self.pending_token_ids.clear();
            if candidate_text.is_empty() {
                return Ok(None);
            }
            return Ok(Some(candidate_text));
        }

        if self.pending_token_ids.len() >= MAX_PENDING_TOKENS {
            return self.flush_via_full_decode();
        }

        Ok(None)
    }

    /// Flushes a final incomplete byte-fallback sequence at request completion.
    pub fn finish(&mut self) -> Result<Option<String>, Qwen3_5TokenizerError> {
        if self.pending_token_ids.is_empty() {
            return Ok(None);
        }
        self.flush_via_full_decode()
    }

    fn flush_via_full_decode(&mut self) -> Result<Option<String>, Qwen3_5TokenizerError> {
        let full_decoded_text = self
            .tokenizer
            .decode(&self.generated_token_ids, true)
            .map_err(|source| Qwen3_5TokenizerError::DecodeGeneratedTokens { source })?;
        if !full_decoded_text.starts_with(&self.emitted_text) {
            return Err(Qwen3_5TokenizerError::DecodedTextRewrotePrefix);
        }
        let new_text_suffix = full_decoded_text[self.emitted_text.len()..].to_owned();
        self.emitted_text = full_decoded_text;
        self.pending_token_ids.clear();
        if new_text_suffix.is_empty() {
            return Ok(None);
        }
        Ok(Some(new_text_suffix))
    }

    pub(crate) fn generated_token_ids_for_diagnostics(&self) -> &[u32] {
        &self.generated_token_ids
    }

    pub(crate) fn pending_token_ids_for_diagnostics(&self) -> &[u32] {
        &self.pending_token_ids
    }

    pub(crate) fn emitted_text_for_diagnostics(&self) -> &str {
        &self.emitted_text
    }
}

/// Extracts one token-count vector per user message from the conversation history.
///
/// Each user message may carry zero or more decoded images. For every image,
/// the Qwen3VL image processor computes the number of `<|image_pad|>` tokens
/// after spatial merge. Text-only user messages produce an empty vector.
struct PreparedChatImages {
    image_token_counts_per_user_message: Vec<Vec<usize>>,
    processed_visual_images: Vec<Qwen3_5ProcessedImage>,
}

fn prepare_chat_images(
    messages: &[ChatMessage],
    image_processor: Option<&Qwen3_5ImageProcessor>,
) -> Result<PreparedChatImages, Qwen3_5TokenizerError> {
    let mut image_token_counts_per_user_message = Vec::new();
    let mut processed_visual_images = Vec::new();
    for message in messages {
        if let ChatMessage::User { images, .. } = message {
            let mut per_image_token_counts = Vec::with_capacity(images.len());
            for image_input in images {
                let image_processor =
                    image_processor.ok_or(Qwen3_5TokenizerError::ImageInputUnsupported)?;
                let processed_image = image_processor
                    .process_image_bytes(&image_input.decoded_bytes)
                    .map_err(Qwen3_5TokenizerError::ImageProcessing)?;
                let image_token_count_after_spatial_merge =
                    processed_image.image_token_count_after_spatial_merge;
                per_image_token_counts.push(image_token_count_after_spatial_merge);
                processed_visual_images.push(processed_image);
            }
            image_token_counts_per_user_message.push(per_image_token_counts);
        }
    }
    Ok(PreparedChatImages {
        image_token_counts_per_user_message,
        processed_visual_images,
    })
}

fn validate_token_identity(
    tokenizer: &Tokenizer,
    token_content: &'static str,
    expected_token_id: u32,
) -> Result<(), Qwen3_5TokenizerError> {
    let actual_token_id = tokenizer.token_to_id(token_content);
    if actual_token_id != Some(expected_token_id)
        || tokenizer.id_to_token(expected_token_id).as_deref() != Some(token_content)
    {
        return Err(Qwen3_5TokenizerError::SpecialTokenMismatch {
            token_content,
            expected_token_id,
            actual_token_id,
        });
    }
    Ok(())
}
