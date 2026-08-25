use astronomical_ipc_protocol::{ChatGenerationOutput, ChatToolDefinition};
use thiserror::Error;

use super::{
    Qwen3_5OutputEvent, Qwen3_5OutputParser, Qwen3_5OutputParserError, Qwen3_5TokenDecoder,
    Qwen3_5Tokenizer, Qwen3_5TokenizerError,
};
use crate::MalformedModelOutputDiagnostic;

/// Request-local owner that translates generated token IDs into neutral chat events.
#[derive(Debug)]
pub struct Qwen3_5RequestOutput {
    output_parser: Qwen3_5OutputParser,
    token_decoder: Qwen3_5TokenDecoder,
    enable_thinking: bool,
    thinking_channel_seed: Option<String>,
    should_emit_seeded_reasoning: bool,
}

impl Qwen3_5RequestOutput {
    /// Creates bounded decode/parser state from the exact request tool declarations.
    pub fn new(
        tokenizer: &Qwen3_5Tokenizer,
        declared_tools: &[ChatToolDefinition],
        enable_thinking: bool,
        thinking_channel_seed: Option<&str>,
    ) -> Result<Self, Qwen3_5RequestOutputError> {
        let output_parser = if enable_thinking {
            Qwen3_5OutputParser::new_after_thinking_prefix(declared_tools)
                .map_err(Self::parser_initialization_error)?
        } else {
            Qwen3_5OutputParser::new(declared_tools).map_err(Self::parser_initialization_error)?
        };
        let normalized_thinking_channel_seed = thinking_channel_seed
            .map(str::trim)
            .filter(|trimmed_seed| !trimmed_seed.is_empty())
            .map(str::to_owned);
        let should_emit_seeded_reasoning =
            enable_thinking && normalized_thinking_channel_seed.is_some();
        Ok(Self {
            output_parser,
            token_decoder: tokenizer.incremental_decoder(),
            enable_thinking,
            thinking_channel_seed: normalized_thinking_channel_seed,
            should_emit_seeded_reasoning,
        })
    }

    /// Returns whether this request's parser uses the reasoning-prefixed state.
    #[must_use]
    pub const fn enable_thinking(&self) -> bool {
        self.enable_thinking
    }

    /// Returns the normalized seed retained for later correction-driven reasoning blocks.
    #[must_use]
    pub fn thinking_channel_seed(&self) -> Option<&str> {
        self.thinking_channel_seed.as_deref()
    }

    /// Emits seeded text once as the first public reasoning fragment for this reasoning block.
    #[must_use]
    pub fn take_seeded_reasoning_output(&mut self) -> Option<ChatGenerationOutput> {
        if !self.should_emit_seeded_reasoning {
            return None;
        }
        self.should_emit_seeded_reasoning = false;
        self.thinking_channel_seed
            .clone()
            .map(|text| ChatGenerationOutput::Reasoning { text })
    }

    /// Decodes and parses one generated model token at a stable engine boundary.
    pub fn push_token(
        &mut self,
        generated_token_id: u32,
    ) -> Result<Vec<Qwen3_5OutputEvent>, Qwen3_5RequestOutputError> {
        let Some(decoded_fragment) = self.token_decoder.push_token(generated_token_id)? else {
            return Ok(Vec::new());
        };
        self.output_parser
            .push_fragment(&decoded_fragment)
            .map_err(|parser_error| self.parser_error(parser_error))
    }

    /// Flushes stable text or rejects an unfinished model control structure.
    pub fn finish(&mut self) -> Result<Vec<Qwen3_5OutputEvent>, Qwen3_5RequestOutputError> {
        let mut output_events = Vec::new();
        if let Some(decoded_fragment) = self.token_decoder.finish()? {
            output_events.extend(
                self.output_parser
                    .push_fragment(&decoded_fragment)
                    .map_err(|parser_error| self.parser_error(parser_error))?,
            );
        }
        output_events.extend(
            self.output_parser
                .finish()
                .map_err(|parser_error| self.parser_error(parser_error))?,
        );
        Ok(output_events)
    }

    pub fn reset_after_model_visible_correction(&mut self, enable_thinking: bool) {
        self.enable_thinking = enable_thinking;
        self.output_parser
            .reset_after_model_visible_correction(enable_thinking);
        self.should_emit_seeded_reasoning = enable_thinking && self.thinking_channel_seed.is_some();
    }

    fn parser_error(&self, parser_error: Qwen3_5OutputParserError) -> Qwen3_5RequestOutputError {
        let diagnostic = MalformedModelOutputDiagnostic {
            diagnostic_code: parser_error.diagnostic_code(),
            parser_error: parser_error.to_string(),
            generated_token_ids: self
                .token_decoder
                .generated_token_ids_for_diagnostics()
                .to_vec(),
            pending_token_ids: self
                .token_decoder
                .pending_token_ids_for_diagnostics()
                .to_vec(),
            decoded_output_text: self.token_decoder.emitted_text_for_diagnostics().to_owned(),
            parser_state: self.output_parser.state_for_diagnostics(),
            parser_pending_output_text: self
                .output_parser
                .pending_output_for_diagnostics()
                .to_owned(),
        };
        Qwen3_5RequestOutputError::Parser {
            source: parser_error,
            diagnostic: Box::new(diagnostic),
        }
    }

    fn parser_initialization_error(
        parser_error: Qwen3_5OutputParserError,
    ) -> Qwen3_5RequestOutputError {
        let diagnostic = MalformedModelOutputDiagnostic {
            diagnostic_code: parser_error.diagnostic_code(),
            parser_error: parser_error.to_string(),
            generated_token_ids: Vec::new(),
            pending_token_ids: Vec::new(),
            decoded_output_text: String::new(),
            parser_state: "initialization",
            parser_pending_output_text: String::new(),
        };
        Qwen3_5RequestOutputError::Parser {
            source: parser_error,
            diagnostic: Box::new(diagnostic),
        }
    }
}

/// A typed failure while translating generated Qwen3.5 tokens.
#[derive(Debug, Error)]
pub enum Qwen3_5RequestOutputError {
    #[error("failed to decode a Qwen3.5 output token")]
    Tokenizer(#[from] Qwen3_5TokenizerError),
    #[error("failed to parse Qwen3.5 structured output")]
    Parser {
        #[source]
        source: Qwen3_5OutputParserError,
        diagnostic: Box<MalformedModelOutputDiagnostic>,
    },
}
