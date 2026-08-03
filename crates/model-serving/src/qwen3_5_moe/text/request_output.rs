use astronomical_ipc_protocol::ChatToolDefinition;
use thiserror::Error;

use super::{
    Qwen3_5MoEOutputEvent, Qwen3_5MoEOutputParser, Qwen3_5MoEOutputParserError,
    Qwen3_5MoETokenDecoder, Qwen3_5MoETokenizer, Qwen3_5MoETokenizerError,
};
use crate::MalformedModelOutputDiagnostic;

/// Request-local owner that translates generated token IDs into neutral chat events.
#[derive(Debug)]
pub struct Qwen3_5MoERequestOutput {
    output_parser: Qwen3_5MoEOutputParser,
    token_decoder: Qwen3_5MoETokenDecoder,
}

impl Qwen3_5MoERequestOutput {
    /// Creates bounded decode/parser state from the exact request tool declarations.
    pub fn new(
        tokenizer: &Qwen3_5MoETokenizer,
        declared_tools: &[ChatToolDefinition],
        enable_thinking: bool,
    ) -> Result<Self, Qwen3_5MoERequestOutputError> {
        let output_parser = if enable_thinking {
            Qwen3_5MoEOutputParser::new_after_thinking_prefix(declared_tools)
                .map_err(Self::parser_initialization_error)?
        } else {
            Qwen3_5MoEOutputParser::new(declared_tools)
                .map_err(Self::parser_initialization_error)?
        };
        Ok(Self {
            output_parser,
            token_decoder: tokenizer.incremental_decoder(),
        })
    }

    /// Decodes and parses one generated model token at a stable engine boundary.
    pub fn push_token(
        &mut self,
        generated_token_id: u32,
    ) -> Result<Vec<Qwen3_5MoEOutputEvent>, Qwen3_5MoERequestOutputError> {
        let Some(decoded_fragment) = self.token_decoder.push_token(generated_token_id)? else {
            return Ok(Vec::new());
        };
        self.output_parser
            .push_fragment(&decoded_fragment)
            .map_err(|parser_error| self.parser_error(parser_error))
    }

    /// Flushes stable text or rejects an unfinished model control structure.
    pub fn finish(&mut self) -> Result<Vec<Qwen3_5MoEOutputEvent>, Qwen3_5MoERequestOutputError> {
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

    pub(crate) fn reset_after_model_visible_correction(&mut self, enable_thinking: bool) {
        self.output_parser
            .reset_after_model_visible_correction(enable_thinking);
    }

    fn parser_error(
        &self,
        parser_error: Qwen3_5MoEOutputParserError,
    ) -> Qwen3_5MoERequestOutputError {
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
        Qwen3_5MoERequestOutputError::Parser {
            source: parser_error,
            diagnostic: Box::new(diagnostic),
        }
    }

    fn parser_initialization_error(
        parser_error: Qwen3_5MoEOutputParserError,
    ) -> Qwen3_5MoERequestOutputError {
        let diagnostic = MalformedModelOutputDiagnostic {
            diagnostic_code: parser_error.diagnostic_code(),
            parser_error: parser_error.to_string(),
            generated_token_ids: Vec::new(),
            pending_token_ids: Vec::new(),
            decoded_output_text: String::new(),
            parser_state: "initialization",
            parser_pending_output_text: String::new(),
        };
        Qwen3_5MoERequestOutputError::Parser {
            source: parser_error,
            diagnostic: Box::new(diagnostic),
        }
    }
}

/// A typed failure while translating generated Qwen3.5-MoE tokens.
#[derive(Debug, Error)]
pub enum Qwen3_5MoERequestOutputError {
    #[error("failed to decode a Qwen3.5-MoE output token")]
    Tokenizer(#[from] Qwen3_5MoETokenizerError),
    #[error("failed to parse Qwen3.5-MoE structured output")]
    Parser {
        #[source]
        source: Qwen3_5MoEOutputParserError,
        diagnostic: Box<MalformedModelOutputDiagnostic>,
    },
}
