use astronomical_ipc_protocol::ChatToolDefinition;
use thiserror::Error;

use crate::MalformedModelOutputDiagnostic;

use super::{
    LagunaOutputEvent, LagunaOutputParser, LagunaOutputParserError, LagunaTextArtifactDescriptor,
    LagunaTokenDecoder, LagunaTokenizer, LagunaTokenizerError,
};

/// Request-local owner of incremental Laguna token decoding and Poolside parsing.
#[derive(Debug)]
pub struct LagunaRequestOutput {
    output_parser: LagunaOutputParser,
    token_decoder: LagunaTokenDecoder,
}

impl LagunaRequestOutput {
    /// Creates decode and parser state from one request's exact declarations.
    pub fn new(
        descriptor: &LagunaTextArtifactDescriptor,
        tokenizer: &LagunaTokenizer,
        declared_tools: &[ChatToolDefinition],
        generation_starts_in_reasoning: bool,
    ) -> Result<Self, LagunaRequestOutputError> {
        let output_parser =
            LagunaOutputParser::new(descriptor, declared_tools, generation_starts_in_reasoning)
                .map_err(Self::parser_initialization_error)?;
        Ok(Self {
            output_parser,
            token_decoder: tokenizer.incremental_decoder(),
        })
    }

    /// Decodes and parses one generated token at the neutral engine boundary.
    pub fn push_token(
        &mut self,
        generated_token_id: u32,
    ) -> Result<Vec<LagunaOutputEvent>, LagunaRequestOutputError> {
        let Some(decoded_fragment) = self.token_decoder.push_token(generated_token_id)? else {
            return Ok(Vec::new());
        };
        self.output_parser
            .push_fragment(&decoded_fragment)
            .map_err(|source| self.parser_error(source))
    }

    /// Flushes tokenizer and parser state at request completion.
    pub fn finish(&mut self) -> Result<Vec<LagunaOutputEvent>, LagunaRequestOutputError> {
        let mut output_events = Vec::new();
        if let Some(decoded_fragment) = self.token_decoder.finish()? {
            output_events.extend(
                self.output_parser
                    .push_fragment(&decoded_fragment)
                    .map_err(|source| self.parser_error(source))?,
            );
        }
        output_events.extend(
            self.output_parser
                .finish()
                .map_err(|source| self.parser_error(source))?,
        );
        Ok(output_events)
    }

    fn parser_error(&self, source: LagunaOutputParserError) -> LagunaRequestOutputError {
        let diagnostic = MalformedModelOutputDiagnostic {
            diagnostic_code: source.diagnostic_code(),
            parser_error: source.to_string(),
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
        LagunaRequestOutputError::Parser {
            source,
            diagnostic: Box::new(diagnostic),
        }
    }

    fn parser_initialization_error(source: LagunaOutputParserError) -> LagunaRequestOutputError {
        let diagnostic = MalformedModelOutputDiagnostic {
            diagnostic_code: source.diagnostic_code(),
            parser_error: source.to_string(),
            generated_token_ids: Vec::new(),
            pending_token_ids: Vec::new(),
            decoded_output_text: String::new(),
            parser_state: "initialization",
            parser_pending_output_text: String::new(),
        };
        LagunaRequestOutputError::Parser {
            source,
            diagnostic: Box::new(diagnostic),
        }
    }
}

/// Distinguishes fatal tokenizer failures from request-local malformed model output.
#[derive(Debug, Error)]
pub enum LagunaRequestOutputError {
    #[error("failed to decode a Laguna output token")]
    Tokenizer(#[from] LagunaTokenizerError),
    #[error("failed to parse Laguna structured output")]
    Parser {
        #[source]
        source: LagunaOutputParserError,
        diagnostic: Box<MalformedModelOutputDiagnostic>,
    },
}
