use astronomical_ipc_protocol::ChatGenerationOutput;

use super::{
    K2HorizonMoVAOutputParser, K2HorizonMoVATokenDecoder, K2HorizonMoVATokenizer,
    K2HorizonMoVATokenizerError,
};
use crate::ModelGenerationOutputError;

/// Request-local decode and IFM parse state.
#[derive(Debug)]
pub struct K2HorizonMoVARequestOutput {
    output_parser: K2HorizonMoVAOutputParser,
    token_decoder: K2HorizonMoVATokenDecoder,
}

impl K2HorizonMoVARequestOutput {
    #[must_use]
    pub fn new(tokenizer: &K2HorizonMoVATokenizer) -> Self {
        Self::new_with_declared_tool_names(tokenizer, Vec::new())
    }

    #[must_use]
    pub fn new_with_declared_tool_names(
        tokenizer: &K2HorizonMoVATokenizer,
        declared_tool_names: Vec<String>,
    ) -> Self {
        Self {
            output_parser: K2HorizonMoVAOutputParser::with_declared_tool_names(declared_tool_names),
            token_decoder: tokenizer.incremental_decoder(),
        }
    }

    pub fn push_token(
        &mut self,
        generated_token_id: u32,
    ) -> Result<Vec<ChatGenerationOutput>, ModelGenerationOutputError> {
        let Some(decoded_fragment) = self
            .token_decoder
            .push_token(generated_token_id)
            .map_err(|error| tokenizer_output_error(&error))?
        else {
            return Ok(Vec::new());
        };
        Ok(self.output_parser.push_text(&decoded_fragment))
    }

    pub fn finish(&mut self) -> Vec<ChatGenerationOutput> {
        self.output_parser.finish()
    }
}

fn tokenizer_output_error(error: &K2HorizonMoVATokenizerError) -> ModelGenerationOutputError {
    ModelGenerationOutputError::Fatal {
        reason: error.to_string(),
    }
}
