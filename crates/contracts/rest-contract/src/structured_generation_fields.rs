//! The structured-output fields Chat Completions and Responses both read, as
//! one group each request flattens in ahead of its catch-all map (see
//! `thinking_control_fields` for why the order matters): `response_format`,
//! and the extra-body constraints `structured_outputs` and `guided_grammar`.

use serde::Deserialize;

use crate::{
    EnforcedStructuredGeneration, OpenAiResponseFormat, OpenAiStructuredOutput,
    OpenAiStructuredOutputValidationError, OpenAiStructuredOutputs,
    OpenAiStructuredOutputsValidationError, enforced_generation_from_extra_body,
};

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub(crate) struct StructuredGenerationFields {
    #[serde(default)]
    response_format: Option<OpenAiResponseFormat>,
    #[serde(default)]
    structured_outputs: Option<OpenAiStructuredOutputs>,
    #[serde(default)]
    guided_grammar: Option<String>,
}

impl StructuredGenerationFields {
    /// The structured output `response_format` asks for, if any.
    pub(crate) fn response_format_output(
        &self,
    ) -> Result<Option<OpenAiStructuredOutput>, OpenAiStructuredOutputValidationError> {
        match self.response_format.clone() {
            Some(response_format) => response_format.into_structured_output(),
            None => Ok(None),
        }
    }

    /// The extra-body constraint that must be token-masked, if any.
    pub(crate) fn enforced_generation(
        &self,
    ) -> Result<Option<EnforcedStructuredGeneration>, OpenAiStructuredOutputsValidationError> {
        enforced_generation_from_extra_body(
            self.structured_outputs.clone(),
            self.guided_grammar.as_deref(),
        )
    }
}
