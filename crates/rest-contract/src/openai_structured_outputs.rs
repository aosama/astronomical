//! Extra-body structured generation that must be token-enforced or rejected.
//!
//! OpenAI `response_format` may degrade to a prompt plus Warning. These fields
//! must not: if the worker cannot mask illegal tokens, the request fails.

use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

/// Bounds one regex pattern so DFA compilation stays bounded and predictable.
pub const MAXIMUM_STRUCTURED_REGEX_PATTERN_BYTES: usize = 2_048;

/// vLLM-style extra body for constrained decoding.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct OpenAiStructuredOutputs {
    #[serde(rename = "json", alias = "json_schema", default)]
    json_schema: Option<Value>,
    #[serde(default)]
    regex: Option<String>,
    #[serde(default)]
    choice: Option<Vec<String>>,
    #[serde(default)]
    grammar: Option<String>,
}

/// Validated extra-body constraint that the worker can compile into a token mask.
#[derive(Clone, Debug, PartialEq)]
pub enum EnforcedStructuredGeneration {
    /// Any JSON value.
    JsonObject,
    /// JSON matching a bounded object schema.
    JsonSchema { schema: Value },
    /// Exact one of these UTF-8 strings.
    Choice { choices: Vec<String> },
    /// The complete visible answer must match this regular expression.
    Regex { pattern: String },
}

/// Why extra-body structured generation was rejected.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum OpenAiStructuredOutputsValidationError {
    #[error("structured_outputs must set exactly one of json, regex, choice, or grammar")]
    MultipleOrEmptyFields,
    #[error("structured_outputs.choice must contain at least one non-empty string")]
    EmptyChoice,
    #[error(
        "structured_outputs.regex pattern exceeds the bounded {maximum_pattern_bytes}-byte limit"
    )]
    RegexPatternTooLarge { maximum_pattern_bytes: usize },
    #[error("structured_outputs.regex pattern is not a supported regular expression: {reason}")]
    RegexNotCompilable { reason: String },
    #[error("structured_outputs.grammar cannot be enforced yet")]
    GrammarNotEnforced,
    #[error("guided_grammar cannot be enforced yet")]
    GuidedGrammarNotEnforced,
    #[error("structured_outputs.json must be an object schema")]
    JsonSchemaMustBeObject,
    #[error("set only one of structured_outputs or guided_grammar")]
    ConflictingExtraBodyFields,
}

impl OpenAiStructuredOutputs {
    /// Compiles extra-body fields into a worker-enforced constraint or fails closed.
    pub fn into_enforced_generation(
        self,
    ) -> Result<EnforcedStructuredGeneration, OpenAiStructuredOutputsValidationError> {
        let set_count = usize::from(self.json_schema.is_some())
            + usize::from(self.regex.is_some())
            + usize::from(self.choice.is_some())
            + usize::from(self.grammar.is_some());
        if set_count != 1 {
            return Err(OpenAiStructuredOutputsValidationError::MultipleOrEmptyFields);
        }
        if let Some(regex_pattern) = self.regex {
            if regex_pattern.is_empty() {
                return Err(OpenAiStructuredOutputsValidationError::RegexNotCompilable {
                    reason: "pattern is empty".to_owned(),
                });
            }
            if regex_pattern.len() > MAXIMUM_STRUCTURED_REGEX_PATTERN_BYTES {
                return Err(
                    OpenAiStructuredOutputsValidationError::RegexPatternTooLarge {
                        maximum_pattern_bytes: MAXIMUM_STRUCTURED_REGEX_PATTERN_BYTES,
                    },
                );
            }
            // Failing the DFA build here means the request cannot be enforced,
            // so it must fail closed at the public boundary instead of at the worker.
            regex_automata::dfa::dense::DFA::new(&regex_pattern).map_err(|build_error| {
                OpenAiStructuredOutputsValidationError::RegexNotCompilable {
                    reason: build_error.to_string(),
                }
            })?;
            return Ok(EnforcedStructuredGeneration::Regex {
                pattern: regex_pattern,
            });
        }
        if self.grammar.is_some() {
            return Err(OpenAiStructuredOutputsValidationError::GrammarNotEnforced);
        }
        if let Some(choices) = self.choice {
            let choices = choices
                .into_iter()
                .map(|choice| choice.trim().to_owned())
                .filter(|choice| !choice.is_empty())
                .collect::<Vec<_>>();
            if choices.is_empty() {
                return Err(OpenAiStructuredOutputsValidationError::EmptyChoice);
            }
            return Ok(EnforcedStructuredGeneration::Choice { choices });
        }
        let schema = self.json_schema.expect("json field counted as set");
        if schema.as_object().is_some_and(|fields| fields.is_empty()) {
            return Ok(EnforcedStructuredGeneration::JsonObject);
        }
        if !schema.is_object() {
            return Err(OpenAiStructuredOutputsValidationError::JsonSchemaMustBeObject);
        }
        Ok(EnforcedStructuredGeneration::JsonSchema { schema })
    }
}

/// Compiles extra-body `structured_outputs` or `guided_grammar`, never both.
pub fn enforced_generation_from_extra_body(
    structured_outputs: Option<OpenAiStructuredOutputs>,
    guided_grammar: Option<&str>,
) -> Result<Option<EnforcedStructuredGeneration>, OpenAiStructuredOutputsValidationError> {
    if structured_outputs.is_some() && guided_grammar.is_some() {
        return Err(OpenAiStructuredOutputsValidationError::ConflictingExtraBodyFields);
    }
    if let Some(structured_outputs) = structured_outputs {
        return structured_outputs.into_enforced_generation().map(Some);
    }
    if let Some(guided_grammar) = guided_grammar {
        return guided_grammar_to_enforced_generation(guided_grammar).map(Some);
    }
    Ok(None)
}

/// Compiles a guided EBNF string. Current workers cannot enforce EBNF.
pub fn guided_grammar_to_enforced_generation(
    guided_grammar: &str,
) -> Result<EnforcedStructuredGeneration, OpenAiStructuredOutputsValidationError> {
    if guided_grammar.trim().is_empty() {
        return Err(OpenAiStructuredOutputsValidationError::GuidedGrammarNotEnforced);
    }
    Err(OpenAiStructuredOutputsValidationError::GuidedGrammarNotEnforced)
}
