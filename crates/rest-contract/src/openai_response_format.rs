//! OpenAI `response_format` for chat and Responses.
//!
//! Sample-time grammar masking is not active yet, so JSON mode is honored by a
//! bounded prompt instruction plus best-effort extraction. Callers learn that
//! the schema was not token-masked through the HTTP Warning header.

use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

use crate::MAX_OPENAI_TOOL_SCHEMA_NESTING_DEPTH;

/// Maximum serialized JSON Schema accepted on `response_format`.
pub const MAX_STRUCTURED_OUTPUT_SCHEMA_BYTES: usize = 65_536;
/// Advertised enforcement while sample-time grammar masking is unavailable.
pub const STRUCTURED_OUTPUT_ENFORCEMENT_NONE: &str = "none";

/// RFC 7234 Warning when json_object / json_schema cannot be grammar-enforced.
pub const UNENFORCED_RESPONSE_FORMAT_WARNING: &str = concat!(
    "199 astronomical ",
    r#""response_format not enforced; grammar-constrained decoding unavailable, output is best-effort""#
);

/// RFC 7234 Warning when `strict` json_schema cannot be grammar-enforced.
pub const UNENFORCED_STRICT_RESPONSE_FORMAT_WARNING: &str = concat!(
    "199 astronomical ",
    r#""response_format strict json_schema not enforced; grammar-constrained decoding unavailable, output is best-effort and NOT schema-enforced""#
);

/// Wire `response_format` object from Chat Completions.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenAiResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
    #[serde(default)]
    json_schema: Option<OpenAiJsonSchemaSpec>,
}

/// Nested `json_schema` object on Chat Completions `response_format`.
/// Extra keys are ignored so OpenAI clients that send unused optional fields still validate.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct OpenAiJsonSchemaSpec {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(rename = "schema", default)]
    schema: Option<Value>,
    #[serde(default)]
    strict: Option<bool>,
}

/// Validated structured-output request after public contract checks.
#[derive(Clone, Debug, PartialEq)]
pub enum OpenAiStructuredOutput {
    /// Any JSON value.
    JsonObject,
    /// JSON that should match the caller-supplied schema.
    JsonSchema {
        name: String,
        description: Option<String>,
        schema: Value,
        strict: bool,
    },
}

/// Rejection for a malformed or unsupported structured-output request.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum OpenAiStructuredOutputValidationError {
    /// `response_format.type` is not text, json_object, or json_schema.
    #[error("response_format type '{format_type}' is unsupported")]
    UnsupportedType { format_type: String },
    /// json_schema requests must name the schema.
    #[error("response_format json_schema name must not be empty")]
    JsonSchemaNameEmpty,
    /// json_schema requests must include a schema object.
    #[error("response_format json_schema.schema must be an object")]
    JsonSchemaMustBeObject,
    /// Schema nesting matches the tool-schema depth cap.
    #[error(
        "response_format schema nesting depth is {actual_schema_nesting_depth}, exceeding {maximum_schema_nesting_depth}"
    )]
    JsonSchemaNestingTooDeep {
        actual_schema_nesting_depth: usize,
        maximum_schema_nesting_depth: usize,
    },
    /// Schema payload is too large to inject into a prompt.
    #[error(
        "response_format schema is {actual_schema_bytes} bytes, exceeding the {maximum_schema_bytes} byte limit"
    )]
    JsonSchemaTooLarge {
        actual_schema_bytes: usize,
        maximum_schema_bytes: usize,
    },
    /// Chat `response_format` and Responses `text.format` disagreed.
    #[error("response_format and text.format must describe the same structured output")]
    ConflictingStructuredOutputFields,
}

impl OpenAiResponseFormat {
    /// Validates Chat Completions `response_format`. `text` becomes `None`.
    pub fn into_structured_output(
        self,
    ) -> Result<Option<OpenAiStructuredOutput>, OpenAiStructuredOutputValidationError> {
        structured_output_from_type_and_schema(
            &self.format_type,
            self.json_schema.as_ref().and_then(|spec| spec.name.clone()),
            self.json_schema
                .as_ref()
                .and_then(|spec| spec.description.clone()),
            self.json_schema
                .as_ref()
                .and_then(|spec| spec.schema.clone()),
            self.json_schema
                .as_ref()
                .and_then(|spec| spec.strict)
                .unwrap_or(false),
        )
    }
}

/// Parses Responses `text.format` into the same structured-output enum.
pub fn structured_output_from_responses_text_format(
    text_configuration: Option<&Value>,
) -> Result<Option<OpenAiStructuredOutput>, OpenAiStructuredOutputValidationError> {
    let Some(text_configuration) = text_configuration else {
        return Ok(None);
    };
    let Some(format_object) = text_configuration.get("format") else {
        return Ok(None);
    };
    let format_type = format_object
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("text");
    let nested_schema = format_object.get("json_schema");
    let schema_name = format_object
        .get("name")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            nested_schema
                .and_then(|spec| spec.get("name"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        });
    let schema_description = format_object
        .get("description")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            nested_schema
                .and_then(|spec| spec.get("description"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        });
    let schema = format_object
        .get("schema")
        .cloned()
        .or_else(|| nested_schema.and_then(|spec| spec.get("schema")).cloned());
    let strict = format_object
        .get("strict")
        .and_then(Value::as_bool)
        .or_else(|| {
            nested_schema
                .and_then(|spec| spec.get("strict"))
                .and_then(Value::as_bool)
        })
        .unwrap_or(false);
    structured_output_from_type_and_schema(
        format_type,
        schema_name,
        schema_description,
        schema,
        strict,
    )
}

/// Picks one structured-output request when both Chat and Responses fields are set.
pub fn merge_structured_output_requests(
    response_format: Option<OpenAiStructuredOutput>,
    text_format: Option<OpenAiStructuredOutput>,
) -> Result<Option<OpenAiStructuredOutput>, OpenAiStructuredOutputValidationError> {
    match (response_format, text_format) {
        (None, None) => Ok(None),
        (Some(structured_output), None) | (None, Some(structured_output)) => {
            Ok(Some(structured_output))
        }
        (Some(left), Some(right)) if left == right => Ok(Some(left)),
        (Some(_), Some(_)) => {
            Err(OpenAiStructuredOutputValidationError::ConflictingStructuredOutputFields)
        }
    }
}

fn structured_output_from_type_and_schema(
    format_type: &str,
    schema_name: Option<String>,
    schema_description: Option<String>,
    schema: Option<Value>,
    strict: bool,
) -> Result<Option<OpenAiStructuredOutput>, OpenAiStructuredOutputValidationError> {
    match format_type {
        "text" => Ok(None),
        "json_object" => Ok(Some(OpenAiStructuredOutput::JsonObject)),
        "json_schema" => {
            let name = schema_name.unwrap_or_default();
            if name.trim().is_empty() {
                return Err(OpenAiStructuredOutputValidationError::JsonSchemaNameEmpty);
            }
            let Some(schema) = schema else {
                return Err(OpenAiStructuredOutputValidationError::JsonSchemaMustBeObject);
            };
            if !schema.is_object() {
                return Err(OpenAiStructuredOutputValidationError::JsonSchemaMustBeObject);
            }
            let schema_nesting_depth = json_nesting_depth(&schema);
            if schema_nesting_depth > MAX_OPENAI_TOOL_SCHEMA_NESTING_DEPTH {
                return Err(
                    OpenAiStructuredOutputValidationError::JsonSchemaNestingTooDeep {
                        actual_schema_nesting_depth: schema_nesting_depth,
                        maximum_schema_nesting_depth: MAX_OPENAI_TOOL_SCHEMA_NESTING_DEPTH,
                    },
                );
            }
            let serialized_schema = serde_json::to_vec(&schema).unwrap_or_default();
            if serialized_schema.len() > MAX_STRUCTURED_OUTPUT_SCHEMA_BYTES {
                return Err(OpenAiStructuredOutputValidationError::JsonSchemaTooLarge {
                    actual_schema_bytes: serialized_schema.len(),
                    maximum_schema_bytes: MAX_STRUCTURED_OUTPUT_SCHEMA_BYTES,
                });
            }
            Ok(Some(OpenAiStructuredOutput::JsonSchema {
                name,
                description: schema_description.filter(|description| !description.is_empty()),
                schema,
                strict,
            }))
        }
        other_format_type => Err(OpenAiStructuredOutputValidationError::UnsupportedType {
            format_type: other_format_type.to_owned(),
        }),
    }
}

impl OpenAiStructuredOutput {
    /// Prompt instruction used while sample-time grammar masking is unavailable.
    #[must_use]
    pub fn json_output_instruction(&self) -> String {
        match self {
            Self::JsonObject => {
                "Output a single JSON value and nothing else after any reasoning: no markdown fences and no prose."
                    .to_owned()
            }
            Self::JsonSchema {
                name,
                description,
                schema,
                ..
            } => {
                let serialized_schema =
                    serde_json::to_string(schema).unwrap_or_else(|_| "{}".to_owned());
                let description_clause = description
                    .as_deref()
                    .map(|description| format!(" ({description})"))
                    .unwrap_or_default();
                format!(
                    "Output a single JSON object named {name}{description_clause} matching this schema and nothing else after any reasoning: no markdown fences and no prose. Schema: {serialized_schema}"
                )
            }
        }
    }

    /// HTTP Warning value when this request cannot be grammar-enforced.
    #[must_use]
    pub fn unenforced_warning_header(&self) -> &'static str {
        match self {
            Self::JsonSchema { strict: true, .. } => UNENFORCED_STRICT_RESPONSE_FORMAT_WARNING,
            Self::JsonObject | Self::JsonSchema { strict: false, .. } => {
                UNENFORCED_RESPONSE_FORMAT_WARNING
            }
        }
    }
}

/// Parses a JSON value from model text without coercing fields or filling defaults.
#[must_use]
pub fn extract_json_value_from_text(visible_text: &str) -> Option<Value> {
    let trimmed_text = visible_text.trim();
    if let Ok(parsed_json) = serde_json::from_str::<Value>(trimmed_text) {
        return Some(parsed_json);
    }
    if let Some(fenced_payload) = fenced_json_payload(trimmed_text)
        && let Ok(parsed_json) = serde_json::from_str::<Value>(fenced_payload)
    {
        return Some(parsed_json);
    }
    extract_first_json_value(trimmed_text)
}

/// Compact JSON text when extraction succeeds.
#[must_use]
pub fn compact_extracted_json_text(visible_text: &str) -> Option<String> {
    let parsed_json = extract_json_value_from_text(visible_text)?;
    serde_json::to_string(&parsed_json).ok()
}

fn fenced_json_payload(visible_text: &str) -> Option<&str> {
    let fence_start = visible_text.find("```")?;
    let after_opening_ticks = &visible_text[fence_start + 3..];
    let after_language = after_opening_ticks
        .strip_prefix("json")
        .unwrap_or(after_opening_ticks);
    let after_newline = after_language
        .strip_prefix("\r\n")
        .or_else(|| after_language.strip_prefix('\n'))?;
    let fence_end = after_newline.find("```")?;
    Some(after_newline[..fence_end].trim())
}

fn extract_first_json_value(visible_text: &str) -> Option<Value> {
    let json_start = visible_text.find(['{', '['])?;
    let mut json_deserializer = serde_json::Deserializer::from_str(&visible_text[json_start..]);
    Value::deserialize(&mut json_deserializer).ok()
}

fn json_nesting_depth(json_value: &Value) -> usize {
    match json_value {
        Value::Array(values) => values
            .iter()
            .map(json_nesting_depth)
            .max()
            .unwrap_or(0)
            .saturating_add(1),
        Value::Object(fields) => fields
            .values()
            .map(json_nesting_depth)
            .max()
            .unwrap_or(0)
            .saturating_add(1),
        _ => 0,
    }
}
