//! Validates the raw JSON schema text one CLI `respond --schema` request
//! carries before it may become a worker generation constraint.
//!
//! The CLI stays thin: it reads the schema file and forwards the text. The
//! daemon owns validation because the IPC request is a trust boundary: an
//! oversized, unparseable, or non-object schema is rejected here with a
//! clear reason instead of reaching the worker or failing mid-generation.

use astronomical_ipc_protocol::{MAXIMUM_CHAT_SCHEMA_JSON_BYTES, StructuredGenerationConstraint};
use astronomical_rest_contract::{
    OpenAiStructuredOutputsValidationError, enforced_generation_from_json_schema,
};

use crate::structured_output::constraint_from_enforced_generation;

/// Parses and bounds the raw schema text into the worker-enforced JSON
/// constraint. The returned reason is a user-facing CLI error message.
///
/// The object-schema rule itself (empty object means any JSON object, a
/// non-object schema is not enforceable) lives in the shared
/// `enforced_generation_from_json_schema` so this surface and the REST
/// surface cannot drift apart; only the user-facing wording is local.
pub fn validated_chat_schema_constraint(
    schema_json: &str,
) -> Result<StructuredGenerationConstraint, String> {
    if schema_json.len() > MAXIMUM_CHAT_SCHEMA_JSON_BYTES {
        return Err(format!(
            "the --schema file is {} bytes, over the {}-byte limit; send a smaller schema file",
            schema_json.len(),
            MAXIMUM_CHAT_SCHEMA_JSON_BYTES
        ));
    }
    let parsed_schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|parse_error| format!("the --schema file is not valid JSON: {parse_error}"))?;
    let enforced_generation =
        enforced_generation_from_json_schema(parsed_schema).map_err(|schema_rejection| {
            match schema_rejection {
                OpenAiStructuredOutputsValidationError::JsonSchemaMustBeObject => {
                    "the --schema file must contain one JSON object schema".to_owned()
                }
                other_schema_rejection => {
                    format!("the --schema file was rejected: {other_schema_rejection}")
                }
            }
        })?;
    // Re-serializing through the constraint keeps the compact canonical form
    // the worker's DFA compiler sees and drops trailing whitespace noise.
    Ok(constraint_from_enforced_generation(enforced_generation))
}
