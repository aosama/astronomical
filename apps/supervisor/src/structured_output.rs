//! Applies OpenAI structured-output fallback while grammar masking is unavailable.

use astronomical_ipc_protocol::{ChatMessage, StructuredGenerationConstraint};
use astronomical_rest_contract::{EnforcedStructuredGeneration, OpenAiStructuredOutput};
use axum::{
    http::{HeaderValue, header::WARNING},
    response::Response,
};

/// Inserts the JSON-output prompt instruction ahead of the chat messages,
/// reusing the template convention that the first system message is the root
/// instruction. Enforcement pairs with this hint: the token mask clamps the
/// visible channel, and without the hint the model does not plan a JSON
/// object answer (observed live: the mask forced a brace and the model
/// answered `{"error": "Invalid JSON"}`).
pub(crate) fn insert_json_output_instruction(
    chat_messages: &mut Vec<ChatMessage>,
    json_output_instruction: String,
) {
    match chat_messages.first_mut() {
        Some(ChatMessage::System { content }) => {
            // Templates treat the first system message as root instruction. A later
            // system turn is lowered to a chronological user update and can be ignored.
            content.push_str("\n\n");
            content.push_str(&json_output_instruction);
        }
        _ => chat_messages.insert(
            0,
            ChatMessage::System {
                content: json_output_instruction,
            },
        ),
    }
}

pub(crate) fn ipc_constraint_from_enforced(
    enforced_structured_generation: Option<EnforcedStructuredGeneration>,
) -> Option<StructuredGenerationConstraint> {
    enforced_structured_generation.map(constraint_from_enforced_generation)
}

pub(crate) fn constraint_from_enforced_generation(
    enforced_structured_generation: EnforcedStructuredGeneration,
) -> StructuredGenerationConstraint {
    match enforced_structured_generation {
        EnforcedStructuredGeneration::JsonObject => StructuredGenerationConstraint::JsonObject,
        EnforcedStructuredGeneration::JsonSchema { schema } => {
            StructuredGenerationConstraint::JsonSchema {
                schema_json: schema.to_string(),
            }
        }
        EnforcedStructuredGeneration::Choice { choices } => {
            StructuredGenerationConstraint::Choice { choices }
        }
        EnforcedStructuredGeneration::Regex { pattern } => {
            StructuredGenerationConstraint::Regex { pattern }
        }
    }
}

/// Prompt instruction paired with an enforced IPC schema constraint. The CLI
/// sends a bare schema file, so there is no schema name or description to
/// include; the wording mirrors the enforced REST instruction otherwise.
pub(crate) fn enforced_schema_output_instruction(schema_json: &str) -> String {
    format!(
        "Output a single JSON object matching this schema and nothing else after any \
         reasoning: no markdown fences and no prose. Schema: {schema_json}"
    )
}

pub(crate) fn apply_structured_output_instruction(
    chat_messages: &mut Vec<ChatMessage>,
    structured_output: Option<&OpenAiStructuredOutput>,
) {
    if let Some(structured_output) = structured_output {
        insert_json_output_instruction(chat_messages, structured_output.json_output_instruction());
    }
}

pub(crate) fn attach_unenforced_structured_output_warning(
    mut response: Response,
    structured_output: Option<&OpenAiStructuredOutput>,
) -> Response {
    let Some(structured_output) = structured_output else {
        return response;
    };
    // Always disclose prompt-injected JSON. Success is not grammar enforcement.
    response.headers_mut().insert(
        WARNING,
        HeaderValue::from_static(structured_output.unenforced_warning_header()),
    );
    response
}
