//! Applies OpenAI structured-output fallback while grammar masking is unavailable.

use astronomical_ipc_protocol::ChatMessage;
use astronomical_rest_contract::OpenAiStructuredOutput;
use axum::{
    http::{HeaderValue, header::WARNING},
    response::Response,
};

fn insert_json_output_instruction(
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
