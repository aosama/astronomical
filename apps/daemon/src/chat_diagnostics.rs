use serde_json::Value;
use sha2::{Digest, Sha256};

const MESSAGE_ROLE_SEQUENCE_PREVIEW_LIMIT: usize = 16;

/// Request data captured for trace-level OpenAI chat diagnostics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenAiChatRequestDiagnosticSnapshot {
    /// Number of raw HTTP body bytes received from the client.
    pub request_body_bytes: usize,
    /// SHA-256 fingerprint of raw request bytes for request correlation.
    pub request_body_sha256: String,
}

/// Bounded request data that is safe enough for default info-level request correlation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenAiChatRequestInfoDiagnosticSnapshot {
    pub message_count: Option<usize>,
    pub message_role_sequence_preview: Option<String>,
    pub last_user_message_character_count: Option<usize>,
    pub last_user_message_sha256: Option<String>,
}

/// Builds the request snapshot written to trace logs before validation mutates the data.
#[must_use]
pub fn build_openai_chat_request_diagnostic_snapshot(
    request_body_bytes: &[u8],
) -> OpenAiChatRequestDiagnosticSnapshot {
    OpenAiChatRequestDiagnosticSnapshot {
        request_body_bytes: request_body_bytes.len(),
        request_body_sha256: sha256_hex(request_body_bytes),
    }
}

/// Builds a compact request summary for info logs without dumping the full prompt.
#[must_use]
pub fn build_openai_chat_request_info_diagnostic_snapshot(
    request_body_bytes: &[u8],
) -> OpenAiChatRequestInfoDiagnosticSnapshot {
    let Ok(request_json_value) = serde_json::from_slice::<Value>(request_body_bytes) else {
        return empty_info_diagnostic_snapshot();
    };
    let Some(messages_json_values) = request_json_value.get("messages").and_then(Value::as_array)
    else {
        return empty_info_diagnostic_snapshot();
    };

    let last_user_message_content = messages_json_values
        .iter()
        .rev()
        .find(|message_json_value| {
            message_json_value
                .get("role")
                .and_then(Value::as_str)
                .is_some_and(|role| role == "user")
        })
        .and_then(|message_json_value| message_json_value.get("content"))
        .and_then(extract_text_from_openai_content_value);

    OpenAiChatRequestInfoDiagnosticSnapshot {
        message_count: Some(messages_json_values.len()),
        message_role_sequence_preview: Some(summarize_message_roles(messages_json_values)),
        last_user_message_character_count: last_user_message_content
            .as_ref()
            .map(|content| content.chars().count()),
        last_user_message_sha256: last_user_message_content
            .as_ref()
            .map(|content| sha256_hex(content.as_bytes())),
    }
}

fn empty_info_diagnostic_snapshot() -> OpenAiChatRequestInfoDiagnosticSnapshot {
    OpenAiChatRequestInfoDiagnosticSnapshot {
        message_count: None,
        message_role_sequence_preview: None,
        last_user_message_character_count: None,
        last_user_message_sha256: None,
    }
}

fn summarize_message_roles(messages_json_values: &[Value]) -> String {
    let mut message_role_sequence_preview = String::new();
    for (message_index, message_json_value) in messages_json_values
        .iter()
        .take(MESSAGE_ROLE_SEQUENCE_PREVIEW_LIMIT)
        .enumerate()
    {
        if message_index > 0 {
            message_role_sequence_preview.push(',');
        }
        message_role_sequence_preview.push_str(message_role_for_diagnostics(message_json_value));
    }
    if messages_json_values.len() > MESSAGE_ROLE_SEQUENCE_PREVIEW_LIMIT {
        message_role_sequence_preview.push_str(",...");
    }
    message_role_sequence_preview
}

fn message_role_for_diagnostics(message_json_value: &Value) -> &'static str {
    match message_json_value.get("role").and_then(Value::as_str) {
        Some("system") => "system",
        Some("user") => "user",
        Some("assistant") => "assistant",
        Some("tool") => "tool",
        Some(_) | None => "unknown",
    }
}

fn extract_text_from_openai_content_value(content_json_value: &Value) -> Option<String> {
    if let Some(content_text) = content_json_value.as_str() {
        return Some(content_text.to_owned());
    }
    let content_parts = content_json_value.as_array()?;
    let mut combined_text_content = String::new();
    for content_part in content_parts {
        if content_part
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|content_part_type| content_part_type == "text")
            && let Some(text_content_part) = content_part.get("text").and_then(Value::as_str)
        {
            combined_text_content.push_str(text_content_part);
        }
    }
    Some(combined_text_content)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex_string = String::with_capacity(digest.len() * 2);
    for digest_byte in digest {
        hex_string.push_str(&format!("{digest_byte:02x}"));
    }
    hex_string
}
