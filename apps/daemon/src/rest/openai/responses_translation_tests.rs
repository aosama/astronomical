//! Unit tests for `crate::rest::openai::responses_translation`: how an
//! OpenAI Responses request lowers into the worker's chat command.

use ipc_protocol::{
    ChatAssistantToolCall, ChatAssistantToolFunction, ChatGenerationCommand,
    ChatGenerationSettings, ChatMessage, ChatToolChoice, ChatToolDefinition, RequestId,
};
use rest_contract::OpenAiResponsesRequest;

#[test]
fn should_translate_string_input_into_one_user_chat_message() {
    let request = serde_json::from_str::<OpenAiResponsesRequest>(
        r#"{
            "model":"astronomical/fake-mixture-of-experts",
            "input":"Explain this repository.",
            "max_output_tokens":512,
            "temperature":0.6,
            "top_p":0.95
        }"#,
    )
    .expect("the Responses request should deserialize");

    let chat_generation_command =
        crate::translate_openai_responses_request(RequestId::new(700), request)
            .expect("the Responses request should translate");

    assert_eq!(
        chat_generation_command,
        ChatGenerationCommand {
            request_id: RequestId::new(700),
            model: "astronomical/fake-mixture-of-experts".to_owned(),
            messages: vec![ChatMessage::User {
                content: "Explain this repository.".to_owned(),
                images: Vec::new(),
            }],
            tools: Vec::new(),
            tool_choice: ChatToolChoice::Auto,
            settings: ChatGenerationSettings {
                max_output_tokens: 512,
                temperature_thousandths: Some(600),
                top_p_thousandths: Some(950),
                repetition_penalty_thousandths: None,
                seed: None,
                thinking_budget: None,
            },
            structured_generation: None,
        }
    );
}

#[test]
fn should_translate_summary_reasoning_and_function_loop_replay() {
    let request = serde_json::from_str::<OpenAiResponsesRequest>(
        r#"{
            "model":"astronomical/fake-mixture-of-experts",
            "instructions":"You are a coding assistant.",
            "input":[
                {"role":"user","content":"Inspect files."},
                {"type":"reasoning","id":"rs_prior","summary":[{"type":"summary_text","text":"I should list files."}],"content":[]},
                {"type":"function_call","id":"fc_prior","call_id":"call_prior","name":"glob","arguments":"{\"pattern\":\"**/*.rs\"}","status":"completed"},
                {"type":"function_call_output","call_id":"call_prior","output":"src/lib.rs"},
                {"role":"user","content":"Summarize it."}
            ],
            "tools":[{"type":"function","name":"glob","parameters":{"type":"object"},"strict":false}],
            "tool_choice":"auto"
        }"#,
    )
    .expect("the function-loop request should deserialize");

    let chat_generation_command =
        crate::translate_openai_responses_request(RequestId::new(701), request)
            .expect("the manual function loop should translate");

    assert_eq!(
        chat_generation_command.messages,
        vec![
            ChatMessage::System {
                content: "You are a coding assistant.".to_owned(),
            },
            ChatMessage::User {
                content: "Inspect files.".to_owned(),
                images: Vec::new(),
            },
            ChatMessage::Assistant {
                content: None,
                reasoning_content: Some("I should list files.".to_owned()),
                tool_calls: vec![ChatAssistantToolCall {
                    id: "call_prior".to_owned(),
                    function: ChatAssistantToolFunction {
                        name: "glob".to_owned(),
                        arguments_json: r#"{"pattern":"**/*.rs"}"#.to_owned(),
                    },
                }],
            },
            ChatMessage::Tool {
                tool_call_id: "call_prior".to_owned(),
                content: "src/lib.rs".to_owned(),
            },
            ChatMessage::User {
                content: "Summarize it.".to_owned(),
                images: Vec::new(),
            },
        ]
    );
    assert_eq!(
        chat_generation_command.tools,
        vec![ChatToolDefinition {
            name: "glob".to_owned(),
            description: None,
            parameters_json: r#"{"type":"object"}"#.to_owned(),
        }]
    );
}

/// A 1x1 red PNG, the smallest image the decoder accepts as real.
const RED_PIXEL_DATA_URI: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";

/// Each chat message as a short label, so a test reads the whole order.
fn message_order_labels(chat_messages: &[ChatMessage]) -> Vec<String> {
    let mut labels = Vec::with_capacity(chat_messages.len());
    for chat_message in chat_messages {
        labels.push(match chat_message {
            ChatMessage::System { .. } => "system".to_owned(),
            ChatMessage::User { content, images } => {
                format!("user '{content}' with {} images", images.len())
            }
            ChatMessage::Assistant { .. } => "assistant".to_owned(),
            ChatMessage::Tool {
                tool_call_id,
                content,
            } => {
                format!("tool {tool_call_id} '{content}'")
            }
        });
    }
    labels
}

fn translated_responses_messages(request_document: serde_json::Value) -> Vec<ChatMessage> {
    let request = serde_json::from_value::<OpenAiResponsesRequest>(request_document)
        .expect("the Responses request should deserialize");
    crate::translate_openai_responses_request(RequestId::new(910), request)
        .expect("the Responses request should translate")
        .messages
}

#[test]
fn should_follow_a_run_of_function_call_outputs_with_the_images_they_returned() {
    let chat_messages = translated_responses_messages(serde_json::json!({
        "model": "astronomical/fake-mixture-of-experts",
        "input": [
            {"role": "user", "content": "Capture both stages."},
            {"type": "function_call", "call_id": "call_1", "name": "screenshot", "arguments": "{}"},
            {"type": "function_call", "call_id": "call_2", "name": "screenshot", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": [
                {"type": "input_text", "text": "first"},
                {"type": "input_image", "image_url": RED_PIXEL_DATA_URI}
            ]},
            {"type": "function_call_output", "call_id": "call_2", "output": [
                {"type": "input_image", "image_url": RED_PIXEL_DATA_URI, "file_id": null}
            ]},
            {"role": "user", "content": "Compare them."}
        ]
    }));

    assert_eq!(
        message_order_labels(&chat_messages),
        [
            "user 'Capture both stages.' with 0 images",
            "assistant",
            "tool call_1 'first'",
            "tool call_2 ''",
            "user 'Images returned by the tool calls above.' with 2 images",
            "user 'Compare them.' with 0 images",
        ]
    );
}

#[test]
fn should_add_the_tool_images_last_when_the_input_ends_with_function_call_outputs() {
    let chat_messages = translated_responses_messages(serde_json::json!({
        "model": "astronomical/fake-mixture-of-experts",
        "input": [
            {"role": "user", "content": "Capture the stage."},
            {"type": "function_call", "call_id": "call_1", "name": "screenshot", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": [
                {"type": "input_image", "image_url": RED_PIXEL_DATA_URI}
            ]}
        ]
    }));

    assert_eq!(
        message_order_labels(&chat_messages),
        [
            "user 'Capture the stage.' with 0 images",
            "assistant",
            "tool call_1 ''",
            "user 'Images returned by the tool calls above.' with 1 images",
        ]
    );
}

#[test]
fn should_add_no_image_message_after_text_function_call_outputs() {
    let chat_messages = translated_responses_messages(serde_json::json!({
        "model": "astronomical/fake-mixture-of-experts",
        "input": [
            {"role": "user", "content": "Read the play."},
            {"type": "function_call", "call_id": "call_1", "name": "read", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "Exit, pursued by a bear."}
        ]
    }));

    assert_eq!(
        message_order_labels(&chat_messages),
        [
            "user 'Read the play.' with 0 images",
            "assistant",
            "tool call_1 'Exit, pursued by a bear.'",
        ]
    );
}
