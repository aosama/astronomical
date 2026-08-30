use astronomical_ipc_protocol::{
    ChatAssistantToolCall, ChatAssistantToolFunction, ChatGenerationCommand,
    ChatGenerationSettings, ChatMessage, ChatToolChoice, ChatToolDefinition, RequestId,
};
use astronomical_rest_contract::OpenAiResponsesRequest;
use astronomical_supervisor::translate_openai_responses_request;

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

    let chat_generation_command = translate_openai_responses_request(RequestId::new(700), request)
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
                seed: None,
                thinking_budget: None,
            },
            qwen_thinking_channel_seed: None,
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

    let chat_generation_command = translate_openai_responses_request(RequestId::new(701), request)
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
