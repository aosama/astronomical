use astronomical_daemon::{ChatGenerationStreamEvent, build_application};
use axum::{
    body::to_bytes,
    http::{StatusCode, header},
};
use ipc_protocol::ChatGenerationCompletionReason;
use rest_contract::UNENFORCED_RESPONSE_FORMAT_WARNING;
use tower::ServiceExt;

use crate::rest_api::chat_endpoint_fixtures::{ScriptedChatExecutor, chat_completion_request};

const MODEL_ID: &str = "astronomical/non-streaming-test-model";

#[tokio::test]
async fn should_return_extracted_json_and_a_warning_for_json_schema_chat() {
    let application = build_application(ScriptedChatExecutor::ready(
        MODEL_ID,
        vec![
        ChatGenerationStreamEvent::TextFragment(
            "Juliet answers:\n```json\n{\"speaker\":\"Juliet\",\"play\":\"Romeo and Juliet\"}\n```"
                .to_owned(),
        ),
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 12,
            generated_token_count: 8,
            reasoning_token_count: 0,
            astro_ssd_prompt_cache_restored_token_count: 0,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        },
    ],
    ));

    let chat_response = application
        .oneshot(
            chat_completion_request(r#"{
                        "model":"astronomical/non-streaming-test-model",
                        "messages":[{"role":"user","content":"O Romeo, Romeo, wherefore art thou Romeo?"}],
                        "response_format":{
                            "type":"json_schema",
                            "json_schema":{
                                "name":"romeo_line",
                                "schema":{"type":"object","properties":{"speaker":{"type":"string"},"play":{"type":"string"}},"required":["speaker","play"]}
                            }
                        },
                        "stream":false
                    }"#),
        )
        .await
        .expect("the application should return a chat response");

    assert_eq!(chat_response.status(), StatusCode::OK);
    let warning_header = chat_response
        .headers()
        .get(header::WARNING)
        .and_then(|header_value| header_value.to_str().ok())
        .expect("unenforced json_schema must disclose a Warning header");
    assert_eq!(warning_header, UNENFORCED_RESPONSE_FORMAT_WARNING);
    let response_body = to_bytes(chat_response.into_body(), 16 * 1024)
        .await
        .expect("the chat body should be readable");
    let response_document: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the chat body should be JSON");
    let visible_content = response_document["choices"][0]["message"]["content"]
        .as_str()
        .expect("assistant content should be text");
    let extracted_json: serde_json::Value =
        serde_json::from_str(visible_content).expect("visible content should be JSON");
    assert_eq!(
        extracted_json,
        serde_json::json!({"speaker": "Juliet", "play": "Romeo and Juliet"})
    );
}

#[tokio::test]
async fn should_keep_original_text_when_json_cannot_be_extracted() {
    let application = build_application(ScriptedChatExecutor::ready(
        MODEL_ID,
        vec![
            ChatGenerationStreamEvent::TextFragment(
                "Two households, both alike in dignity.".to_owned(),
            ),
            ChatGenerationStreamEvent::Completed {
                prompt_token_count: 8,
                generated_token_count: 8,
                reasoning_token_count: 0,
                astro_ssd_prompt_cache_restored_token_count: 0,
                reason: ChatGenerationCompletionReason::EndOfSequence,
            },
        ],
    ));

    let chat_response = application
        .oneshot(
            chat_completion_request(r#"{
                        "model":"astronomical/non-streaming-test-model",
                        "messages":[{"role":"user","content":"O Romeo, Romeo, wherefore art thou Romeo?"}],
                        "response_format":{"type":"json_object"},
                        "stream":false
                    }"#),
        )
        .await
        .expect("the application should return a chat response");

    assert_eq!(chat_response.status(), StatusCode::OK);
    let response_body = to_bytes(chat_response.into_body(), 16 * 1024)
        .await
        .expect("the chat body should be readable");
    let response_document: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the chat body should be JSON");
    assert_eq!(
        response_document["choices"][0]["message"]["content"],
        "Two households, both alike in dignity."
    );
}

#[tokio::test]
async fn should_forward_enforced_regex_constraint_to_the_worker() {
    let executor = ScriptedChatExecutor::ready(
        MODEL_ID,
        vec![
            ChatGenerationStreamEvent::TextFragment("Romeo".to_owned()),
            ChatGenerationStreamEvent::Completed {
                prompt_token_count: 8,
                generated_token_count: 1,
                reasoning_token_count: 0,
                astro_ssd_prompt_cache_restored_token_count: 0,
                reason: ChatGenerationCompletionReason::EndOfSequence,
            },
        ],
    );
    let received_constraint = executor.received_structured_constraint_handle();
    let application = build_application(executor);

    let chat_response = application
        .oneshot(chat_completion_request(
            r#"{
                        "model":"astronomical/non-streaming-test-model",
                        "messages":[{"role":"user","content":"Who answers?"}],
                        "structured_outputs":{"regex":"Romeo|Juliet"},
                        "stream":false
                    }"#,
        ))
        .await
        .expect("the application should return a chat response");

    assert_eq!(chat_response.status(), StatusCode::OK);
    let received_constraint = received_constraint
        .lock()
        .expect("the structured constraint log should not be poisoned")
        .clone();
    assert_eq!(
        received_constraint,
        Some(ipc_protocol::StructuredGenerationConstraint::Regex {
            pattern: "Romeo|Juliet".to_owned(),
        })
    );
    let warning_header = chat_response.headers().get(header::WARNING);
    assert!(
        warning_header.is_none(),
        "an enforced extra-body constraint must not carry the unenforced degradation warning"
    );
}
