use super::*;

#[tokio::test]
async fn should_return_a_non_streaming_response_from_the_public_endpoint() {
    let application = build_application(ScriptedResponsesExecutor::new(vec![
        ChatGenerationStreamEvent::TextFragment("Done.".to_owned()),
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 10,
            generated_token_count: 2,
            reasoning_token_count: 0,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        },
    ]));
    let http_response = application
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(format!(
                    r#"{{"model":"{MODEL_ID}","input":"hello","stream":false}}"#
                )))
                .expect("the Responses request should be valid"),
        )
        .await
        .expect("the application should return an HTTP response");

    assert_eq!(http_response.status(), StatusCode::OK);
    let response_body = to_bytes(http_response.into_body(), 16 * 1024)
        .await
        .expect("the Responses body should be readable");
    let response_document: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the response should be JSON");
    assert_eq!(response_document["object"], "response");
    assert_eq!(response_document["status"], "completed");
    assert_eq!(response_document["output_text"], "Done.");
    // Response metadata echoes caller intent; runtime defaults are not rewritten as explicit input.
    assert_eq!(response_document["temperature"], serde_json::Value::Null);
    assert_eq!(response_document["top_p"], serde_json::Value::Null);
    assert_eq!(
        response_document["max_output_tokens"],
        serde_json::Value::Null
    );
}

#[tokio::test]
async fn should_echo_validated_request_configuration_in_the_response() {
    let application = build_application(ScriptedResponsesExecutor::new(vec![
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 4,
            generated_token_count: 1,
            reasoning_token_count: 0,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        },
    ]));
    let request_body = format!(
        r#"{{
            "model":"{MODEL_ID}",
            "input":"hello",
            "metadata":{{"task":"edit"}},
            "temperature":0.5,
            "top_p":0.8,
            "max_output_tokens":64,
            "tool_choice":"none",
            "tools":[{{
                "type":"function",
                "name":"read_file",
                "description":"Read one file",
                "parameters":{{"type":"object"}}
            }}]
        }}"#
    );

    let http_response = post_response_body(application, &request_body).await;

    assert_eq!(http_response.status(), StatusCode::OK);
    let response_body = to_bytes(http_response.into_body(), 16 * 1024)
        .await
        .expect("the Responses body should be readable");
    let response_document: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the response should be JSON");
    assert_eq!(response_document["metadata"]["task"], "edit");
    assert_eq!(response_document["temperature"], 0.5);
    assert_eq!(response_document["top_p"], 0.8);
    assert_eq!(response_document["max_output_tokens"], 64);
    assert_eq!(response_document["tool_choice"], "none");
    assert_eq!(response_document["tools"][0]["type"], "function");
    assert_eq!(response_document["tools"][0]["name"], "read_file");
    assert_eq!(
        response_document["tools"][0]["parameters"]["type"],
        "object"
    );
}

#[tokio::test]
async fn should_return_non_streaming_responses_reasoning_without_visible_output_text() {
    let application = build_application(ScriptedResponsesExecutor::new(vec![
        ChatGenerationStreamEvent::ReasoningFragment("internal thought".to_owned()),
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 10,
            generated_token_count: 2,
            reasoning_token_count: 2,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        },
    ]));

    let http_response = post_response(application, false).await;

    assert_eq!(http_response.status(), StatusCode::OK);
    let response_body = to_bytes(http_response.into_body(), 32 * 1024)
        .await
        .expect("the Responses body should be readable");
    let response_document: Value =
        serde_json::from_slice(&response_body).expect("the response should be JSON");
    assert_eq!(response_document["output_text"], "");
    assert_eq!(response_document["output"][0]["type"], "reasoning");
    assert_eq!(
        response_document["output"][0]["summary"][0],
        serde_json::json!({"type":"summary_text","text":"internal thought"})
    );
    assert!(
        response_document["output"]
            .as_array()
            .expect("output should be an array")
            .iter()
            .all(|output_item_document| output_item_document["type"] != "message"),
        "reasoning-only responses must not fabricate a visible message: {response_document}"
    );
}

#[tokio::test]
async fn should_return_extracted_json_and_a_warning_for_text_format_json_schema() {
    let application = build_application(ScriptedResponsesExecutor::new(vec![
        ChatGenerationStreamEvent::TextFragment(
            "```json\n{\"speaker\":\"Juliet\",\"play\":\"Romeo and Juliet\"}\n```".to_owned(),
        ),
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 12,
            generated_token_count: 8,
            reasoning_token_count: 0,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        },
    ]));
    let http_response = post_response_body(
        application,
        r#"{
            "model":"astronomical/responses-endpoint-test-model",
            "input":"O Romeo, Romeo, wherefore art thou Romeo?",
            "text":{"format":{"type":"json_schema","name":"romeo_line","schema":{"type":"object"}}},
            "stream":false
        }"#,
    )
    .await;

    assert_eq!(http_response.status(), StatusCode::OK);
    let warning_header = http_response
        .headers()
        .get(header::WARNING)
        .and_then(|header_value| header_value.to_str().ok())
        .expect("unenforced json_schema must disclose a Warning header");
    assert_eq!(
        warning_header,
        astronomical_rest_contract::UNENFORCED_RESPONSE_FORMAT_WARNING
    );
    let response_body = to_bytes(http_response.into_body(), 16 * 1024)
        .await
        .expect("the Responses body should be readable");
    let response_document: Value =
        serde_json::from_slice(&response_body).expect("the response should be JSON");
    let extracted_json: Value = serde_json::from_str(
        response_document["output_text"]
            .as_str()
            .expect("output_text should be a string"),
    )
    .expect("output_text should be JSON");
    assert_eq!(
        extracted_json,
        serde_json::json!({"speaker": "Juliet", "play": "Romeo and Juliet"})
    );
}
