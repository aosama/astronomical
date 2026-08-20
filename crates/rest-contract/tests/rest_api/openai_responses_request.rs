use astronomical_rest_contract::{OpenAiResponseInputParts, OpenAiResponsesRequest};

#[test]
fn should_parse_a_non_streaming_response_request_with_string_input() {
    let request = serde_json::from_str::<OpenAiResponsesRequest>(
        r#"{
            "model": "mlx-community/Ornith-1.0-35B-OptiQ-4bit",
            "input": "Explain the repository.",
            "max_output_tokens": 512,
            "temperature": 0.6,
            "top_p": 0.95
        }"#,
    )
    .expect("the minimal Responses request should deserialize");

    let request_parts = request
        .into_parts()
        .expect("the minimal Responses request should validate");

    assert_eq!(
        request_parts.model,
        "mlx-community/Ornith-1.0-35B-OptiQ-4bit"
    );
    assert_eq!(
        request_parts.input,
        OpenAiResponseInputParts::Text("Explain the repository.".to_owned())
    );
    assert_eq!(request_parts.maximum_output_tokens, 512);
    assert_eq!(request_parts.requested_maximum_output_tokens, Some(512));
    assert_eq!(request_parts.temperature, Some(0.6));
    assert_eq!(request_parts.top_p, Some(0.95));
    assert!(!request_parts.stream);
}

#[test]
fn should_preserve_omitted_responses_generation_settings_and_request_only_metadata() {
    let request = serde_json::from_str::<OpenAiResponsesRequest>(
        r#"{"model":"organization/model","input":"hello"}"#,
    )
    .expect("the request without generation settings should deserialize");

    let request_parts = request
        .into_parts()
        .expect("the request without generation settings should validate");
    let response_configuration = request_parts.response_configuration();

    assert_eq!(request_parts.requested_maximum_output_tokens, None);
    assert_eq!(request_parts.temperature, None);
    assert_eq!(request_parts.top_p, None);
    assert_eq!(response_configuration.max_output_tokens, None);
    assert_eq!(response_configuration.temperature, None);
    assert_eq!(response_configuration.top_p, None);
}

#[test]
fn should_preserve_ordered_response_items_for_manual_function_loop_replay() {
    let request = serde_json::from_str::<OpenAiResponsesRequest>(
        r#"{
            "model": "mlx-community/Ornith-1.0-35B-OptiQ-4bit",
            "instructions": "Be precise.",
            "input": [
                {"role":"user","content":[{"type":"input_text","text":"Inspect the repository."}]},
                {"type":"reasoning","id":"rs_prior","summary":[],"content":[{"type":"reasoning_text","text":"I should inspect files."}]},
                {"type":"message","id":"msg_prior","role":"assistant","status":"completed","content":[{"type":"output_text","text":"I will inspect it.","annotations":[],"logprobs":[]}]},
                {"type":"function_call","id":"fc_prior","call_id":"call_prior","name":"glob","arguments":"{\"pattern\":\"**/*.rs\"}","status":"completed"},
                {"type":"function_call_output","call_id":"call_prior","output":"src/lib.rs"}
            ]
        }"#,
    )
    .expect("the manual replay request should deserialize");

    let request_parts = request
        .into_parts()
        .expect("Astronomical-produced response items should be replayable");

    assert_eq!(request_parts.instructions.as_deref(), Some("Be precise."));
    let OpenAiResponseInputParts::Items(response_input_items) = request_parts.input else {
        panic!("expected ordered response input items");
    };
    assert_eq!(response_input_items.len(), 5);
    assert_eq!(response_input_items[0].kind_name(), "user_message");
    assert_eq!(response_input_items[1].kind_name(), "reasoning");
    assert_eq!(response_input_items[2].kind_name(), "assistant_message");
    assert_eq!(response_input_items[3].kind_name(), "function_call");
    assert_eq!(response_input_items[4].kind_name(), "function_call_output");
}

#[test]
fn should_decode_a_responses_data_uri_image_in_user_content_order() {
    let red_pixel_png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";
    let request_json = format!(
        r#"{{
            "model":"mlx-community/Ornith-1.0-35B-OptiQ-4bit",
            "input":[{{
                "role":"user",
                "content":[
                    {{"type":"input_text","text":"Describe this image."}},
                    {{"type":"input_image","image_url":"data:image/png;base64,{red_pixel_png_base64}","detail":"auto"}}
                ]
            }}]
        }}"#
    );
    let request = serde_json::from_str::<OpenAiResponsesRequest>(&request_json)
        .expect("the Responses image request should deserialize");

    let request_parts = request
        .into_parts()
        .expect("the bounded data-URI image should validate");

    let OpenAiResponseInputParts::Items(response_input_items) = request_parts.input else {
        panic!("expected response input items");
    };
    let astronomical_rest_contract::OpenAiResponseInputItemParts::UserMessage { content, images } =
        &response_input_items[0]
    else {
        panic!("expected a user message");
    };
    assert_eq!(content, "Describe this image.");
    assert_eq!(images.len(), 1);
    assert_eq!(images[0].mime_type(), "image/png");
}

#[test]
fn should_accept_native_function_tools_and_harmless_compatibility_fields() {
    let request = serde_json::from_str::<OpenAiResponsesRequest>(
        r#"{
            "model":"mlx-community/Ornith-1.0-35B-OptiQ-4bit",
            "input":"List Rust files.",
            "tools":[{
                "type":"function",
                "name":"glob",
                "description":"List matching files.",
                "parameters":{"type":"object","properties":{"pattern":{"type":"string"}}},
                "strict":false
            }],
            "tool_choice":"none",
            "metadata":{"session":"local"},
            "store":false,
            "background":false,
            "truncation":"disabled",
            "service_tier":"auto",
            "user":"single-user",
            "safety_identifier":"local-user",
            "prompt_cache_key":"session-prefix"
        }"#,
    )
    .expect("the broad local Responses request should deserialize");

    let request_parts = request
        .into_parts()
        .expect("harmless compatibility fields should be accepted");

    assert_eq!(request_parts.tools.len(), 1);
    assert_eq!(request_parts.tools[0].name, "glob");
    assert_eq!(request_parts.tool_choice.kind_name(), "none");
    assert_eq!(
        request_parts.metadata.get("session").map(String::as_str),
        Some("local")
    );
}

#[test]
fn should_parse_recognized_behavior_changing_fields_before_rejecting_them() {
    let unsupported_requests = [
        (
            r#"{"model":"ornith","input":"hello","previous_response_id":"resp_prior"}"#,
            "previous_response_id",
        ),
        (
            r#"{"model":"ornith","input":"hello","text":{"format":{"type":"json_schema","name":"answer","schema":{"type":"object"}}}}"#,
            "text",
        ),
        (
            r#"{"model":"ornith","input":"hello","tools":[{"type":"web_search"}]}"#,
            "tools[].type",
        ),
    ];

    for (request_json, expected_option_name) in unsupported_requests {
        let request = serde_json::from_str::<OpenAiResponsesRequest>(request_json)
            .expect("recognized Responses fields should deserialize before validation");
        let validation_error = request
            .into_parts()
            .expect_err("behavior-changing unsupported fields must be rejected");
        assert!(
            validation_error.to_string().contains(expected_option_name),
            "expected {expected_option_name} in {validation_error}"
        );
    }
}

#[test]
fn should_parse_a_foreign_response_item_before_rejecting_its_type() {
    let request = serde_json::from_str::<OpenAiResponsesRequest>(
        r#"{
            "model":"ornith",
            "input":[{"type":"file_search_call","id":"fs_1","queries":["docs"]}]
        }"#,
    )
    .expect("a recognized foreign item shape should reach semantic validation");

    let validation_error = request
        .into_parts()
        .expect_err("hosted file-search items are not locally executable");

    assert!(validation_error.to_string().contains("file_search_call"));
}

#[test]
fn should_accept_the_include_field_as_a_harmless_noop() {
    let request = serde_json::from_str::<OpenAiResponsesRequest>(
        r#"{
            "model":"ornith",
            "input":"hello",
            "include":["reasoning.encrypted_content"]
        }"#,
    )
    .expect("the include field should deserialize");

    let request_parts = request.into_parts().expect(
        "include is a harmless no-op because Astronomical never produces encrypted reasoning",
    );

    assert_eq!(request_parts.model, "ornith");
}

#[test]
fn should_accept_copilot_reasoning_effort_as_a_harmless_noop() {
    let request = serde_json::from_str::<OpenAiResponsesRequest>(
        r#"{
            "model":"ornith",
            "input":"hello",
            "reasoning":{"effort":"medium"}
        }"#,
    )
    .expect("the Responses reasoning field should deserialize");

    let request_parts = request
        .into_parts()
        .expect("the fixed local reasoning behavior should ignore Copilot's effort hint");

    assert_eq!(request_parts.model, "ornith");
}
