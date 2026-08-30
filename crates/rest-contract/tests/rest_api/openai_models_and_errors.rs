use astronomical_rest_contract::{
    OpenAiErrorResponse, OpenAiImageModelParts, OpenAiModel, OpenAiModelList, OpenAiModelParts,
};

#[test]
fn should_serialize_complete_ready_model_capability_metadata_without_losing_openai_fields() {
    let model_list = OpenAiModelList::single_model(OpenAiModelParts {
        model_id: "astronomical/fake-mixture-of-experts".to_owned(),
        created: 1_784_231_803,
        owned_by: "astronomical".to_owned(),
        context_window: 262_144,
        max_input_tokens: 241_664,
        max_output_tokens: 20_480,
        input_modalities: vec!["text".to_owned(), "image".to_owned()],
        output_modalities: vec!["text".to_owned()],
        supports_streaming: true,
        supports_reasoning: true,
        reasoning_format: Some(
            "openai_chat_reasoning_content_and_responses_reasoning_summary_text".to_owned(),
        ),
        supports_tool_calls: true,
        tool_call_format: Some("openai_function_call".to_owned()),
        supported_endpoints: vec![
            "/v1/chat/completions".to_owned(),
            "/v1/responses".to_owned(),
        ],
    })
    .expect("complete model capability metadata should be valid");

    assert_eq!(
        serde_json::to_string(&model_list).expect("the OpenAI model list should serialize"),
        r#"{"object":"list","data":[{"id":"astronomical/fake-mixture-of-experts","object":"model","created":1784231803,"owned_by":"astronomical","context_window":262144,"max_input_tokens":241664,"max_output_tokens":20480,"input_modalities":["text","image"],"output_modalities":["text"],"supports_streaming":true,"supports_reasoning":true,"reasoning_format":"openai_chat_reasoning_content_and_responses_reasoning_summary_text","supports_tool_calls":true,"tool_call_format":"openai_function_call","supported_endpoints":["/v1/chat/completions","/v1/responses"]}]}"#
    );
}

#[test]
fn should_omit_reasoning_and_tool_formats_when_model_does_not_support_them() {
    let model_list = OpenAiModelList::single_model(OpenAiModelParts {
        model_id: "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit".to_owned(),
        created: 1_784_231_803,
        owned_by: "astronomical".to_owned(),
        context_window: 131_072,
        max_input_tokens: 110_592,
        max_output_tokens: 20_480,
        input_modalities: vec!["text".to_owned()],
        output_modalities: vec!["text".to_owned()],
        supports_streaming: true,
        supports_reasoning: false,
        reasoning_format: None,
        supports_tool_calls: false,
        tool_call_format: None,
        supported_endpoints: vec![
            "/v1/chat/completions".to_owned(),
            "/v1/responses".to_owned(),
        ],
    })
    .expect("text-only model capability metadata should be valid");

    assert_eq!(
        serde_json::to_string(&model_list).expect("the OpenAI model list should serialize"),
        r#"{"object":"list","data":[{"id":"mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit","object":"model","created":1784231803,"owned_by":"astronomical","context_window":131072,"max_input_tokens":110592,"max_output_tokens":20480,"input_modalities":["text"],"output_modalities":["text"],"supports_streaming":true,"supports_reasoning":false,"supports_tool_calls":false,"supported_endpoints":["/v1/chat/completions","/v1/responses"]}]}"#
    );
}

#[test]
fn should_accept_independent_input_and_output_maxima_that_share_one_context_window() {
    let model_list_result = OpenAiModelList::single_model(OpenAiModelParts {
        model_id: "astronomical/independent-token-limit-model".to_owned(),
        created: 1_784_231_803,
        owned_by: "astronomical".to_owned(),
        context_window: 10,
        max_input_tokens: 9,
        max_output_tokens: 9,
        input_modalities: vec!["text".to_owned()],
        output_modalities: vec!["text".to_owned()],
        supports_streaming: true,
        supports_reasoning: false,
        reasoning_format: None,
        supports_tool_calls: false,
        tool_call_format: None,
        supported_endpoints: vec!["/v1/chat/completions".to_owned()],
    });

    assert!(
        model_list_result.is_ok(),
        "independent maxima must remain valid because request admission enforces the shared context"
    );
}

#[test]
fn should_reject_an_independent_maximum_that_leaves_no_position_for_the_other_side() {
    let model_list_result = OpenAiModelList::single_model(OpenAiModelParts {
        model_id: "astronomical/exhausted-token-limit-model".to_owned(),
        created: 1_784_231_803,
        owned_by: "astronomical".to_owned(),
        context_window: 10,
        max_input_tokens: 10,
        max_output_tokens: 1,
        input_modalities: vec!["text".to_owned()],
        output_modalities: vec!["text".to_owned()],
        supports_streaming: true,
        supports_reasoning: false,
        reasoning_format: None,
        supports_tool_calls: false,
        tool_call_format: None,
        supported_endpoints: vec!["/v1/chat/completions".to_owned()],
    });

    assert!(
        matches!(
            model_list_result,
            Err(astronomical_rest_contract::OpenAiModelValidationError::InputTokenBudgetMustLeaveGenerationPosition {
                max_input_tokens: 10,
                context_window: 10,
            })
        ),
        "each independent maximum must leave one context position for the other side"
    );
}

#[test]
fn should_reject_an_output_maximum_that_leaves_no_prompt_position() {
    let model_list_result = OpenAiModelList::single_model(OpenAiModelParts {
        model_id: "astronomical/exhausted-output-limit-model".to_owned(),
        created: 1_784_231_803,
        owned_by: "astronomical".to_owned(),
        context_window: 10,
        max_input_tokens: 1,
        max_output_tokens: 10,
        input_modalities: vec!["text".to_owned()],
        output_modalities: vec!["text".to_owned()],
        supports_streaming: true,
        supports_reasoning: false,
        reasoning_format: None,
        supports_tool_calls: false,
        tool_call_format: None,
        supported_endpoints: vec!["/v1/chat/completions".to_owned()],
    });

    assert!(
        matches!(
            model_list_result,
            Err(astronomical_rest_contract::OpenAiModelValidationError::OutputTokenBudgetMustLeavePromptPosition {
                max_output_tokens: 10,
                context_window: 10,
            })
        ),
        "the output maximum must leave one context position for prompt input"
    );
}

#[test]
fn should_advertise_an_image_output_model_without_fake_token_limits() {
    let image_model = OpenAiModel::from_image_parts(OpenAiImageModelParts {
        model_id: "black-forest-labs/FLUX.2-klein-4B".to_owned(),
        created: 1_787_010_400,
        owned_by: "astronomical".to_owned(),
        input_modalities: vec!["text".to_owned()],
        output_modalities: vec!["image".to_owned()],
        supported_endpoints: vec!["/v1/images/generations".to_owned()],
    })
    .expect("image generation metadata should not require token limits");
    let model_list = OpenAiModelList::from_models(vec![image_model]);

    assert_eq!(
        serde_json::to_string(&model_list).expect("the image model metadata should serialize"),
        r#"{"object":"list","data":[{"id":"black-forest-labs/FLUX.2-klein-4B","object":"model","created":1787010400,"owned_by":"astronomical","input_modalities":["text"],"output_modalities":["image"],"supports_streaming":false,"supports_reasoning":false,"supports_tool_calls":false,"supported_endpoints":["/v1/images/generations"]}]}"#
    );
}

#[test]
fn should_omit_capability_fields_when_no_worker_is_ready() {
    let model_list = OpenAiModelList::empty();

    let serialized_model_list =
        serde_json::to_string(&model_list).expect("the OpenAI model list should serialize");
    assert_eq!(serialized_model_list, r#"{"object":"list","data":[]}"#);
    assert!(!serialized_model_list.contains("context_window"));
    assert!(!serialized_model_list.contains("max_input_tokens"));
    assert!(!serialized_model_list.contains("max_output_tokens"));
    assert!(!serialized_model_list.contains("supports_reasoning"));
    assert!(!serialized_model_list.contains("supports_tool_calls"));
    assert!(!serialized_model_list.contains("supports_streaming"));
    assert!(!serialized_model_list.contains("input_modalities"));
    assert!(!serialized_model_list.contains("output_modalities"));
    assert!(!serialized_model_list.contains("reasoning_format"));
    assert!(!serialized_model_list.contains("tool_call_format"));
    assert!(!serialized_model_list.contains("supported_endpoints"));
}

#[test]
fn should_serialize_an_empty_openai_model_list_when_no_worker_is_ready() {
    let model_list = OpenAiModelList::empty();

    assert_eq!(
        serde_json::to_string(&model_list).expect("the OpenAI model list should serialize"),
        r#"{"object":"list","data":[]}"#
    );
}

#[test]
fn should_serialize_a_standard_openai_invalid_request_error() {
    let error_response = OpenAiErrorResponse::invalid_request(
        "model is not loaded by the local worker",
        Some("model"),
        Some("model_not_found"),
    );

    assert_eq!(
        serde_json::to_string(&error_response).expect("the OpenAI error response should serialize"),
        r#"{"error":{"message":"model is not loaded by the local worker","type":"invalid_request_error","param":"model","code":"model_not_found"}}"#
    );
}
