use astronomical_rest_contract::{
    OpenAiResponsesRequest, OpenAiResponsesRequestParts, OpenAiResponsesValidationError,
    ThinkingControlsError,
};

#[test]
fn should_resolve_the_reasoning_object_max_tokens_into_the_thinking_budget() {
    let request_parts = parse_responses_request_parts(
        r#"{
            "model": "astronomical/fake-mixture-of-experts",
            "input": "Summarize this conversation.",
            "reasoning": {"max_tokens": 5000}
        }"#,
    );
    assert_eq!(request_parts.thinking_budget, Some(5000));
}

#[test]
fn should_resolve_the_reasoning_object_effort_levels() {
    for (effort, expected_budget) in [
        ("minimal", 1024),
        ("low", 2048),
        ("medium", 8192),
        ("high", 16384),
        ("xhigh", 16384),
        ("max", 16384),
    ] {
        let request_parts = parse_responses_request_parts(&format!(
            r#"{{
                "model": "astronomical/fake-mixture-of-experts",
                "input": "Summarize this conversation.",
                "reasoning": {{"effort": "{effort}"}}
            }}"#
        ));
        assert_eq!(
            request_parts.thinking_budget,
            Some(expected_budget),
            "reasoning.effort {effort} must enforce its token budget"
        );
    }
}

#[test]
fn should_disable_thinking_from_every_disable_spelling() {
    for disable_spelling in [
        r#""reasoning": {"effort": "none"}"#,
        r#""reasoning": {"enabled": false}"#,
        r#""enable_thinking": false"#,
        r#""chat_template_kwargs": {"enable_thinking": false}"#,
        r#""reasoning_effort": "off""#,
    ] {
        let request_parts = parse_responses_request_parts(&format!(
            r#"{{
                "model": "astronomical/fake-mixture-of-experts",
                "input": "Summarize this conversation.",
                {disable_spelling}
            }}"#
        ));
        assert_eq!(
            request_parts.thinking_budget,
            Some(0),
            "disable spelling {disable_spelling} must close the thinking channel"
        );
    }
}

#[test]
fn should_flag_reasoning_exclusion_without_changing_the_budget() {
    let request_parts = parse_responses_request_parts(
        r#"{
            "model": "astronomical/fake-mixture-of-experts",
            "input": "Summarize this conversation.",
            "reasoning": {"max_tokens": 5000, "exclude": true}
        }"#,
    );
    assert_eq!(request_parts.thinking_budget, Some(5000));
    assert!(request_parts.reasoning_excluded);

    let request_parts = parse_responses_request_parts(
        r#"{
            "model": "astronomical/fake-mixture-of-experts",
            "input": "Summarize this conversation."
        }"#,
    );
    assert!(!request_parts.reasoning_excluded);
}

#[test]
fn should_prefer_the_top_level_numeric_budget_over_reasoning_max_tokens_conflict_free() {
    let request_parts = parse_responses_request_parts(
        r#"{
            "model": "astronomical/fake-mixture-of-experts",
            "input": "Summarize this conversation.",
            "thinking_budget": 5000,
            "reasoning": {"max_tokens": 5000, "exclude": true}
        }"#,
    );
    assert_eq!(request_parts.thinking_budget, Some(5000));
    assert!(request_parts.reasoning_excluded);
}

#[test]
fn should_reject_disagreeing_numeric_and_level_spellings() {
    // numeric vs reasoning.max_tokens
    let validation_error = validate_responses_request(
        r#"{
            "model": "astronomical/fake-mixture-of-experts",
            "input": "hello",
            "thinking_budget": 96,
            "reasoning": {"max_tokens": 5000}
        }"#,
    );
    assert!(matches!(
        validation_error,
        OpenAiResponsesValidationError::ThinkingControls(
            ThinkingControlsError::ConflictingNumericThinkingBudgets { .. }
        )
    ));

    // reasoning_effort vs reasoning.effort
    let validation_error = validate_responses_request(
        r#"{
            "model": "astronomical/fake-mixture-of-experts",
            "input": "hello",
            "reasoning_effort": "high",
            "reasoning": {"effort": "low"}
        }"#,
    );
    assert!(matches!(
        validation_error,
        OpenAiResponsesValidationError::ThinkingControls(
            ThinkingControlsError::ConflictingReasoningEfforts { .. }
        )
    ));

    // enable flags
    let validation_error = validate_responses_request(
        r#"{
            "model": "astronomical/fake-mixture-of-experts",
            "input": "hello",
            "enable_thinking": true,
            "reasoning": {"enabled": false}
        }"#,
    );
    assert!(matches!(
        validation_error,
        OpenAiResponsesValidationError::ThinkingControls(
            ThinkingControlsError::ConflictingThinkingEnableFlags { .. }
        )
    ));

    // disable with positive budget
    let validation_error = validate_responses_request(
        r#"{
            "model": "astronomical/fake-mixture-of-experts",
            "input": "hello",
            "enable_thinking": false,
            "thinking_budget": 5000
        }"#,
    );
    assert!(matches!(
        validation_error,
        OpenAiResponsesValidationError::ThinkingControls(
            ThinkingControlsError::ThinkingDisabledWhileBudgetRequested { .. }
        )
    ));
}

#[test]
fn should_reject_unknown_subfields_of_the_reasoning_object() {
    for unknown_subfield in ["{\"effort\": \"low\", \"bogus\": 1}", "{\"bogus\": true}"] {
        let decode_result = serde_json::from_str::<OpenAiResponsesRequest>(&format!(
            r#"{{
                "model": "astronomical/fake-mixture-of-experts",
                "input": "hello",
                "reasoning": {unknown_subfield}
            }}"#
        ));
        assert!(
            decode_result.is_err(),
            "an unknown reasoning subfield must fail loudly: {unknown_subfield}"
        );
    }

    let decode_result = serde_json::from_str::<OpenAiResponsesRequest>(
        r#"{
            "model": "astronomical/fake-mixture-of-experts",
            "input": "hello",
            "chat_template_kwargs": {"enable_thinking": true, "bogus": 1}
        }"#,
    );
    assert!(
        decode_result.is_err(),
        "unknown chat_template_kwargs entries must fail loudly"
    );
}

#[test]
fn should_reject_an_unknown_reasoning_effort_label() {
    let validation_error = validate_responses_request(
        r#"{
            "model": "astronomical/fake-mixture-of-experts",
            "input": "hello",
            "reasoning_effort": "turbo"
        }"#,
    );
    assert_eq!(
        validation_error,
        OpenAiResponsesValidationError::ThinkingControls(
            ThinkingControlsError::UnknownReasoningEffort {
                reasoning_effort: "turbo".to_owned(),
            }
        )
    );
}

fn parse_responses_request_parts(request_json: &str) -> OpenAiResponsesRequestParts {
    let responses_request = serde_json::from_str::<OpenAiResponsesRequest>(request_json)
        .unwrap_or_else(|decode_error| panic!("request should decode: {decode_error}"));
    responses_request
        .into_parts()
        .unwrap_or_else(|validation_error| panic!("request should validate: {validation_error}"))
}

fn validate_responses_request(request_json: &str) -> OpenAiResponsesValidationError {
    let responses_request = serde_json::from_str::<OpenAiResponsesRequest>(request_json)
        .unwrap_or_else(|decode_error| panic!("request should decode: {decode_error}"));
    responses_request
        .into_parts()
        .expect_err("conflicting or invalid spellings must fail loudly")
}
