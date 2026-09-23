use super::*;
use astronomical_rest_contract::{OpenAiChatCompletionRequestParts, ThinkingControlsError};

#[test]
fn should_resolve_the_coding_agent_thinking_budget_spelling_alone() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "thinking_token_budget": 96
    }
    "#;
    let request_parts = parse_chat_request_parts(request_json);
    assert_eq!(request_parts.thinking_budget, Some(96));
}

#[test]
fn should_resolve_the_documented_thinking_budget_alias_alone() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "thinking_budget_tokens": 64
    }
    "#;
    let request_parts = parse_chat_request_parts(request_json);
    assert_eq!(request_parts.thinking_budget, Some(64));
}

#[test]
fn should_accept_agreeing_thinking_budget_spellings() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "thinking_budget": 32,
        "thinking_token_budget": 32,
        "thinking_budget_tokens": 32
    }
    "#;
    let request_parts = parse_chat_request_parts(request_json);
    assert_eq!(request_parts.thinking_budget, Some(32));
}

#[test]
fn should_reject_disagreeing_thinking_budget_spellings() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "thinking_budget": 32,
        "thinking_token_budget": 96
    }
    "#;
    let chat_completion_request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("disagreeing budget spellings should decode");

    let validation_error = chat_completion_request
        .validate()
        .expect_err("disagreeing budget spellings must fail loudly");

    assert_eq!(
        validation_error,
        OpenAiChatCompletionValidationError::ThinkingControls(
            ThinkingControlsError::ConflictingNumericThinkingBudgets {
                thinking_budget: Some(32),
                thinking_token_budget: Some(96),
                thinking_budget_tokens: None,
                reasoning_max_tokens: None,
            }
        )
    );
}

#[test]
fn should_map_the_reasoning_effort_levels_to_thinking_budgets() {
    for (reasoning_effort, expected_budget) in [
        ("minimal", 1024),
        ("low", 2048),
        ("medium", 8192),
        ("high", 16384),
        ("xhigh", 16384),
        ("max", 16384),
    ] {
        let request_json = format!(
            r#"{{
                "model": "astronomical/fake-mixture-of-experts",
                "messages": [{{"role": "user", "content": "write a function"}}],
                "reasoning_effort": "{reasoning_effort}"
            }}"#,
        );
        let request_parts = parse_chat_request_parts(&request_json);
        assert_eq!(
            request_parts.thinking_budget,
            Some(expected_budget),
            "reasoning_effort {reasoning_effort} must enforce its token budget"
        );
    }
}

#[test]
fn should_prefer_the_explicit_thinking_budget_over_reasoning_effort() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "thinking_budget": 300,
        "reasoning_effort": "high"
    }
    "#;
    let request_parts = parse_chat_request_parts(request_json);
    assert_eq!(request_parts.thinking_budget, Some(300));
}

#[test]
fn should_reject_an_unknown_reasoning_effort_label() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning_effort": "turbo"
    }
    "#;
    let chat_completion_request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("an unknown effort label should still decode");

    let validation_error = chat_completion_request
        .validate()
        .expect_err("an unknown reasoning effort must fail loudly instead of being dropped");

    assert_eq!(
        validation_error,
        OpenAiChatCompletionValidationError::ThinkingControls(
            ThinkingControlsError::UnknownReasoningEffort {
                reasoning_effort: "turbo".to_owned(),
            }
        )
    );
}

#[test]
fn should_disable_thinking_for_off_reasoning_effort() {
    for reasoning_effort in ["off", "none"] {
        let request_json = format!(
            r#"{{
                "model": "astronomical/fake-mixture-of-experts",
                "messages": [{{"role": "user", "content": "write a function"}}],
                "reasoning_effort": "{reasoning_effort}"
            }}"#,
        );
        let request_parts = parse_chat_request_parts(&request_json);
        assert_eq!(
            request_parts.thinking_budget,
            Some(0),
            "reasoning_effort {reasoning_effort} must close the thinking channel"
        );
    }
}

#[test]
fn should_resolve_the_reasoning_object_effort_and_max_tokens() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning": {"effort": "low"}
    }
    "#;
    let request_parts = parse_chat_request_parts(request_json);
    assert_eq!(request_parts.thinking_budget, Some(2048));

    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning": {"max_tokens": 5000}
    }
    "#;
    let request_parts = parse_chat_request_parts(request_json);
    assert_eq!(request_parts.thinking_budget, Some(5000));

    // reasoning.max_tokens must agree with the top-level numeric spellings.
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "thinking_budget": 5000,
        "reasoning": {"max_tokens": 5000}
    }
    "#;
    let request_parts = parse_chat_request_parts(request_json);
    assert_eq!(request_parts.thinking_budget, Some(5000));

    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "thinking_budget": 96,
        "reasoning": {"max_tokens": 5000}
    }
    "#;
    let chat_completion_request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("a numeric disagreement should decode");
    assert!(matches!(
        chat_completion_request.validate(),
        Err(OpenAiChatCompletionValidationError::ThinkingControls(
            ThinkingControlsError::ConflictingNumericThinkingBudgets { .. }
        ))
    ));
}

#[test]
fn should_disable_thinking_from_the_reasoning_object_enabled_flag() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning": {"enabled": false}
    }
    "#;
    let request_parts = parse_chat_request_parts(request_json);
    assert_eq!(request_parts.thinking_budget, Some(0));
}

#[test]
fn should_flag_reasoning_exclusion_without_changing_the_budget() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning": {"max_tokens": 5000, "exclude": true}
    }
    "#;
    let request_parts = parse_chat_request_parts(request_json);
    assert_eq!(request_parts.thinking_budget, Some(5000));
    assert!(request_parts.reasoning_excluded);

    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}]
    }
    "#;
    let request_parts = parse_chat_request_parts(request_json);
    assert!(!request_parts.reasoning_excluded);
}

#[test]
fn should_accept_the_flat_enable_thinking_flag_and_chat_template_kwargs() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "enable_thinking": false
    }
    "#;
    let request_parts = parse_chat_request_parts(request_json);
    assert_eq!(request_parts.thinking_budget, Some(0));

    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "enable_thinking": true,
        "chat_template_kwargs": {"enable_thinking": true},
        "reasoning": {"effort": "medium"}
    }
    "#;
    let request_parts = parse_chat_request_parts(request_json);
    assert_eq!(request_parts.thinking_budget, Some(8192));

    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "chat_template_kwargs": {"enable_thinking": false}
    }
    "#;
    let request_parts = parse_chat_request_parts(request_json);
    assert_eq!(request_parts.thinking_budget, Some(0));

    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "enable_thinking": true,
        "chat_template_kwargs": {"enable_thinking": false}
    }
    "#;
    let chat_completion_request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("disagreeing flags should decode");
    assert!(matches!(
        chat_completion_request.validate(),
        Err(OpenAiChatCompletionValidationError::ThinkingControls(
            ThinkingControlsError::ConflictingThinkingEnableFlags { .. }
        ))
    ));
}

#[test]
fn should_prefer_an_explicit_disable_over_a_level_name() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning": {"effort": "none"},
        "reasoning_effort": "high"
    }
    "#;
    let request_parts = parse_chat_request_parts(request_json);
    assert_eq!(request_parts.thinking_budget, Some(0));
}

#[test]
fn should_reject_disagreeing_effort_level_names() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning": {"effort": "low"},
        "reasoning_effort": "high"
    }
    "#;
    let chat_completion_request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("disagreeing levels should decode");
    assert!(matches!(
        chat_completion_request.validate(),
        Err(OpenAiChatCompletionValidationError::ThinkingControls(
            ThinkingControlsError::ConflictingReasoningEfforts { .. }
        ))
    ));
}

#[test]
fn should_reject_disabled_thinking_with_a_positive_budget() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "enable_thinking": false,
        "thinking_budget": 5000
    }
    "#;
    let chat_completion_request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("disable plus budget should decode");
    assert!(matches!(
        chat_completion_request.validate(),
        Err(OpenAiChatCompletionValidationError::ThinkingControls(
            ThinkingControlsError::ThinkingDisabledWhileBudgetRequested { .. }
        ))
    ));
}

#[test]
fn should_reject_unknown_subfields_of_the_reasoning_and_template_kwarg_objects() {
    for unknown_subfield in [
        "{\"effort\": \"low\", \"bogus\": 1}",
        "{\"max_tokens\": 1, \"bogus\": 1}",
    ] {
        let request_json = format!(
            r#"{{
                "model": "astronomical/fake-mixture-of-experts",
                "messages": [{{"role": "user", "content": "write a function"}}],
                "reasoning": {unknown_subfield}
            }}"#
        );
        assert!(
            serde_json::from_str::<OpenAiChatCompletionRequest>(&request_json).is_err(),
            "an unknown reasoning subfield must fail loudly: {request_json}"
        );
    }

    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "chat_template_kwargs": {"enable_thinking": true, "temperature": 0.1}
    }
    "#;
    assert!(
        serde_json::from_str::<OpenAiChatCompletionRequest>(request_json).is_err(),
        "unknown chat_template_kwargs entries must fail loudly"
    );
}

fn parse_chat_request_parts(request_json: &str) -> OpenAiChatCompletionRequestParts {
    let chat_completion_request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .unwrap_or_else(|decode_error| panic!("request should decode: {decode_error}"));
    chat_completion_request
        .into_parts()
        .unwrap_or_else(|validation_error| panic!("request should validate: {validation_error}"))
}
