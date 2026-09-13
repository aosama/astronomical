use super::*;

#[test]
fn should_resolve_the_coding_agent_thinking_budget_spelling_alone() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "thinking_token_budget": 96
    }
    "#;
    let chat_completion_request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("the coding-agent budget spelling should decode");

    let request_parts = chat_completion_request
        .into_parts()
        .expect("an alias-only thinking budget should validate");

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
    let chat_completion_request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("the documented budget alias should decode");

    let request_parts = chat_completion_request
        .into_parts()
        .expect("an alias-only thinking budget should validate");

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
    let chat_completion_request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("agreeing budget spellings should decode");

    let request_parts = chat_completion_request
        .into_parts()
        .expect("agreeing budget spellings should validate");

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
        OpenAiChatCompletionValidationError::ConflictingThinkingBudgets {
            thinking_budget: Some(32),
            thinking_token_budget: Some(96),
            thinking_budget_tokens: None,
        }
    );
}

#[test]
fn should_map_the_openai_reasoning_effort_levels_to_thinking_budgets() {
    for (reasoning_effort, expected_budget) in [
        ("minimal", 1024),
        ("low", 2048),
        ("medium", 8192),
        ("high", 16384),
    ] {
        let request_json = format!(
            r#"{{
                "model": "astronomical/fake-mixture-of-experts",
                "messages": [{{"role": "user", "content": "write a function"}}],
                "reasoning_effort": "{reasoning_effort}"
            }}"#,
        );
        let chat_completion_request =
            serde_json::from_str::<OpenAiChatCompletionRequest>(&request_json)
                .expect("the reasoning effort level should decode");

        let request_parts =
            chat_completion_request
                .into_parts()
                .unwrap_or_else(|validation_error| {
                    panic!("reasoning_effort {reasoning_effort} should resolve: {validation_error}")
                });

        assert_eq!(
            request_parts.thinking_budget,
            Some(expected_budget),
            "reasoning_effort {reasoning_effort} must enforce its token budget"
        );
    }
}

#[test]
fn should_clamp_the_xhigh_reasoning_effort_to_the_high_budget() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning_effort": "xhigh"
    }
    "#;
    let chat_completion_request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("the xhigh reasoning effort should decode");

    let request_parts = chat_completion_request
        .into_parts()
        .expect("xhigh should clamp to the high budget");

    assert_eq!(request_parts.thinking_budget, Some(16384));
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
    let chat_completion_request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("a request with both budget spellings should decode");

    let request_parts = chat_completion_request
        .into_parts()
        .expect("the explicit budget must win over the level name");

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
        OpenAiChatCompletionValidationError::UnknownReasoningEffort {
            reasoning_effort: "turbo".to_owned(),
        }
    );
}

#[test]
fn should_accept_off_reasoning_effort_without_a_budget() {
    for reasoning_effort in ["off", "none"] {
        let request_json = format!(
            r#"{{
                "model": "astronomical/fake-mixture-of-experts",
                "messages": [{{"role": "user", "content": "write a function"}}],
                "reasoning_effort": "{reasoning_effort}"
            }}"#,
        );
        let chat_completion_request =
            serde_json::from_str::<OpenAiChatCompletionRequest>(&request_json)
                .expect("the opt-out effort label should decode");

        let request_parts = chat_completion_request
            .into_parts()
            .expect("opt-out effort labels must not fail validation");

        assert_eq!(
            request_parts.thinking_budget, None,
            "reasoning_effort {reasoning_effort} carries no enforceable budget today"
        );
    }
}
