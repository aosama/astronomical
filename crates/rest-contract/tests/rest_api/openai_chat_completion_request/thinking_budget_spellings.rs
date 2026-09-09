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
