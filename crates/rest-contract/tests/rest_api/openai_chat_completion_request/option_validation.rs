use super::*;

#[test]
fn should_reject_an_output_budget_above_the_worker_representation_limit() {
    let request_json = format!(
        r#"{{
            "model": "astronomical/fake-mixture-of-experts",
            "messages": [{{"role": "user", "content": "write a function"}}],
            "max_tokens": {}
        }}"#,
        MAX_OPENAI_OUTPUT_TOKENS + 1
    );
    let chat_completion_request =
        serde_json::from_str::<OpenAiChatCompletionRequest>(&request_json)
            .expect("the output budget should decode before bounded validation");

    let validation_error = chat_completion_request
        .validate()
        .expect_err("the public endpoint must reject an oversized output budget");

    assert_eq!(
        validation_error,
        OpenAiChatCompletionValidationError::OutputTokenCountOutOfRange {
            actual_output_tokens: MAX_OPENAI_OUTPUT_TOKENS + 1,
            maximum_output_tokens: MAX_OPENAI_OUTPUT_TOKENS,
        }
    );
}

#[test]
fn should_reject_caller_supplied_stop_sequences_before_worker_admission() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "Inspect the repository."}],
        "stop": ["</tool_call>"]
    }
    "#;
    let chat_completion_request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("stop sequences should decode before unsupported-option validation");

    let validation_error = chat_completion_request
        .validate()
        .expect_err("caller-supplied stop sequences must not be accepted and ignored");

    assert_eq!(
        validation_error,
        OpenAiChatCompletionValidationError::UnsupportedStopSequences
    );
}

#[test]
fn should_reject_required_tool_choice_before_worker_admission() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "Inspect the repository."}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "glob", "parameters": {"type": "object"}}
            }
        ],
        "tool_choice": "required"
    }
    "#;
    let chat_completion_request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("required tool choice should decode before unsupported-option validation");

    let validation_error = chat_completion_request
        .validate()
        .expect_err("required tool choice must not rely on unenforced prompt hints");

    assert_eq!(
        validation_error,
        OpenAiChatCompletionValidationError::UnsupportedToolChoice {
            mode: "required".to_owned(),
        }
    );
}

#[test]
fn should_reject_a_named_forced_function_before_worker_admission() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "Inspect the repository."}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "glob", "parameters": {"type": "object"}}
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "glob"}}
    }
    "#;
    let chat_completion_request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("a named function choice should decode before unsupported-option validation");

    let validation_error = chat_completion_request.validate().expect_err(
        "named function choices must not be accepted without deterministic enforcement",
    );

    assert_eq!(
        validation_error,
        OpenAiChatCompletionValidationError::UnsupportedForcedToolChoice {
            function_name: "glob".to_owned(),
        }
    );
}

#[test]
fn should_reject_a_known_but_unsupported_opencode_option_explicitly() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "Inspect the repository."}],
        "store": true
    }
    "#;
    let chat_completion_request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("known OpenCode wire fields should decode before option validation");

    assert_eq!(
        chat_completion_request.validate(),
        Err(OpenAiChatCompletionValidationError::UnsupportedOption {
            option_name: "store",
        })
    );
}

#[test]
fn should_reject_an_unknown_openai_field_explicitly() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "Inspect the repository."}],
        "response_format": {"type": "json_object"}
    }
    "#;
    let chat_completion_request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("unknown fields should decode so contract validation can return a typed error");

    assert_eq!(
        chat_completion_request.validate(),
        Err(OpenAiChatCompletionValidationError::UnknownField {
            field_name: "response_format".to_owned(),
        })
    );
}
