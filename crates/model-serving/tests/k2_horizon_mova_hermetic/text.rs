use astronomical_ipc_protocol::{
    ChatAssistantToolCall, ChatAssistantToolFunction, ChatGenerationOutput, ChatMessage,
    ChatToolChoice, ChatToolDefinition,
};
use astronomical_model_serving::{
    K2HorizonMoVAOutputParser, K2HorizonMoVAPromptRenderer, K2HorizonMoVAThinkingBudgetState,
    resolve_k2_horizon_mova_thinking_budget,
};

#[test]
fn should_render_romeo_and_juliet_user_turn_into_ifm_think_channel() {
    let renderer = K2HorizonMoVAPromptRenderer::new();
    let prompt = renderer.render(
        &[ChatMessage::User {
            content: "Two households, both alike in dignity.".to_owned(),
            images: Vec::new(),
        }],
        &[],
        &ChatToolChoice::Auto,
    );
    assert!(prompt.starts_with("<|ifm|begin_of_text|>"));
    assert!(
        prompt
            .contains("<|ifm|im_start|>user\nTwo households, both alike in dignity.<|ifm|im_end|>")
    );
    assert!(prompt.ends_with("<|ifm|im_start|>assistant\n<ifm|think>\n"));
}

#[test]
fn should_render_json_tool_catalog_and_xml_call_instructions() {
    let renderer = K2HorizonMoVAPromptRenderer::new();
    let prompt = renderer.render(
        &[ChatMessage::User {
            content: "Look up the scene.".to_owned(),
            images: Vec::new(),
        }],
        &[ChatToolDefinition {
            name: "get_scene".to_owned(),
            description: Some("Return a scene summary".to_owned()),
            parameters_json: r#"{"type":"object","properties":{"title":{"type":"string"}}}"#
                .to_owned(),
        }],
        &ChatToolChoice::Auto,
    );
    assert!(prompt.contains("<ifm|tools>"));
    assert!(prompt.contains("get_scene"));
    assert!(prompt.contains("<ifm|tool_calls>"));
    assert!(prompt.contains("<ifm|arg_key>"));
    assert!(prompt.contains("</ifm|think> before any tool call"));
}

#[test]
fn should_parse_ifm_think_then_visible_text() {
    let mut parser = K2HorizonMoVAOutputParser::new_after_thinking_prefix();
    let reasoning = parser.push_text("consider the families");
    assert_eq!(
        reasoning,
        vec![ChatGenerationOutput::Reasoning {
            text: "consider the families".to_owned()
        }]
    );
    let mixed = parser.push_text("</ifm|think>In fair Verona");
    assert!(mixed.iter().any(|output| matches!(
        output,
        ChatGenerationOutput::Text { text } if text.contains("In fair Verona")
    )));
}

#[test]
fn should_parse_xml_and_json_ifm_tool_calls() {
    let mut parser = K2HorizonMoVAOutputParser::new_after_thinking_prefix();
    let _ = parser.push_text("</ifm|think>");
    let xml = parser.push_text(
        "<ifm|tool_calls>\n<ifm|tool_call>get_scene\n<ifm|arg_key>title</ifm|arg_key>\n<ifm|arg_value>Romeo and Juliet</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>",
    );
    assert!(xml.iter().any(|output| matches!(
        output,
        ChatGenerationOutput::ToolCall { function_name, arguments_json, tool_call_index: 0 }
            if function_name == "get_scene" && arguments_json.contains("Romeo and Juliet")
    )));

    let mut json_parser = K2HorizonMoVAOutputParser::new_after_thinking_prefix();
    let _ = json_parser.push_text("</ifm|think>");
    let json = json_parser.push_text(
        "<ifm|tool_calls>\n<ifm|tool_call>{\"name\": \"lookup_line\", \"arguments\": {\"act\": 1}}</ifm|tool_call>\n</ifm|tool_calls>",
    );
    assert!(json.iter().any(|output| matches!(
        output,
        ChatGenerationOutput::ToolCall { function_name, arguments_json, .. }
            if function_name == "lookup_line" && arguments_json.contains("\"act\":1")
    )));
}

#[test]
fn should_promote_unclosed_think_channel_to_visible_text_without_turn_end_markers() {
    let mut parser = K2HorizonMoVAOutputParser::new_after_thinking_prefix();
    let streamed =
        parser.push_text("The two households are the Montagues and the Capulets.<|ifm|im_end|>");
    assert!(
        streamed.iter().all(|output| match output {
            ChatGenerationOutput::Reasoning { text } | ChatGenerationOutput::Text { text } => {
                !text.contains("<|ifm|im_end|>") && !text.contains("<|ifm|endoftext|>")
            }
            _ => true,
        }),
        "turn-end markers must not leak into streamed chat output",
    );
    let finished = parser.finish();
    assert!(
        finished.iter().any(|output| matches!(
            output,
            ChatGenerationOutput::Text { text }
                if text.contains("Montagues") && text.contains("Capulets")
                    && !text.contains("<|ifm|im_end|>")
        )),
        "a think-only reply must still become visible assistant text: {finished:?}",
    );
}

#[test]
fn should_salvage_compact_declared_tool_calls_from_the_think_channel() {
    let mut parser =
        K2HorizonMoVAOutputParser::with_declared_tool_names(vec!["get_scene".to_owned()]);
    let mut finished =
        parser.push_text("I will call the tool.\nget_scene{\"title\":\"Romeo and Juliet\"}");
    finished.extend(parser.finish());
    assert!(finished.iter().any(|output| matches!(
        output,
        ChatGenerationOutput::ToolCall { function_name, arguments_json, .. }
            if function_name == "get_scene" && arguments_json.contains("Romeo and Juliet")
    )));
}

#[test]
fn should_replay_assistant_tool_calls_in_ifm_xml() {
    let renderer = K2HorizonMoVAPromptRenderer::new();
    let prompt = renderer.render(
        &[
            ChatMessage::User {
                content: "Call the tool.".to_owned(),
                images: Vec::new(),
            },
            ChatMessage::Assistant {
                content: None,
                reasoning_content: Some("need the scene".to_owned()),
                tool_calls: vec![ChatAssistantToolCall {
                    id: "call_1".to_owned(),
                    function: ChatAssistantToolFunction {
                        name: "get_scene".to_owned(),
                        arguments_json: r#"{"title":"Romeo and Juliet"}"#.to_owned(),
                    },
                }],
            },
            ChatMessage::Tool {
                tool_call_id: "call_1".to_owned(),
                content: "Two households.".to_owned(),
            },
        ],
        &[],
        &ChatToolChoice::Auto,
    );
    assert!(prompt.contains("<ifm|tool_call>get_scene"));
    assert!(prompt.contains("<ifm|arg_key>title</ifm|arg_key>"));
    assert!(prompt.contains("<|ifm|im_start|>tool\nTwo households."));
}

#[test]
fn should_salvage_json_declared_tool_calls_from_the_think_channel() {
    let mut parser =
        K2HorizonMoVAOutputParser::with_declared_tool_names(vec!["get_scene".to_owned()]);
    let mut finished = parser.push_text(
        r#"I'll look that up. {"name":"get_scene","arguments":{"title":"Romeo and Juliet"}}"#,
    );
    finished.extend(parser.finish());
    assert!(finished.iter().any(|output| matches!(
        output,
        ChatGenerationOutput::ToolCall { function_name, arguments_json, .. }
            if function_name == "get_scene" && arguments_json.contains("Romeo and Juliet")
    )));
}

#[test]
fn should_force_think_close_after_the_family_output_budget() {
    let close_token_id = 250030_u32;
    let thinking_budget = resolve_k2_horizon_mova_thinking_budget(None, 96, 1)
        .expect("short Chat Completions must receive a K2 think budget");
    assert!(
        thinking_budget < 96,
        "the think budget must leave room for a visible answer"
    );
    let mut budget_state = K2HorizonMoVAThinkingBudgetState::new(
        Some(thinking_budget),
        vec![close_token_id],
        vec![close_token_id],
    )
    .expect("think-close token sequence should be valid");
    for _think_token in 0..thinking_budget {
        assert!(
            budget_state
                .next_forced_transition_token_id()
                .expect("ordinary think tokens are not forced")
                .is_none()
        );
        assert!(
            budget_state
                .observe_committed_token(7)
                .expect("think tokens should stay in the think channel")
        );
    }
    assert_eq!(
        budget_state
            .next_forced_transition_token_id()
            .expect("the close token should be forced once the budget is spent"),
        Some(close_token_id)
    );
    assert!(
        !budget_state
            .observe_committed_token(close_token_id)
            .expect("the forced close must leave the think channel")
    );
    assert!(!budget_state.is_inside_thinking());
}
