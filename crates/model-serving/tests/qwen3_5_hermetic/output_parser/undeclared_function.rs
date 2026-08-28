use super::*;

#[test]
fn should_forward_an_undeclared_function_as_a_tool_call_for_the_harness() {
    let declared_tools = [
        ChatToolDefinition {
            name: "open-brain_openbrain_recall".to_owned(),
            description: None,
            parameters_json: r#"{"type":"object","properties":{"query":{"type":"string"}}}"#
                .to_owned(),
        },
        ChatToolDefinition {
            name: "bash".to_owned(),
            description: None,
            parameters_json: r#"{"type":"object","properties":{"command":{"type":"string"}}}"#
                .to_owned(),
        },
    ];
    let mut output_parser =
        Qwen3_5OutputParser::new(&declared_tools).expect("the declared tools should be valid");

    let tool_call_xml = format!(
        "{THINK_START}I should search memory.{THINK_END}{TOOL_CALL_START}\n<function=open_brain>\n<parameter=query>repo history\n</parameter>\n</function>\n{TOOL_CALL_END}"
    );
    let output_events = output_parser
        .push_fragment(&tool_call_xml)
        .expect("a well-formed undeclared function should reach the harness as a tool call");

    assert!(
        output_events.contains(&Qwen3_5OutputEvent::ReasoningDelta(
            "I should search memory.".to_owned()
        )),
        "reasoning emitted before the undeclared function should still stream"
    );
    assert!(
        !output_events
            .iter()
            .any(|event| matches!(event, Qwen3_5OutputEvent::ModelVisibleCorrection { .. })),
        "undeclared names must not be rewritten as model-visible corrections"
    );
    assert_eq!(
        output_events
            .iter()
            .filter_map(|event| match event {
                Qwen3_5OutputEvent::ToolCall(tool_call) => Some(tool_call),
                _ => None,
            })
            .collect::<Vec<_>>(),
        vec![&Qwen3_5ToolCall {
            index: 0,
            function_name: "open_brain".to_owned(),
            arguments_json: r#"{"query":"repo history"}"#.to_owned(),
        }]
    );
}
