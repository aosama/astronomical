use super::*;

#[test]
fn should_emit_a_schema_validated_tool_call_after_its_closing_marker() {
    let declared_tools = [ChatToolDefinition {
        name: "glob".to_owned(),
        description: Some("List matching files.".to_owned()),
        parameters_json:
            r#"{"type":"object","properties":{"pattern":{"type":"string"}},"required":["pattern"]}"#
                .to_owned(),
    }];
    let mut output_parser = Qwen3_5MoEOutputParser::new(&declared_tools)
        .expect("the declared glob schema should be valid");

    let tool_call_xml = format!(
        "{TOOL_CALL_START}\n<function=glob>\n<parameter=pattern>\nsrc/**/*.rs\n</parameter>\n</function>\n{TOOL_CALL_END}"
    );
    let output_events = output_parser
        .push_fragment(&tool_call_xml)
        .expect("a complete Qwen3.5-MoE XML tool call should parse");

    assert_eq!(
        output_events,
        vec![Qwen3_5MoEOutputEvent::ToolCall(Qwen3_5MoEToolCall {
            index: 0,
            function_name: "glob".to_owned(),
            arguments_json: r#"{"pattern":"src/**/*.rs"}"#.to_owned(),
        })]
    );
    assert!(
        output_parser
            .finish()
            .expect("a closed tool call should leave no retained parser output")
            .is_empty()
    );
}

#[test]
fn should_accept_more_than_sixteen_tool_calls_when_the_model_emits_a_valid_batch() {
    let declared_tools = [ChatToolDefinition {
        name: "glob".to_owned(),
        description: Some("List matching files.".to_owned()),
        parameters_json:
            r#"{"type":"object","properties":{"pattern":{"type":"string"}},"required":["pattern"]}"#
                .to_owned(),
    }];
    let mut output_parser = Qwen3_5MoEOutputParser::new(&declared_tools)
        .expect("the declared glob schema should be valid");
    let tool_call_xml = (0..17)
        .map(|tool_call_number| {
            format!(
                "{TOOL_CALL_START}<function=glob><parameter=pattern>src/{tool_call_number}.rs</parameter></function>{TOOL_CALL_END}"
            )
        })
        .collect::<String>();

    let output_events = output_parser.push_fragment(&tool_call_xml).expect(
        "valid tool-call batches should be bounded by bytes and protocol indexes, not sixteen",
    );

    assert_eq!(output_events.len(), 17);
    assert_eq!(
        output_events.last(),
        Some(&Qwen3_5MoEOutputEvent::ToolCall(Qwen3_5MoEToolCall {
            index: 16,
            function_name: "glob".to_owned(),
            arguments_json: r#"{"pattern":"src/16.rs"}"#.to_owned(),
        }))
    );
}

#[test]
fn should_accept_a_large_generated_tool_parameter_when_the_output_frame_fits() {
    let declared_tools = [ChatToolDefinition {
        name: "edit".to_owned(),
        description: Some("Apply a generated patch.".to_owned()),
        parameters_json:
            r#"{"type":"object","properties":{"patch":{"type":"string"}},"required":["patch"]}"#
                .to_owned(),
    }];
    let mut output_parser = Qwen3_5MoEOutputParser::new(&declared_tools)
        .expect("the declared edit schema should be valid");
    let large_generated_patch = "x".repeat(160 * 1024);

    output_parser
        .push_fragment(&format!(
            "{TOOL_CALL_START}<function=edit><parameter=patch>"
        ))
        .expect("the parser should enter tool-call state");
    for generated_patch_fragment in large_generated_patch.as_bytes().chunks(4 * 1024) {
        let generated_patch_fragment = std::str::from_utf8(generated_patch_fragment)
            .expect("the generated patch fixture should be UTF-8");
        output_parser
            .push_fragment(generated_patch_fragment)
            .expect("large generated tool parameters should be bounded by output/frame budgets");
    }
    let output_events = output_parser
        .push_fragment(&format!("</parameter></function>{TOOL_CALL_END}"))
        .expect("a closed large generated tool parameter should parse");

    let [Qwen3_5MoEOutputEvent::ToolCall(tool_call)] = output_events.as_slice() else {
        panic!("expected one parsed tool call, got {output_events:?}");
    };
    assert_eq!(tool_call.function_name, "edit");
    let arguments_value = serde_json::from_str::<Value>(&tool_call.arguments_json)
        .expect("the emitted tool arguments should be valid JSON");
    assert_eq!(arguments_value["patch"], large_generated_patch);
}
