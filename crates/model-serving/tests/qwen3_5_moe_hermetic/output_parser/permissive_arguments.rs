use super::*;

#[test]
fn should_pass_through_a_string_tool_parameter_outside_its_declared_enum() {
    let declared_tools = [ChatToolDefinition {
        name: "search".to_owned(),
        description: None,
        parameters_json: r#"{"type":"object","properties":{"mode":{"type":"string","enum":["files","content"]}},"required":["mode"]}"#.to_owned(),
    }];
    let mut output_parser = Qwen3_5MoEOutputParser::new(&declared_tools)
        .expect("the declared string enum schema should be supported");

    let tool_call_xml = format!(
        "{TOOL_CALL_START}<function=search><parameter=mode>network</parameter></function>{TOOL_CALL_END}"
    );
    let output_events = output_parser
        .push_fragment(&tool_call_xml)
        .expect("a model value outside the declared enum should pass through to the client, not kill the generation");

    assert_eq!(
        output_events,
        vec![Qwen3_5MoEOutputEvent::ToolCall(Qwen3_5MoEToolCall {
            index: 0,
            function_name: "search".to_owned(),
            arguments_json: r#"{"mode":"network"}"#.to_owned(),
        })]
    );
}

#[test]
fn should_pass_through_an_array_tool_parameter_with_a_wrong_item_type() {
    let declared_tools = [ChatToolDefinition {
        name: "inspect".to_owned(),
        description: None,
        parameters_json: r#"{"type":"object","properties":{"paths":{"type":"array","items":{"type":"string"}}},"required":["paths"]}"#.to_owned(),
    }];
    let mut output_parser = Qwen3_5MoEOutputParser::new(&declared_tools)
        .expect("the declared array item schema should be supported");

    let tool_call_xml = format!(
        "{TOOL_CALL_START}<function=inspect><parameter=paths>[\"src/lib.rs\",7]</parameter></function>{TOOL_CALL_END}"
    );
    let output_events = output_parser
        .push_fragment(&tool_call_xml)
        .expect("an array item with the wrong declared type should pass through to the client, not kill the generation");

    assert_eq!(
        output_events,
        vec![Qwen3_5MoEOutputEvent::ToolCall(Qwen3_5MoEToolCall {
            index: 0,
            function_name: "inspect".to_owned(),
            arguments_json: r#"{"paths":["src/lib.rs",7]}"#.to_owned(),
        })]
    );
}

#[test]
fn should_pass_through_a_nested_object_tool_parameter_missing_a_required_property() {
    let declared_tools = [ChatToolDefinition {
        name: "edit".to_owned(),
        description: None,
        parameters_json: r#"{"type":"object","properties":{"change":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}},"required":["change"]}"#.to_owned(),
    }];
    let mut output_parser = Qwen3_5MoEOutputParser::new(&declared_tools)
        .expect("the declared nested object schema should be supported");

    let tool_call_xml = format!(
        "{TOOL_CALL_START}<function=edit><parameter=change>{{}}</parameter></function>{TOOL_CALL_END}"
    );
    let output_events = output_parser
        .push_fragment(&tool_call_xml)
        .expect("a nested object missing a required field should pass through to the client, not kill the generation");

    assert_eq!(
        output_events,
        vec![Qwen3_5MoEOutputEvent::ToolCall(Qwen3_5MoEToolCall {
            index: 0,
            function_name: "edit".to_owned(),
            arguments_json: r#"{"change":{}}"#.to_owned(),
        })]
    );
}

#[test]
fn should_pass_through_a_nested_object_tool_parameter_with_an_undeclared_property() {
    let declared_tools = [ChatToolDefinition {
        name: "edit".to_owned(),
        description: None,
        parameters_json: r#"{"type":"object","properties":{"change":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}},"required":["change"]}"#.to_owned(),
    }];
    let mut output_parser = Qwen3_5MoEOutputParser::new(&declared_tools)
        .expect("the declared nested object schema should be supported");

    let tool_call_xml = format!(
        "{TOOL_CALL_START}<function=edit><parameter=change>{{\"path\":\"src/lib.rs\",\"mode\":\"append\"}}</parameter></function>{TOOL_CALL_END}"
    );
    let output_events = output_parser
        .push_fragment(&tool_call_xml)
        .expect("a nested object property absent from the schema should pass through to the client, not kill the generation");

    let arguments_json: Value = serde_json::from_str(
        output_events
            .iter()
            .find_map(|event| match event {
                Qwen3_5MoEOutputEvent::ToolCall(qwen3_5_moe_tool_call) => {
                    Some(qwen3_5_moe_tool_call.arguments_json.as_str())
                }
                _ => None,
            })
            .expect("the parser should emit a tool call event"),
    )
    .expect("the tool arguments should be valid JSON");

    assert_eq!(arguments_json["change"]["path"], "src/lib.rs");
    assert_eq!(arguments_json["change"]["mode"], "append");
}

#[test]
fn should_pass_through_a_nested_object_tool_parameter_with_a_wrong_property_type() {
    let declared_tools = [ChatToolDefinition {
        name: "edit".to_owned(),
        description: None,
        parameters_json: r#"{"type":"object","properties":{"change":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}},"required":["change"]}"#.to_owned(),
    }];
    let mut output_parser = Qwen3_5MoEOutputParser::new(&declared_tools)
        .expect("the declared nested object schema should be supported");

    let tool_call_xml = format!(
        "{TOOL_CALL_START}<function=edit><parameter=change>{{\"path\":7}}</parameter></function>{TOOL_CALL_END}"
    );
    let output_events = output_parser
        .push_fragment(&tool_call_xml)
        .expect("a nested object property with the wrong declared type should pass through to the client, not kill the generation");

    let arguments_json: Value = serde_json::from_str(
        output_events
            .iter()
            .find_map(|event| match event {
                Qwen3_5MoEOutputEvent::ToolCall(qwen3_5_moe_tool_call) => {
                    Some(qwen3_5_moe_tool_call.arguments_json.as_str())
                }
                _ => None,
            })
            .expect("the parser should emit a tool call event"),
    )
    .expect("the tool arguments should be valid JSON");

    assert_eq!(arguments_json["change"]["path"], 7);
}

#[test]
fn should_pass_through_tool_arguments_with_extra_object_properties_without_rejecting() {
    let declared_tools = [ChatToolDefinition {
        name: "open-brain_recall".to_owned(),
        description: None,
        parameters_json: r#"{"type":"object","properties":{"query":{"type":"string"},"scope":{"type":"object","properties":{"visibility":{"type":"string"},"project_only":{"type":"boolean"},"include_unconfirmed":{"type":"boolean"},"include_stale":{"type":"boolean"}}}},"required":["query"]}"#.to_owned(),
    }];
    let mut output_parser = Qwen3_5MoEOutputParser::new(&declared_tools)
        .expect("the declared tool schema should be supported");

    let tool_call_xml = format!(
        "{TOOL_CALL_START}<function=open-brain_recall><parameter=query>astronomical recent work session context</parameter><parameter=scope>{{\"recency_days\":7,\"include_unconfirmed\":false}}</parameter></function>{TOOL_CALL_END}"
    );
    let output_events = output_parser
        .push_fragment(&tool_call_xml)
        .expect("a tool call with extra object properties should pass through to the client, not kill the generation");

    let arguments_json: Value = serde_json::from_str(
        output_events
            .iter()
            .find_map(|event| match event {
                Qwen3_5MoEOutputEvent::ToolCall(qwen3_5_moe_tool_call) => {
                    Some(qwen3_5_moe_tool_call.arguments_json.as_str())
                }
                _ => None,
            })
            .expect("the parser should emit a tool call event"),
    )
    .expect("the tool arguments should be valid JSON");

    assert_eq!(
        arguments_json["query"],
        "astronomical recent work session context"
    );
    assert_eq!(arguments_json["scope"]["recency_days"], 7);
    assert_eq!(arguments_json["scope"]["include_unconfirmed"], false);
}

#[test]
fn should_overwrite_duplicate_tool_parameters_instead_of_rejecting() {
    let declared_tools = [ChatToolDefinition {
        name: "edit".to_owned(),
        description: None,
        parameters_json: r#"{"type":"object","properties":{"path":{"type":"string"}}}"#.to_owned(),
    }];
    let mut output_parser = Qwen3_5MoEOutputParser::new(&declared_tools)
        .expect("the declared tool schema should be supported");

    let tool_call_xml = format!(
        "{TOOL_CALL_START}<function=edit><parameter=path>first.rs</parameter><parameter=path>second.rs</parameter></function>{TOOL_CALL_END}"
    );
    let output_events = output_parser
        .push_fragment(&tool_call_xml)
        .expect("duplicate parameters should overwrite (last-value-wins), not kill the generation");

    let arguments_json: Value = serde_json::from_str(
        output_events
            .iter()
            .find_map(|event| match event {
                Qwen3_5MoEOutputEvent::ToolCall(qwen3_5_moe_tool_call) => {
                    Some(qwen3_5_moe_tool_call.arguments_json.as_str())
                }
                _ => None,
            })
            .expect("the parser should emit a tool call event"),
    )
    .expect("the tool arguments should be valid JSON");

    assert_eq!(arguments_json["path"], "second.rs");
}

#[test]
fn should_fall_back_to_string_when_type_parsing_fails_instead_of_rejecting() {
    let declared_tools = [ChatToolDefinition {
        name: "search".to_owned(),
        description: None,
        parameters_json: r#"{"type":"object","properties":{"count":{"type":"integer"},"enabled":{"type":"boolean"},"ratio":{"type":"number"}}}"#.to_owned(),
    }];
    let mut output_parser = Qwen3_5MoEOutputParser::new(&declared_tools)
        .expect("the declared tool schema should be supported");

    let tool_call_xml = format!(
        "{TOOL_CALL_START}<function=search><parameter=count>not-a-number</parameter><parameter=enabled>maybe</parameter><parameter=ratio>undefined</parameter></function>{TOOL_CALL_END}"
    );
    let output_events = output_parser
        .push_fragment(&tool_call_xml)
        .expect("type parsing failures should fall back to string, not kill the generation");

    let arguments_json: Value = serde_json::from_str(
        output_events
            .iter()
            .find_map(|event| match event {
                Qwen3_5MoEOutputEvent::ToolCall(qwen3_5_moe_tool_call) => {
                    Some(qwen3_5_moe_tool_call.arguments_json.as_str())
                }
                _ => None,
            })
            .expect("the parser should emit a tool call event"),
    )
    .expect("the tool arguments should be valid JSON");

    assert_eq!(arguments_json["count"], "not-a-number");
    assert_eq!(arguments_json["enabled"], "maybe");
    assert_eq!(arguments_json["ratio"], "undefined");
}
