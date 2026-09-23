use super::*;

use astronomical_model_serving::{
    MalformedModelOutputDiagnostic, Qwen3_5OutputParserError, Qwen3_5RequestOutputError,
};

#[test]
fn should_emit_null_for_a_nullable_string_tool_parameter() {
    let declared_tools = [ChatToolDefinition {
        name: "read".to_owned(),
        description: None,
        parameters_json: r#"{"type":"object","properties":{"path":{"type":["string","null"]}},"required":["path"]}"#.to_owned(),
    }];
    let mut output_parser = Qwen3_5OutputParser::new(&declared_tools)
        .expect("the declared nullable string schema should be supported");

    let tool_call_xml = format!(
        "{TOOL_CALL_START}<function=read><parameter=path>null</parameter></function>{TOOL_CALL_END}"
    );
    let output_events = output_parser
        .push_fragment(&tool_call_xml)
        .expect("a null value should satisfy a nullable string schema");

    assert_eq!(
        output_events,
        vec![Qwen3_5OutputEvent::ToolCall(Qwen3_5ToolCall {
            index: 0,
            function_name: "read".to_owned(),
            arguments_json: r#"{"path":null}"#.to_owned(),
        })]
    );
}

#[test]
fn should_emit_null_for_an_any_of_nullable_string_tool_parameter() {
    let declared_tools = [ChatToolDefinition {
        name: "recall".to_owned(),
        description: None,
        parameters_json: r#"{"type":"object","properties":{"project_id":{"anyOf":[{"type":"string"},{"type":"null"}]}}}"#.to_owned(),
    }];
    let mut output_parser = Qwen3_5OutputParser::new(&declared_tools)
        .expect("OpenCode nullable anyOf schemas should be supported");

    let tool_call_xml = format!(
        "{TOOL_CALL_START}<function=recall><parameter=project_id>null</parameter></function>{TOOL_CALL_END}"
    );
    let output_events = output_parser
        .push_fragment(&tool_call_xml)
        .expect("a null value should satisfy an anyOf nullable string schema");

    assert_eq!(
        output_events,
        vec![Qwen3_5OutputEvent::ToolCall(Qwen3_5ToolCall {
            index: 0,
            function_name: "recall".to_owned(),
            arguments_json: r#"{"project_id":null}"#.to_owned(),
        })]
    );
}

#[test]
fn should_accept_an_opencode_constrained_nullable_integer_tool_parameter() {
    let declared_tools = [ChatToolDefinition {
        name: "recall".to_owned(),
        description: None,
        parameters_json: r#"{"type":"object","properties":{"recency_days":{"anyOf":[{"exclusiveMinimum":0,"maximum":9007199254740991,"type":"integer"},{"type":"null"}]}}}"#.to_owned(),
    }];
    let mut output_parser = Qwen3_5OutputParser::new(&declared_tools)
        .expect("OpenCode constrained nullable integer schemas should be supported");

    let tool_call_xml = format!(
        "{TOOL_CALL_START}<function=recall><parameter=recency_days>7</parameter></function>{TOOL_CALL_END}"
    );
    let output_events = output_parser
        .push_fragment(&tool_call_xml)
        .expect("an integer value should satisfy the constrained nullable schema");

    assert_eq!(
        output_events,
        vec![Qwen3_5OutputEvent::ToolCall(Qwen3_5ToolCall {
            index: 0,
            function_name: "recall".to_owned(),
            arguments_json: r#"{"recency_days":7}"#.to_owned(),
        })]
    );
}

#[test]
fn should_accept_an_open_brain_nullable_string_with_minlength_maxlength_constraints() {
    let declared_tools = [ChatToolDefinition {
        name: "open-brain_recall".to_owned(),
        description: None,
        parameters_json: r#"{"type":"object","properties":{"project_id":{"anyOf":[{"type":"string","minLength":1,"maxLength":200},{"type":"null"}]}}}"#.to_owned(),
    }];
    let mut output_parser = Qwen3_5OutputParser::new(&declared_tools)
        .expect("Open Brain nullable string schemas with minLength/maxLength constraints should be supported");

    let tool_call_xml = format!(
        "{TOOL_CALL_START}<function=open-brain_recall><parameter=project_id>ob1-staging</parameter></function>{TOOL_CALL_END}"
    );
    let output_events = output_parser
        .push_fragment(&tool_call_xml)
        .expect("a string value should satisfy the nullable string schema with length constraints");

    assert_eq!(
        output_events,
        vec![Qwen3_5OutputEvent::ToolCall(Qwen3_5ToolCall {
            index: 0,
            function_name: "open-brain_recall".to_owned(),
            arguments_json: r#"{"project_id":"ob1-staging"}"#.to_owned(),
        })]
    );
}

#[test]
fn should_accept_copilot_string_or_array_tool_parameters() {
    let declared_tools = [ChatToolDefinition {
        name: "grep".to_owned(),
        description: None,
        parameters_json: r#"{"type":"object","properties":{"paths":{"anyOf":[{"type":"string"},{"type":"array","items":{"type":"string"}}]}}}"#.to_owned(),
    }];

    for (parameter_text, expected_arguments_json) in [
        ("src", r#"{"paths":"src"}"#),
        (r#"["src","tests"]"#, r#"{"paths":["src","tests"]}"#),
    ] {
        let mut output_parser = Qwen3_5OutputParser::new(&declared_tools)
            .expect("Copilot string-or-array tool schemas should be supported");
        let tool_call_xml = format!(
            "{TOOL_CALL_START}<function=grep><parameter=paths>{parameter_text}</parameter></function>{TOOL_CALL_END}"
        );

        let output_events = output_parser
            .push_fragment(&tool_call_xml)
            .expect("either declared Copilot parameter shape should parse");

        assert_eq!(
            output_events,
            vec![Qwen3_5OutputEvent::ToolCall(Qwen3_5ToolCall {
                index: 0,
                function_name: "grep".to_owned(),
                arguments_json: expected_arguments_json.to_owned(),
            })]
        );
    }
}

#[test]
fn should_accept_copilot_opaque_canvas_action_parameters() {
    let declared_tools = [ChatToolDefinition {
        name: "invoke_canvas_action".to_owned(),
        description: None,
        parameters_json: r#"{"type":"object","properties":{"input":{"description":"Action input matching the action input schema"}},"required":["input"]}"#.to_owned(),
    }];
    let mut output_parser = Qwen3_5OutputParser::new(&declared_tools)
        .expect("Copilot opaque canvas action schemas should be supported");
    let tool_call_xml = format!(
        "{TOOL_CALL_START}<function=invoke_canvas_action><parameter=input>{{\"action\":\"zoom\"}}</parameter></function>{TOOL_CALL_END}"
    );

    let output_events = output_parser
        .push_fragment(&tool_call_xml)
        .expect("the opaque canvas action input should use dynamic JSON parsing");

    assert_eq!(
        output_events,
        vec![Qwen3_5OutputEvent::ToolCall(Qwen3_5ToolCall {
            index: 0,
            function_name: "invoke_canvas_action".to_owned(),
            arguments_json: r#"{"input":{"action":"zoom"}}"#.to_owned(),
        })]
    );
}

#[test]
fn should_accept_a_multi_branch_string_any_of_tool_parameter() {
    // A client tool inventory may declare an enum-like parameter as more than two
    // anyOf branches; only the branch types affect argument parsing, so the count
    // must not reject the complete chat thread.
    let declared_tools = [ChatToolDefinition {
        name: "xcode_build".to_owned(),
        description: None,
        parameters_json: r#"{"type":"object","properties":{"includeBuildLog":{"anyOf":[{"type":"string","const":"onFailure"},{"type":"string","const":"always"},{"type":"string","const":"never"}]}}}"#.to_owned(),
    }];
    let mut output_parser = Qwen3_5OutputParser::new(&declared_tools)
        .expect("a multi-branch string anyOf schema should be supported");

    let tool_call_xml = format!(
        "{TOOL_CALL_START}<function=xcode_build><parameter=includeBuildLog>always</parameter></function>{TOOL_CALL_END}"
    );
    let output_events = output_parser
        .push_fragment(&tool_call_xml)
        .expect("the multi-branch anyOf parameter should parse");

    assert_eq!(
        output_events,
        vec![Qwen3_5OutputEvent::ToolCall(Qwen3_5ToolCall {
            index: 0,
            function_name: "xcode_build".to_owned(),
            arguments_json: r#"{"includeBuildLog":"always"}"#.to_owned(),
        })]
    );
}

#[test]
fn should_accept_a_multi_branch_any_of_with_a_null_branch() {
    let declared_tools = [ChatToolDefinition {
        name: "recall".to_owned(),
        description: None,
        parameters_json: r#"{"type":"object","properties":{"limit":{"anyOf":[{"type":"string"},{"type":"integer"},{"type":"null"}]}}}"#.to_owned(),
    }];
    let mut output_parser = Qwen3_5OutputParser::new(&declared_tools)
        .expect("a multi-branch anyOf with a null branch should be supported");

    let tool_call_xml = format!(
        "{TOOL_CALL_START}<function=recall><parameter=limit>7</parameter></function>{TOOL_CALL_END}"
    );
    let output_events = output_parser
        .push_fragment(&tool_call_xml)
        .expect("mixed branch types should parse arguments dynamically");

    assert_eq!(
        output_events,
        vec![Qwen3_5OutputEvent::ToolCall(Qwen3_5ToolCall {
            index: 0,
            function_name: "recall".to_owned(),
            arguments_json: r#"{"limit":7}"#.to_owned(),
        })]
    );
}

#[test]
fn should_name_the_offending_tool_and_property_when_a_declared_schema_is_rejected() {
    let parser_error = Qwen3_5OutputParser::new(&[ChatToolDefinition {
        name: "broken_tool".to_owned(),
        description: None,
        parameters_json: r#"{"type":"object","properties":{"flag":{"type":42}}}"#.to_owned(),
    }])
    .expect_err("an unsupported property type declaration must be rejected");
    let request_output_error = Qwen3_5RequestOutputError::Parser {
        source: parser_error,
        diagnostic: Box::new(MalformedModelOutputDiagnostic {
            diagnostic_code: "invalid_tool_parameter_type_declaration",
            parser_error:
                "declared tool 'broken_tool' property 'flag' has an unsupported type declaration"
                    .to_owned(),
            generated_token_ids: Vec::new(),
            pending_token_ids: Vec::new(),
            decoded_output_text: String::new(),
            parser_state: "initialization",
            parser_pending_output_text: String::new(),
        }),
    };

    let surfaced_message = request_output_error.to_string();

    assert!(
        surfaced_message.contains("broken_tool") && surfaced_message.contains("flag"),
        "a rejected declaration must name the offending tool and property so the failure is actionable, but the message was: {surfaced_message}"
    );
    assert!(
        matches!(
            &request_output_error,
            Qwen3_5RequestOutputError::Parser { source, .. }
                if matches!(source, Qwen3_5OutputParserError::InvalidToolParameterTypeDeclaration { .. })
        ),
        "the typed parser error must stay available as the error source"
    );
}

#[test]
fn should_tolerate_unresolvable_any_of_in_a_declared_tool_property() {
    // Real harness inventories (e.g. Copilot Desktop's save_workflow.agent_scope) declare
    // properties whose anyOf no resolver can reduce to a single coercion type: branches
    // without a type member, single-branch unions, type-list shorthands, and non-array
    // anyOf values. JSON Schema treats type as optional, so these fall back to dynamic
    // JSON argument parsing instead of rejecting the complete chat thread (#771).
    let unresolvable_shapes = [
        (
            r#"{"anyOf":[{"enum":["agent","team","workspace"]}]}"#,
            "workspace",
            r#"{"agent_scope":"workspace"}"#,
        ),
        (
            r#"{"anyOf":["string","null"]}"#,
            "agent",
            r#"{"agent_scope":"agent"}"#,
        ),
        (
            r#"{"anyOf":{"type":"string"}}"#,
            "team",
            r#"{"agent_scope":"team"}"#,
        ),
    ];
    for (property_schema, parameter_value, expected_arguments_json) in unresolvable_shapes {
        let declared_tools = [ChatToolDefinition {
            name: "save_workflow".to_owned(),
            description: None,
            parameters_json: format!(
                r#"{{"type":"object","properties":{{"agent_scope":{property_schema}}}}}"#
            ),
        }];
        let mut output_parser = Qwen3_5OutputParser::new(&declared_tools)
            .expect("an unresolvable anyOf declaration must not reject the declared tool");

        let tool_call_xml = format!(
            "{TOOL_CALL_START}<function=save_workflow><parameter=agent_scope>{parameter_value}</parameter></function>{TOOL_CALL_END}"
        );
        let output_events = output_parser
            .push_fragment(&tool_call_xml)
            .expect("the tool call should parse even though the property type is unresolvable");

        assert_eq!(
            output_events,
            vec![Qwen3_5OutputEvent::ToolCall(Qwen3_5ToolCall {
                index: 0,
                function_name: "save_workflow".to_owned(),
                arguments_json: expected_arguments_json.to_owned(),
            })]
        );
    }
}

#[test]
fn should_accept_a_declared_tool_schema_deeper_than_the_previous_parser_limit() {
    let mut nested_schema = r#"{"type":"string"}"#.to_owned();
    for _nesting_level in 0..10 {
        nested_schema = format!(
            r#"{{"type":"object","properties":{{"child":{nested_schema}}},"required":["child"]}}"#
        );
    }
    let declared_tools = [ChatToolDefinition {
        name: "deep".to_owned(),
        description: None,
        parameters_json: format!(
            r#"{{"type":"object","properties":{{"root":{nested_schema}}},"required":["root"]}}"#
        ),
    }];

    let _output_parser = Qwen3_5OutputParser::new(&declared_tools)
        .expect("a deep declared schema should parse without recursive validation since arguments pass through to the client");
}
