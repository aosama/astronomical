use astronomical_ipc_protocol::ChatToolDefinition;
use astronomical_model_serving::{LagunaOutputEvent, LagunaOutputParser};

use super::super::text_support::SyntheticLagunaTextArtifact;
use super::support::literary_output_parser_starting_in_reasoning;

#[test]
fn should_start_in_reasoning_for_a_prompt_owned_opening_think_marker() {
    let mut output_parser = literary_output_parser_starting_in_reasoning(true);

    let output_events = output_parser
        .push_fragment("Compare their motives</think>The contrast is decisive.")
        .expect("prompt-opened reasoning should transition into visible text");

    assert_eq!(
        output_events,
        vec![
            LagunaOutputEvent::ReasoningDelta("Compare their motives".to_owned()),
            LagunaOutputEvent::TextDelta("The contrast is decisive.".to_owned()),
        ]
    );
}

#[test]
fn should_parse_reasoning_text_and_tool_markers_fragmented_at_every_boundary() {
    let mut output_parser = literary_output_parser_starting_in_reasoning(true);

    assert_eq!(
        output_parser
            .push_fragment("Plan from the play</thi")
            .expect("a fragmented thinking-end marker should stay buffered"),
        vec![LagunaOutputEvent::ReasoningDelta(
            "Plan from the play".to_owned()
        )]
    );
    assert_eq!(
        output_parser
            .push_fragment("nk>Answer.<tool_ca")
            .expect("a fragmented tool-call start should preserve preceding text"),
        vec![LagunaOutputEvent::TextDelta("Answer.".to_owned())]
    );
    assert!(
        output_parser
            .push_fragment("ll>find_character<arg_key>na")
            .expect("a fragmented argument key should remain pending")
            .is_empty()
    );
    assert!(
        output_parser
            .push_fragment("me</arg_key><arg_value>Jul")
            .expect("a fragmented argument value should remain pending")
            .is_empty()
    );
    assert_eq!(
        output_parser
            .push_fragment("iet</arg_value></tool_call>")
            .expect("the completed Poolside tool call should emit exactly once"),
        vec![LagunaOutputEvent::ToolCall {
            index: 0,
            function_name: "find_character".to_owned(),
            arguments_json: r#"{"name":"Juliet"}"#.to_owned(),
        }]
    );
}

#[test]
fn should_emit_multiple_poolside_tool_calls_in_source_order() {
    let mut output_parser = literary_output_parser_starting_in_reasoning(false);

    let output_events = output_parser
        .push_fragment(
            "<tool_call>find_character<arg_key>name</arg_key><arg_value>Romeo</arg_value></tool_call>\
             <tool_call>summarize_scene<arg_key>scene</arg_key><arg_value>tomb</arg_value></tool_call>",
        )
        .expect("two complete declared tool calls should parse in one fragment");

    assert_eq!(
        output_events,
        vec![
            LagunaOutputEvent::ToolCall {
                index: 0,
                function_name: "find_character".to_owned(),
                arguments_json: r#"{"name":"Romeo"}"#.to_owned(),
            },
            LagunaOutputEvent::ToolCall {
                index: 1,
                function_name: "summarize_scene".to_owned(),
                arguments_json: r#"{"scene":"tomb"}"#.to_owned(),
            },
        ]
    );
}

#[test]
fn should_accept_opencode_array_object_union_and_untyped_tool_parameters() {
    let descriptor = SyntheticLagunaTextArtifact::extra_small_inline().normalize();
    let declared_tools = vec![
        ChatToolDefinition {
            name: "todowrite".to_owned(),
            description: None,
            parameters_json: r#"{"type":"object","properties":{"todos":{"type":"array","items":{"type":"object","properties":{"content":{"type":"string"},"status":{"type":"string"}},"required":["content","status"]}}},"required":["todos"]}"#.to_owned(),
        },
        ChatToolDefinition {
            name: "grep".to_owned(),
            description: None,
            parameters_json: r#"{"type":"object","properties":{"paths":{"anyOf":[{"type":"string"},{"type":"array","items":{"type":"string"}}]}}}"#.to_owned(),
        },
        ChatToolDefinition {
            name: "invoke_canvas_action".to_owned(),
            description: None,
            parameters_json: r#"{"type":"object","properties":{"input":{"description":"Action input matching the action schema"}},"required":["input"]}"#.to_owned(),
        },
    ];
    let mut output_parser = LagunaOutputParser::new(&descriptor, &declared_tools, false)
        .expect("OpenCode tool schema forms should initialize Laguna output parsing");

    let output_events = output_parser
        .push_fragment(
            r#"<tool_call>todowrite<arg_key>todos</arg_key><arg_value>[{"content":"test Laguna","status":"pending"}]</arg_value></tool_call><tool_call>grep<arg_key>paths</arg_key><arg_value>["src","tests"]</arg_value></tool_call><tool_call>invoke_canvas_action<arg_key>input</arg_key><arg_value>{"action":"zoom"}</arg_value></tool_call>"#,
        )
        .expect("structured OpenCode tool arguments should remain strict JSON");

    assert_eq!(
        output_events,
        vec![
            LagunaOutputEvent::ToolCall {
                index: 0,
                function_name: "todowrite".to_owned(),
                arguments_json: r#"{"todos":[{"content":"test Laguna","status":"pending"}]}"#
                    .to_owned(),
            },
            LagunaOutputEvent::ToolCall {
                index: 1,
                function_name: "grep".to_owned(),
                arguments_json: r#"{"paths":["src","tests"]}"#.to_owned(),
            },
            LagunaOutputEvent::ToolCall {
                index: 2,
                function_name: "invoke_canvas_action".to_owned(),
                arguments_json: r#"{"input":{"action":"zoom"}}"#.to_owned(),
            },
        ]
    );
}

#[test]
fn should_forward_a_declared_array_argument_that_is_not_json() {
    let descriptor = SyntheticLagunaTextArtifact::extra_small_inline().normalize();
    let declared_tools = [ChatToolDefinition {
        name: "todowrite".to_owned(),
        description: None,
        parameters_json:
            r#"{"type":"object","properties":{"todos":{"type":"array"}},"required":["todos"]}"#
                .to_owned(),
    }];
    let mut output_parser = LagunaOutputParser::new(&descriptor, &declared_tools, false)
        .expect("a declared array should initialize parser state");

    let output_events = output_parser
        .push_fragment(
            "<tool_call>todowrite<arg_key>todos</arg_key><arg_value>not-json</arg_value></tool_call>",
        )
        .expect("a closed envelope must reach the harness even when an array value is not JSON");

    assert_eq!(
        output_events,
        vec![LagunaOutputEvent::ToolCall {
            index: 0,
            function_name: "todowrite".to_owned(),
            arguments_json: r#"{"todos":"not-json"}"#.to_owned(),
        }]
    );
}

#[test]
fn should_forward_javascript_object_literals_on_a_declared_array_argument() {
    let descriptor = SyntheticLagunaTextArtifact::extra_small_inline().normalize();
    let declared_tools = [ChatToolDefinition {
        name: "annotate_scene".to_owned(),
        description: None,
        parameters_json: r#"{"type":"object","properties":{"quotes":{"type":"array"},"scene":{"type":"string"}},"required":["quotes"]}"#.to_owned(),
    }];
    let mut output_parser = LagunaOutputParser::new(&descriptor, &declared_tools, false)
        .expect("a declared array-plus-string schema should initialize parser state");

    let output_events = output_parser
        .push_fragment(
            "<tool_call>annotate_scene<arg_key>quotes</arg_key><arg_value>{\n    speaker: \"Romeo\",\n    line: \"O Juliet\",\n  },\n]</arg_value><arg_key>scene</arg_key><arg_value>balcony</arg_value></tool_call>",
        )
        .expect("a closed envelope must reach the harness even when array text is a JavaScript literal");

    let [
        LagunaOutputEvent::ToolCall {
            function_name,
            arguments_json,
            ..
        },
    ] = output_events.as_slice()
    else {
        panic!("expected one tool call, got {output_events:?}");
    };
    assert_eq!(function_name, "annotate_scene");
    let arguments_value = serde_json::from_str::<serde_json::Value>(arguments_json)
        .expect("fail-open arguments should be JSON");
    assert_eq!(arguments_value["scene"], "balcony");
    assert!(arguments_value["quotes"].as_str().is_some());
}

#[test]
fn should_keep_a_json_array_argument_as_an_array_when_object_keys_repeat() {
    let descriptor = SyntheticLagunaTextArtifact::extra_small_inline().normalize();
    let declared_tools = [ChatToolDefinition {
        name: "annotate_scene".to_owned(),
        description: None,
        parameters_json:
            r#"{"type":"object","properties":{"quotes":{"type":"array"}},"required":["quotes"]}"#
                .to_owned(),
    }];
    let mut output_parser = LagunaOutputParser::new(&descriptor, &declared_tools, false)
        .expect("a declared array schema should initialize parser state");

    let output_events = output_parser
        .push_fragment(
            r#"<tool_call>annotate_scene<arg_key>quotes</arg_key><arg_value>[{"line":"O Romeo","line":"O Juliet"}]</arg_value></tool_call>"#,
        )
        .expect("duplicate JSON keys must not stringify a structured array away from the harness");

    let [LagunaOutputEvent::ToolCall { arguments_json, .. }] = output_events.as_slice() else {
        panic!("expected one tool call, got {output_events:?}");
    };
    let arguments_value = serde_json::from_str::<serde_json::Value>(arguments_json)
        .expect("fail-open arguments should be JSON");
    assert!(arguments_value["quotes"].is_array());
    assert_eq!(arguments_value["quotes"][0]["line"], "O Juliet");
}

#[test]
fn should_forward_an_undeclared_poolside_function_to_the_client() {
    let mut output_parser = literary_output_parser_starting_in_reasoning(false);

    let output_events = output_parser
        .push_fragment("<tool_call>invent_ending</tool_call>")
        .expect("a well-formed undeclared function should reach the tool client");

    assert_eq!(
        output_events,
        vec![LagunaOutputEvent::ToolCall {
            index: 0,
            function_name: "invent_ending".to_owned(),
            arguments_json: "{}".to_owned(),
        }]
    );
}

#[test]
fn should_recover_a_poolside_call_that_dropped_the_opening_argument_bracket() {
    let mut output_parser = literary_output_parser_starting_in_reasoning(false);

    let output_events = output_parser
        .push_fragment(
            "<tool_call>read\narg_key>path</arg_key><arg_value>balcony.md</arg_value></tool_call>",
        )
        .expect("a closed tool_call with a usable name must reach the harness");

    assert_eq!(
        output_events,
        vec![LagunaOutputEvent::ToolCall {
            index: 0,
            function_name: "read".to_owned(),
            arguments_json: r#"{"path":"balcony.md"}"#.to_owned(),
        }]
    );
}

#[test]
fn should_forward_a_closed_tool_call_when_argument_markers_cannot_be_salvaged() {
    let mut output_parser = literary_output_parser_starting_in_reasoning(false);

    let output_events = output_parser
        .push_fragment(
            "<tool_call>read\n<_key>argument-key</arg_key><arg_value>path</arg_value></tool_call>",
        )
        .expect("unsalvageable argument slop must not abort a named tool_call");

    assert_eq!(
        output_events,
        vec![LagunaOutputEvent::ToolCall {
            index: 0,
            function_name: "read".to_owned(),
            arguments_json: "{}".to_owned(),
        }]
    );
}

#[test]
fn should_preserve_a_bounded_undeclared_poolside_tool_argument_for_client_validation() {
    let mut output_parser = literary_output_parser_starting_in_reasoning(false);

    let output_events = output_parser
        .push_fragment(
            "<tool_call>find_character\
             <arg_key>name</arg_key><arg_value>Romeo</arg_value>\
             <arg_key>description</arg_key><arg_value>Locate the character</arg_value>\
             </tool_call>",
        )
        .expect("bounded model-generated metadata should pass through to the tool client");

    assert_eq!(
        output_events,
        vec![LagunaOutputEvent::ToolCall {
            index: 0,
            function_name: "find_character".to_owned(),
            arguments_json: r#"{"description":"Locate the character","name":"Romeo"}"#.to_owned(),
        }]
    );
}

#[test]
fn should_forward_a_declared_call_when_a_required_argument_is_missing() {
    let mut output_parser = literary_output_parser_starting_in_reasoning(false);

    let output_events = output_parser
        .push_fragment(
            "<tool_call>find_character\
             <arg_key>description</arg_key><arg_value>Locate the character</arg_value>\
             </tool_call>",
        )
        .expect("a closed envelope must reach the harness even without required arguments");

    assert_eq!(
        output_events,
        vec![LagunaOutputEvent::ToolCall {
            index: 0,
            function_name: "find_character".to_owned(),
            arguments_json: r#"{"description":"Locate the character"}"#.to_owned(),
        }]
    );
}

#[test]
fn should_forward_a_duplicate_poolside_tool_argument_with_the_last_value() {
    let mut output_parser = literary_output_parser_starting_in_reasoning(false);

    let output_events = output_parser
        .push_fragment(
            "<tool_call>find_character\
             <arg_key>name</arg_key><arg_value>Romeo</arg_value>\
             <arg_key>name</arg_key><arg_value>Juliet</arg_value>\
             </tool_call>",
        )
        .expect("a closed envelope must reach the harness even with duplicate argument names");

    assert_eq!(
        output_events,
        vec![LagunaOutputEvent::ToolCall {
            index: 0,
            function_name: "find_character".to_owned(),
            arguments_json: r#"{"name":"Juliet"}"#.to_owned(),
        }]
    );
}

#[test]
fn should_forward_a_closed_tool_call_missing_its_required_argument() {
    let mut output_parser = literary_output_parser_starting_in_reasoning(false);

    let output_events = output_parser
        .push_fragment("<tool_call>find_character</tool_call>")
        .expect("a closed envelope must reach the harness even without required arguments");

    assert_eq!(
        output_events,
        vec![LagunaOutputEvent::ToolCall {
            index: 0,
            function_name: "find_character".to_owned(),
            arguments_json: "{}".to_owned(),
        }]
    );
}

#[test]
fn should_forward_an_unclosed_poolside_tool_call_when_generation_finishes() {
    let mut output_parser = literary_output_parser_starting_in_reasoning(false);
    assert!(
        output_parser
            .push_fragment("<tool_call>find_character<arg_key>name</arg_key><arg_value>Romeo")
            .expect("an incomplete call should remain pending while output can continue")
            .is_empty()
    );

    let finish_events = output_parser
        .finish()
        .expect("an unclosed tool call must reach the harness when generation ends");

    assert_eq!(
        finish_events,
        vec![LagunaOutputEvent::ToolCall {
            index: 0,
            function_name: "find_character".to_owned(),
            arguments_json: "{}".to_owned(),
        }]
    );
}

#[test]
fn should_forward_a_closed_call_when_argument_markers_are_nested() {
    let mut output_parser = literary_output_parser_starting_in_reasoning(false);

    let output_events = output_parser
        .push_fragment(
            "<tool_call>find_character\
             <arg_key>name<arg_key>nested</arg_key></arg_key>\
             <arg_value>Romeo</arg_value></tool_call>",
        )
        .expect("a closed envelope must reach the harness even when argument markers are nested");

    assert_eq!(
        output_events,
        vec![LagunaOutputEvent::ToolCall {
            index: 0,
            function_name: "find_character".to_owned(),
            arguments_json: "{}".to_owned(),
        }]
    );
}

#[test]
fn should_salvage_an_oversized_tool_argument_without_aborting_generation() {
    let mut output_parser = literary_output_parser_starting_in_reasoning(false);
    output_parser
        .push_fragment("<tool_call>find_character<arg_key>name</arg_key><arg_value>")
        .expect("the parser should enter argument-value state");
    let oversized_argument = "R".repeat(LagunaOutputParser::MAXIMUM_TOOL_ARGUMENT_BYTES + 1);
    let mut salvaged_events = None;

    // Small fragments prove the aggregate argument bound rather than a per-fragment guard.
    for argument_fragment_bytes in oversized_argument.as_bytes().chunks(4 * 1_024) {
        let argument_fragment = std::str::from_utf8(argument_fragment_bytes)
            .expect("the generated ASCII argument fixture should remain UTF-8");
        let output_events = output_parser.push_fragment(argument_fragment).expect(
            "crossing the tool-argument bound must salvage the call instead of aborting generation",
        );
        if !output_events.is_empty() {
            salvaged_events = Some(output_events);
            break;
        }
    }
    let salvaged_events = salvaged_events
        .expect("the aggregate tool argument must salvage once it hits its byte bound");

    assert_eq!(
        salvaged_events,
        vec![LagunaOutputEvent::ToolCall {
            index: 0,
            function_name: "find_character".to_owned(),
            arguments_json: "{}".to_owned(),
        }]
    );
    assert!(
        output_parser
            .finish()
            .expect("salvage should leave the parser able to complete the request")
            .is_empty()
    );
}

#[test]
fn should_finish_streamed_reasoning_without_requiring_a_model_owned_closing_marker() {
    let mut output_parser = literary_output_parser_starting_in_reasoning(true);

    assert_eq!(
        output_parser
            .push_fragment("The short output budget ends during analysis")
            .expect("reasoning should stream before the request token budget ends"),
        vec![LagunaOutputEvent::ReasoningDelta(
            "The short output budget ends during analysis".to_owned()
        )]
    );
    assert!(
        output_parser
            .finish()
            .expect("budget exhaustion inside reasoning preserves prior reasoning behavior")
            .is_empty()
    );
}
