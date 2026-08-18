use astronomical_ipc_protocol::ChatToolDefinition;
use astronomical_model_serving::{LagunaOutputEvent, LagunaOutputParser, LagunaOutputParserError};

use super::text_support::{SyntheticLagunaTextArtifact, declared_literary_tools};

#[test]
fn should_start_in_reasoning_for_a_prompt_owned_opening_think_marker() {
    let mut output_parser = output_parser(true);

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
    let mut output_parser = output_parser(true);

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
    let mut output_parser = output_parser(false);

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
fn should_reject_a_malformed_declared_array_argument_after_parser_initialization() {
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

    let parser_error = output_parser
        .push_fragment(
            "<tool_call>todowrite<arg_key>todos</arg_key><arg_value>not-json</arg_value></tool_call>",
        )
        .expect_err("a generated array argument must contain a JSON array");

    assert!(matches!(
        parser_error,
        LagunaOutputParserError::InvalidToolArgumentValue
    ));
}

#[test]
fn should_reject_an_undeclared_poolside_function() {
    let mut output_parser = output_parser(false);

    let parser_error = output_parser
        .push_fragment("<tool_call>invent_ending</tool_call>")
        .expect_err("generated functions must match one declared tool exactly");

    assert!(matches!(
        &parser_error,
        LagunaOutputParserError::UndeclaredFunction { function_name }
            if function_name == "invent_ending"
    ));
    assert_bounded_error(&parser_error);
}

#[test]
fn should_preserve_a_bounded_undeclared_poolside_tool_argument_for_client_validation() {
    let mut output_parser = output_parser(false);

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
fn should_still_require_declared_arguments_when_extra_metadata_is_present() {
    let mut output_parser = output_parser(false);

    let parser_error = output_parser
        .push_fragment(
            "<tool_call>find_character\
             <arg_key>description</arg_key><arg_value>Locate the character</arg_value>\
             </tool_call>",
        )
        .expect_err("extra metadata must not satisfy a missing required argument");

    assert!(matches!(
        &parser_error,
        LagunaOutputParserError::MissingRequiredToolArgument {
            function_name,
            argument_name,
        } if function_name == "find_character" && argument_name == "name"
    ));
    assert_bounded_error(&parser_error);
}

#[test]
fn should_reject_a_duplicate_poolside_tool_argument() {
    let mut output_parser = output_parser(false);

    let parser_error = output_parser
        .push_fragment(
            "<tool_call>find_character\
             <arg_key>name</arg_key><arg_value>Romeo</arg_value>\
             <arg_key>name</arg_key><arg_value>Juliet</arg_value>\
             </tool_call>",
        )
        .expect_err("duplicate argument names must not silently overwrite user-visible calls");

    assert!(matches!(
        &parser_error,
        LagunaOutputParserError::DuplicateToolArgument { argument_name }
            if argument_name == "name"
    ));
    assert_bounded_error(&parser_error);
}

#[test]
fn should_reject_a_closed_tool_call_missing_its_required_argument() {
    let mut output_parser = output_parser(false);

    let parser_error = output_parser
        .push_fragment("<tool_call>find_character</tool_call>")
        .expect_err("a syntactically closed call remains incomplete without required arguments");

    assert!(matches!(
        &parser_error,
        LagunaOutputParserError::MissingRequiredToolArgument {
            function_name,
            argument_name,
        } if function_name == "find_character" && argument_name == "name"
    ));
    assert_bounded_error(&parser_error);
}

#[test]
fn should_reject_an_incomplete_poolside_tool_call_when_generation_finishes() {
    let mut output_parser = output_parser(false);
    assert!(
        output_parser
            .push_fragment("<tool_call>find_character<arg_key>name</arg_key><arg_value>Romeo")
            .expect("an incomplete call should remain pending while output can continue")
            .is_empty()
    );

    let parser_error = output_parser
        .finish()
        .expect_err("generation cannot complete with an open tool argument");

    assert!(matches!(
        &parser_error,
        LagunaOutputParserError::IncompleteToolCall
    ));
    assert_bounded_error(&parser_error);
}

#[test]
fn should_reject_nested_poolside_tool_argument_markers() {
    let mut output_parser = output_parser(false);

    let parser_error = output_parser
        .push_fragment(
            "<tool_call>find_character\
             <arg_key>name<arg_key>nested</arg_key></arg_key>\
             <arg_value>Romeo</arg_value></tool_call>",
        )
        .expect_err("Poolside arguments are flat key/value pairs, not nested marker trees");

    assert!(matches!(
        &parser_error,
        LagunaOutputParserError::NestedToolArgumentMarker
    ));
    assert_bounded_error(&parser_error);
}

#[test]
fn should_reject_oversized_tool_arguments_without_echoing_the_payload() {
    let mut output_parser = output_parser(false);
    output_parser
        .push_fragment("<tool_call>find_character<arg_key>name</arg_key><arg_value>")
        .expect("the parser should enter argument-value state");
    let oversized_argument = "R".repeat(LagunaOutputParser::MAXIMUM_TOOL_ARGUMENT_BYTES + 1);
    let mut parser_error = None;

    // Small fragments prove the aggregate argument bound rather than a per-fragment guard.
    for argument_fragment_bytes in oversized_argument.as_bytes().chunks(4 * 1_024) {
        let argument_fragment = std::str::from_utf8(argument_fragment_bytes)
            .expect("the generated ASCII argument fixture should remain UTF-8");
        if let Err(observed_parser_error) = output_parser.push_fragment(argument_fragment) {
            parser_error = Some(observed_parser_error);
            break;
        }
    }
    let parser_error = parser_error.expect("the aggregate tool argument must hit its byte bound");

    assert!(matches!(
        &parser_error,
        LagunaOutputParserError::ToolArgumentsTooLarge { .. }
    ));
    assert_bounded_error(&parser_error);
    assert!(!parser_error.to_string().contains(&"R".repeat(512)));
}

#[test]
fn should_finish_streamed_reasoning_without_requiring_a_model_owned_closing_marker() {
    let mut output_parser = output_parser(true);

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

fn output_parser(generation_starts_in_reasoning: bool) -> LagunaOutputParser {
    let text_descriptor = SyntheticLagunaTextArtifact::extra_small_inline().normalize();
    LagunaOutputParser::new(
        &text_descriptor,
        &declared_literary_tools(),
        generation_starts_in_reasoning,
    )
    .expect("the supported poolside_v1 descriptor and declared tools should construct a parser")
}

fn assert_bounded_error(parser_error: &LagunaOutputParserError) {
    // Public malformed-output diagnostics must remain safe even when generated content is huge.
    assert!(parser_error.to_string().len() <= 256);
}
