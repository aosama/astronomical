//! Well-formed Qwen3.5 tool-call permutations for dense and MoE.
//!
//! Both families share `Qwen3_5OutputParser`. Every case keeps complete
//! `<function=` / `<parameter=` tags so the harness receives a tool call.

use astronomical_model_serving::{Qwen3_5OutputEvent, Qwen3_5ToolCall};
use serde_json::Value;

use super::support::{
    BALCONY_ARGUMENTS_JSON, CHARACTER_NAME, DECLARED_CHARACTER_FUNCTION, DECLARED_SCENE_FUNCTION,
    EMPTY_ARGUMENTS_JSON, ROMEO_ARGUMENTS_JSON, UNDECLARED_FUNCTION_NAME, literary_output_parser,
};
use super::{TOOL_CALL_END, TOOL_CALL_START};

struct WellFormedCallCase {
    layout_description: &'static str,
    qwen_function_body: &'static str,
    function_name: &'static str,
    arguments_json: &'static str,
}

#[test]
fn should_emit_a_tool_call_for_every_well_formed_qwen_layout() {
    for well_formed_call in well_formed_single_call_cases() {
        let mut output_parser = literary_output_parser();
        let tool_call_xml = format!(
            "{TOOL_CALL_START}{}{TOOL_CALL_END}",
            well_formed_call.qwen_function_body
        );
        let output_events =
            output_parser
                .push_fragment(&tool_call_xml)
                .unwrap_or_else(|parser_error| {
                    panic!(
                        "well-formed layout {} must parse: {parser_error}; xml={tool_call_xml}",
                        well_formed_call.layout_description
                    )
                });
        assert_eq!(
            output_events,
            vec![Qwen3_5OutputEvent::ToolCall(Qwen3_5ToolCall {
                index: 0,
                function_name: well_formed_call.function_name.to_owned(),
                arguments_json: well_formed_call.arguments_json.to_owned(),
            })],
            "layout {}",
            well_formed_call.layout_description
        );
        assert!(
            output_parser
                .finish()
                .expect("a well-formed closed envelope should finish cleanly")
                .is_empty(),
            "layout {} left pending output",
            well_formed_call.layout_description
        );
    }
}

#[test]
fn should_emit_sequential_well_formed_qwen_tool_calls_in_source_order() {
    let mut output_parser = literary_output_parser();
    let output_events = output_parser
        .push_fragment(&format!(
            "They are central.{TOOL_CALL_START}<function=find_character><parameter=name>Romeo</parameter></function>{TOOL_CALL_END}{TOOL_CALL_START}<function=summarize_scene><parameter=scene>balcony</parameter></function>{TOOL_CALL_END}Then continue."
        ))
        .expect("text plus two well-formed Qwen calls must all stream");

    assert_eq!(
        output_events,
        vec![
            Qwen3_5OutputEvent::TextDelta("They are central.".to_owned()),
            Qwen3_5OutputEvent::ToolCall(Qwen3_5ToolCall {
                index: 0,
                function_name: DECLARED_CHARACTER_FUNCTION.to_owned(),
                arguments_json: ROMEO_ARGUMENTS_JSON.to_owned(),
            }),
            Qwen3_5OutputEvent::ToolCall(Qwen3_5ToolCall {
                index: 1,
                function_name: DECLARED_SCENE_FUNCTION.to_owned(),
                arguments_json: BALCONY_ARGUMENTS_JSON.to_owned(),
            }),
            Qwen3_5OutputEvent::TextDelta("Then continue.".to_owned()),
        ]
    );
}

#[test]
fn should_keep_qwen_argument_order_independent_for_required_and_extra_fields() {
    let mut extra_after_required = literary_output_parser();
    let mut extra_before_required = literary_output_parser();
    let extra_after_required_events = extra_after_required
        .push_fragment(&format!(
            "{TOOL_CALL_START}<function=find_character><parameter=name>Romeo</parameter><parameter=description>Locate the character</parameter></function>{TOOL_CALL_END}"
        ))
        .expect("required then extra Qwen parameters should parse");
    let extra_before_required_events = extra_before_required
        .push_fragment(&format!(
            "{TOOL_CALL_START}<function=find_character><parameter=description>Locate the character</parameter><parameter=name>Romeo</parameter></function>{TOOL_CALL_END}"
        ))
        .expect("extra then required Qwen parameters should parse");

    assert_eq!(
        tool_call_function_name(&extra_after_required_events),
        DECLARED_CHARACTER_FUNCTION
    );
    assert_eq!(
        tool_call_function_name(&extra_before_required_events),
        DECLARED_CHARACTER_FUNCTION
    );
    assert_eq!(
        tool_call_argument_object(&extra_after_required_events),
        tool_call_argument_object(&extra_before_required_events)
    );
    assert_eq!(
        tool_call_argument_object(&extra_after_required_events)["name"],
        Value::String(CHARACTER_NAME.to_owned())
    );
}

fn well_formed_single_call_cases() -> Vec<WellFormedCallCase> {
    vec![
        WellFormedCallCase {
            layout_description: "jammed declared character call",
            qwen_function_body: "<function=find_character><parameter=name>Romeo</parameter></function>",
            function_name: DECLARED_CHARACTER_FUNCTION,
            arguments_json: ROMEO_ARGUMENTS_JSON,
        },
        WellFormedCallCase {
            layout_description: "pretty-printed declared character call",
            qwen_function_body: "\n<function=find_character>\n<parameter=name>\nRomeo\n</parameter>\n</function>\n",
            function_name: DECLARED_CHARACTER_FUNCTION,
            arguments_json: ROMEO_ARGUMENTS_JSON,
        },
        WellFormedCallCase {
            layout_description: "spaces between Qwen tags",
            qwen_function_body: "<function=find_character> <parameter=name>Romeo</parameter> </function>",
            function_name: DECLARED_CHARACTER_FUNCTION,
            arguments_json: ROMEO_ARGUMENTS_JSON,
        },
        WellFormedCallCase {
            layout_description: "tabs between Qwen tags",
            qwen_function_body: "<function=find_character>\t<parameter=name>Romeo</parameter></function>",
            function_name: DECLARED_CHARACTER_FUNCTION,
            arguments_json: ROMEO_ARGUMENTS_JSON,
        },
        WellFormedCallCase {
            layout_description: "declared scene call",
            qwen_function_body: "<function=summarize_scene><parameter=scene>balcony</parameter></function>",
            function_name: DECLARED_SCENE_FUNCTION,
            arguments_json: BALCONY_ARGUMENTS_JSON,
        },
        WellFormedCallCase {
            layout_description: "empty required string argument",
            qwen_function_body: "<function=find_character><parameter=name></parameter></function>",
            function_name: DECLARED_CHARACTER_FUNCTION,
            arguments_json: r#"{"name":""}"#,
        },
        WellFormedCallCase {
            layout_description: "multi-word character name",
            qwen_function_body: "<function=find_character><parameter=name>Romeo Montague</parameter></function>",
            function_name: DECLARED_CHARACTER_FUNCTION,
            arguments_json: r#"{"name":"Romeo Montague"}"#,
        },
        WellFormedCallCase {
            layout_description: "path-shaped extra argument on an undeclared function",
            qwen_function_body: "<function=read><parameter=path>romeo-and-juliet.md</parameter></function>",
            function_name: "read",
            arguments_json: r#"{"path":"romeo-and-juliet.md"}"#,
        },
        WellFormedCallCase {
            layout_description: "hyphenated undeclared name with no arguments",
            qwen_function_body: "<function=repo-discovery-guide></function>",
            function_name: "repo-discovery-guide",
            arguments_json: EMPTY_ARGUMENTS_JSON,
        },
        WellFormedCallCase {
            layout_description: "undeclared function with a literary argument",
            qwen_function_body: "<function=inspect_verse><parameter=name>Romeo</parameter></function>",
            function_name: UNDECLARED_FUNCTION_NAME,
            arguments_json: ROMEO_ARGUMENTS_JSON,
        },
        WellFormedCallCase {
            layout_description: "undeclared function with two arguments",
            qwen_function_body: "<function=inspect_verse><parameter=name>Romeo</parameter><parameter=scene>balcony</parameter></function>",
            function_name: UNDECLARED_FUNCTION_NAME,
            arguments_json: r#"{"name":"Romeo","scene":"balcony"}"#,
        },
        WellFormedCallCase {
            layout_description: "JSON array extra argument stays structured",
            qwen_function_body: r#"<function=inspect_verse><parameter=quotes>["O Romeo","O Juliet"]</parameter></function>"#,
            function_name: UNDECLARED_FUNCTION_NAME,
            arguments_json: r#"{"quotes":["O Romeo","O Juliet"]}"#,
        },
        WellFormedCallCase {
            layout_description: "numeric extra argument stays a number",
            qwen_function_body: "<function=inspect_verse><parameter=act>2</parameter></function>",
            function_name: UNDECLARED_FUNCTION_NAME,
            arguments_json: r#"{"act":2}"#,
        },
        WellFormedCallCase {
            layout_description: "boolean extra argument stays a boolean",
            qwen_function_body: "<function=inspect_verse><parameter=tragic>true</parameter></function>",
            function_name: UNDECLARED_FUNCTION_NAME,
            arguments_json: r#"{"tragic":true}"#,
        },
        WellFormedCallCase {
            layout_description: "scene name with punctuation",
            qwen_function_body: "<function=summarize_scene><parameter=scene>Capulet's orchard</parameter></function>",
            function_name: DECLARED_SCENE_FUNCTION,
            arguments_json: r#"{"scene":"Capulet's orchard"}"#,
        },
    ]
}

fn tool_call_function_name(output_events: &[Qwen3_5OutputEvent]) -> &str {
    match output_events {
        [Qwen3_5OutputEvent::ToolCall(tool_call)] => tool_call.function_name.as_str(),
        other => panic!("expected one tool call, got {other:?}"),
    }
}

fn tool_call_argument_object(output_events: &[Qwen3_5OutputEvent]) -> Value {
    match output_events {
        [Qwen3_5OutputEvent::ToolCall(tool_call)] => {
            serde_json::from_str(&tool_call.arguments_json).expect("tool arguments should be JSON")
        }
        other => panic!("expected one tool call, got {other:?}"),
    }
}
