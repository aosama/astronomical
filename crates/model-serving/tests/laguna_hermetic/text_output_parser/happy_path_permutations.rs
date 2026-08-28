//! Well-formed Poolside tool-call permutations.
//!
//! Every case keeps complete opening and closing tags. The suite proves the
//! parser serves those envelopes to the harness instead of only covering defects.

use astronomical_model_serving::LagunaOutputEvent;
use serde_json::Value;

use super::support::literary_output_parser;

const DECLARED_CHARACTER_FUNCTION: &str = "find_character";
const CHARACTER_NAME: &str = "Romeo";
const DECLARED_SCENE_FUNCTION: &str = "summarize_scene";
const UNDECLARED_FUNCTION_NAME: &str = "inspect_verse";
const ROMEO_ARGUMENTS_JSON: &str = r#"{"name":"Romeo"}"#;
const BALCONY_ARGUMENTS_JSON: &str = r#"{"scene":"balcony"}"#;
const EMPTY_ARGUMENTS_JSON: &str = "{}";

struct WellFormedCallCase {
    layout_description: &'static str,
    poolside_fragment: &'static str,
    function_name: &'static str,
    arguments_json: &'static str,
}

#[test]
fn should_emit_a_tool_call_for_every_well_formed_poolside_layout() {
    for well_formed_call in well_formed_single_call_cases() {
        let mut output_parser = literary_output_parser();
        let output_events = output_parser
            .push_fragment(well_formed_call.poolside_fragment)
            .unwrap_or_else(|parser_error| {
                panic!(
                    "well-formed layout {} must parse: {parser_error}; fragment={}",
                    well_formed_call.layout_description, well_formed_call.poolside_fragment
                )
            });
        assert_eq!(
            output_events,
            vec![LagunaOutputEvent::ToolCall {
                index: 0,
                function_name: well_formed_call.function_name.to_owned(),
                arguments_json: well_formed_call.arguments_json.to_owned(),
            }],
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
fn should_emit_sequential_well_formed_tool_calls_in_source_order() {
    let mut output_parser = literary_output_parser();
    let output_events = output_parser
        .push_fragment(
            "They are central.\
             <tool_call>find_character<arg_key>name</arg_key><arg_value>Romeo</arg_value></tool_call>\
             <tool_call>summarize_scene<arg_key>scene</arg_key><arg_value>balcony</arg_value></tool_call>\
             Then continue.",
        )
        .expect("text plus two well-formed calls must all stream");

    assert_eq!(
        output_events,
        vec![
            LagunaOutputEvent::TextDelta("They are central.".to_owned()),
            LagunaOutputEvent::ToolCall {
                index: 0,
                function_name: DECLARED_CHARACTER_FUNCTION.to_owned(),
                arguments_json: ROMEO_ARGUMENTS_JSON.to_owned(),
            },
            LagunaOutputEvent::ToolCall {
                index: 1,
                function_name: DECLARED_SCENE_FUNCTION.to_owned(),
                arguments_json: BALCONY_ARGUMENTS_JSON.to_owned(),
            },
            LagunaOutputEvent::TextDelta("Then continue.".to_owned()),
        ]
    );
}

#[test]
fn should_keep_argument_order_independent_for_required_and_extra_fields() {
    let mut extra_after_required = literary_output_parser();
    let mut extra_before_required = literary_output_parser();
    let extra_after_required_events = extra_after_required
        .push_fragment(
            "<tool_call>find_character\
             <arg_key>name</arg_key><arg_value>Romeo</arg_value>\
             <arg_key>description</arg_key><arg_value>Locate the character</arg_value>\
             </tool_call>",
        )
        .expect("required then extra arguments should parse");
    let extra_before_required_events = extra_before_required
        .push_fragment(
            "<tool_call>find_character\
             <arg_key>description</arg_key><arg_value>Locate the character</arg_value>\
             <arg_key>name</arg_key><arg_value>Romeo</arg_value>\
             </tool_call>",
        )
        .expect("extra then required arguments should parse");

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

fn tool_call_function_name(output_events: &[LagunaOutputEvent]) -> &str {
    match output_events {
        [LagunaOutputEvent::ToolCall { function_name, .. }] => function_name,
        other => panic!("expected one tool call, got {other:?}"),
    }
}

fn tool_call_argument_object(output_events: &[LagunaOutputEvent]) -> Value {
    match output_events {
        [LagunaOutputEvent::ToolCall { arguments_json, .. }] => {
            serde_json::from_str(arguments_json).expect("tool arguments should be JSON")
        }
        other => panic!("expected one tool call, got {other:?}"),
    }
}

fn well_formed_single_call_cases() -> Vec<WellFormedCallCase> {
    vec![
        WellFormedCallCase {
            layout_description: "jammed declared character call",
            poolside_fragment: "<tool_call>find_character<arg_key>name</arg_key><arg_value>Romeo</arg_value></tool_call>",
            function_name: DECLARED_CHARACTER_FUNCTION,
            arguments_json: ROMEO_ARGUMENTS_JSON,
        },
        WellFormedCallCase {
            layout_description: "newline after declared function name",
            poolside_fragment: "<tool_call>find_character\n<arg_key>name</arg_key><arg_value>Romeo</arg_value></tool_call>",
            function_name: DECLARED_CHARACTER_FUNCTION,
            arguments_json: ROMEO_ARGUMENTS_JSON,
        },
        WellFormedCallCase {
            layout_description: "newline after every declared tag",
            poolside_fragment: "<tool_call>find_character\n<arg_key>name</arg_key>\n<arg_value>Romeo</arg_value>\n</tool_call>",
            function_name: DECLARED_CHARACTER_FUNCTION,
            arguments_json: ROMEO_ARGUMENTS_JSON,
        },
        WellFormedCallCase {
            layout_description: "spaces between declared tags",
            poolside_fragment: "<tool_call>find_character <arg_key>name</arg_key> <arg_value>Romeo</arg_value> </tool_call>",
            function_name: DECLARED_CHARACTER_FUNCTION,
            arguments_json: ROMEO_ARGUMENTS_JSON,
        },
        WellFormedCallCase {
            layout_description: "tabs between declared tags",
            poolside_fragment: "<tool_call>find_character\t<arg_key>name</arg_key>\t<arg_value>Romeo</arg_value></tool_call>",
            function_name: DECLARED_CHARACTER_FUNCTION,
            arguments_json: ROMEO_ARGUMENTS_JSON,
        },
        WellFormedCallCase {
            layout_description: "leading and trailing whitespace inside the envelope",
            poolside_fragment: "<tool_call>\n  find_character\n  <arg_key>name</arg_key><arg_value>Romeo</arg_value>\n</tool_call>",
            function_name: DECLARED_CHARACTER_FUNCTION,
            arguments_json: ROMEO_ARGUMENTS_JSON,
        },
        WellFormedCallCase {
            layout_description: "declared scene call",
            poolside_fragment: "<tool_call>summarize_scene<arg_key>scene</arg_key><arg_value>balcony</arg_value></tool_call>",
            function_name: DECLARED_SCENE_FUNCTION,
            arguments_json: BALCONY_ARGUMENTS_JSON,
        },
        WellFormedCallCase {
            layout_description: "empty required string argument",
            poolside_fragment: "<tool_call>find_character<arg_key>name</arg_key><arg_value></arg_value></tool_call>",
            function_name: DECLARED_CHARACTER_FUNCTION,
            arguments_json: r#"{"name":""}"#,
        },
        WellFormedCallCase {
            layout_description: "multi-word character name",
            poolside_fragment: "<tool_call>find_character<arg_key>name</arg_key><arg_value>Romeo Montague</arg_value></tool_call>",
            function_name: DECLARED_CHARACTER_FUNCTION,
            arguments_json: r#"{"name":"Romeo Montague"}"#,
        },
        WellFormedCallCase {
            layout_description: "path-shaped extra argument on an undeclared function",
            poolside_fragment: "<tool_call>read<arg_key>path</arg_key><arg_value>romeo-and-juliet.md</arg_value></tool_call>",
            function_name: "read",
            arguments_json: r#"{"path":"romeo-and-juliet.md"}"#,
        },
        WellFormedCallCase {
            layout_description: "newline after undeclared read name",
            poolside_fragment: "<tool_call>read\n<arg_key>path</arg_key><arg_value>romeo-and-juliet.md</arg_value></tool_call>",
            function_name: "read",
            arguments_json: r#"{"path":"romeo-and-juliet.md"}"#,
        },
        WellFormedCallCase {
            layout_description: "hyphenated undeclared name with no arguments",
            poolside_fragment: "<tool_call>repo-discovery-guide</tool_call>",
            function_name: "repo-discovery-guide",
            arguments_json: EMPTY_ARGUMENTS_JSON,
        },
        WellFormedCallCase {
            layout_description: "undeclared function with a literary argument",
            poolside_fragment: "<tool_call>inspect_verse<arg_key>name</arg_key><arg_value>Romeo</arg_value></tool_call>",
            function_name: UNDECLARED_FUNCTION_NAME,
            arguments_json: ROMEO_ARGUMENTS_JSON,
        },
        WellFormedCallCase {
            layout_description: "undeclared function with two arguments",
            poolside_fragment: "<tool_call>inspect_verse<arg_key>name</arg_key><arg_value>Romeo</arg_value><arg_key>scene</arg_key><arg_value>balcony</arg_value></tool_call>",
            function_name: UNDECLARED_FUNCTION_NAME,
            arguments_json: r#"{"name":"Romeo","scene":"balcony"}"#,
        },
        WellFormedCallCase {
            layout_description: "JSON array extra argument stays structured",
            poolside_fragment: r#"<tool_call>inspect_verse<arg_key>quotes</arg_key><arg_value>["O Romeo","O Juliet"]</arg_value></tool_call>"#,
            function_name: UNDECLARED_FUNCTION_NAME,
            arguments_json: r#"{"quotes":["O Romeo","O Juliet"]}"#,
        },
        WellFormedCallCase {
            layout_description: "numeric extra argument stays a number",
            poolside_fragment: "<tool_call>inspect_verse<arg_key>act</arg_key><arg_value>2</arg_value></tool_call>",
            function_name: UNDECLARED_FUNCTION_NAME,
            arguments_json: r#"{"act":2}"#,
        },
        WellFormedCallCase {
            layout_description: "boolean extra argument stays a boolean",
            poolside_fragment: "<tool_call>inspect_verse<arg_key>tragic</arg_key><arg_value>true</arg_value></tool_call>",
            function_name: UNDECLARED_FUNCTION_NAME,
            arguments_json: r#"{"tragic":true}"#,
        },
        WellFormedCallCase {
            layout_description: "scene name with punctuation",
            poolside_fragment: "<tool_call>summarize_scene<arg_key>scene</arg_key><arg_value>Capulet's orchard</arg_value></tool_call>",
            function_name: DECLARED_SCENE_FUNCTION,
            arguments_json: r#"{"scene":"Capulet's orchard"}"#,
        },
    ]
}
