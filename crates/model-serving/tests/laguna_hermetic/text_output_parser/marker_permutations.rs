//! Permutation coverage for Poolside tool-call marker defects.
//!
//! Product contract: a closed `<tool_call>…</tool_call>` fails open. Unclosed envelopes
//! stay pending while tokens can still arrive, then salvage at generation end.
//! Resource bounds cap memory and salvage; they never abort the stream.
//! Empty names stream as text instead of aborting generation.

use astronomical_model_serving::{LagunaOutputEvent, LagunaOutputParserError};

use super::support::literary_output_parser;

const DECLARED_FUNCTION_NAME: &str = "find_character";
const UNDECLARED_FUNCTION_NAME: &str = "inspect_verse";
const ROMEO_ARGUMENTS_JSON: &str = r#"{"name":"Romeo"}"#;
const EMPTY_ARGUMENTS_JSON: &str = "{}";

#[derive(Clone, Copy, Debug)]
enum ClosedEnvelopeExpectation {
    ToolCall {
        function_name: &'static str,
        arguments_json: &'static str,
    },
    VisibleText,
}

#[test]
fn should_honor_the_fail_open_contract_for_every_closed_marker_defect() {
    for closed_envelope_case in closed_envelope_cases() {
        let mut output_parser = literary_output_parser();
        let parse_outcome = output_parser.push_fragment(closed_envelope_case.poolside_fragment);
        assert_closed_envelope_outcome(
            closed_envelope_case.marker_defect,
            closed_envelope_case.poolside_fragment,
            parse_outcome,
            closed_envelope_case.expectation,
        );
        assert!(
            output_parser
                .finish()
                .expect("a completed closed envelope should finish cleanly")
                .is_empty(),
            "defect {} left pending parser output",
            closed_envelope_case.marker_defect
        );
    }
}

#[test]
fn should_forward_unclosed_envelopes_when_generation_finishes() {
    for unclosed_envelope_case in unclosed_envelope_cases() {
        let mut output_parser = literary_output_parser();
        let push_outcome = output_parser
            .push_fragment(unclosed_envelope_case.poolside_fragment)
            .unwrap_or_else(|parser_error| {
                panic!(
                    "unclosed defect {} should not fail during push: {parser_error}; fragment={}",
                    unclosed_envelope_case.marker_defect, unclosed_envelope_case.poolside_fragment
                )
            });
        assert!(
            !push_outcome
                .iter()
                .any(|output_event| matches!(output_event, LagunaOutputEvent::ToolCall { .. })),
            "unclosed defect {} emitted a tool call during push: {push_outcome:?}",
            unclosed_envelope_case.marker_defect
        );
        let finish_events = output_parser.finish().unwrap_or_else(|parser_error| {
            panic!(
                "unclosed defect {} must not abort generation: {parser_error}; fragment={}",
                unclosed_envelope_case.marker_defect, unclosed_envelope_case.poolside_fragment
            )
        });
        assert_unclosed_finish_outcome(
            unclosed_envelope_case.marker_defect,
            unclosed_envelope_case.poolside_fragment,
            finish_events,
            unclosed_envelope_case.expectation,
        );
    }
}

#[test]
fn should_treat_missing_tool_call_open_as_visible_text_not_a_tool_call() {
    for visible_text_case in missing_tool_call_open_cases() {
        let mut output_parser = literary_output_parser();
        let output_events = output_parser
            .push_fragment(visible_text_case.poolside_fragment)
            .unwrap_or_else(|parser_error| {
                panic!(
                    "missing-open defect {} should stream as text: {parser_error}; fragment={}",
                    visible_text_case.marker_defect, visible_text_case.poolside_fragment
                )
            });
        assert!(
            !output_events
                .iter()
                .any(|output_event| matches!(output_event, LagunaOutputEvent::ToolCall { .. })),
            "missing-open defect {} emitted a tool call: {output_events:?}",
            visible_text_case.marker_defect
        );
        assert!(
            output_events.iter().any(|output_event| matches!(
                output_event,
                LagunaOutputEvent::TextDelta(text)
                    if text.contains(DECLARED_FUNCTION_NAME)
                        || text.contains(UNDECLARED_FUNCTION_NAME)
                        || text.contains("arg_key")
            )),
            "missing-open defect {} did not stream the model text: {output_events:?}",
            visible_text_case.marker_defect
        );
    }
}

#[test]
fn should_forward_every_logged_live_tool_call_abort_as_a_harness_tool_call() {
    // Shapes copied from Development worker aborts, with literary fixture paths.
    for live_abort_case in logged_live_tool_call_abort_cases() {
        let mut output_parser = literary_output_parser();
        let output_events = output_parser
            .push_fragment(live_abort_case.poolside_fragment)
            .unwrap_or_else(|parser_error| {
                panic!(
                    "logged abort {} still fail-closed: {parser_error}; fragment={}",
                    live_abort_case.marker_defect, live_abort_case.poolside_fragment
                )
            });
        assert_eq!(
            output_events,
            vec![LagunaOutputEvent::ToolCall {
                index: 0,
                function_name: live_abort_case.function_name.to_owned(),
                arguments_json: live_abort_case.arguments_json.to_owned(),
            }],
            "logged abort {}",
            live_abort_case.marker_defect
        );
        assert!(
            output_parser
                .finish()
                .expect("a forwarded logged abort should finish cleanly")
                .is_empty()
        );
    }
}

struct LoggedLiveAbortCase {
    marker_defect: &'static str,
    poolside_fragment: &'static str,
    function_name: &'static str,
    arguments_json: &'static str,
}

fn logged_live_tool_call_abort_cases() -> Vec<LoggedLiveAbortCase> {
    vec![
        LoggedLiveAbortCase {
            marker_defect: "undeclared skill name with no arguments",
            poolside_fragment: "<tool_call>repo-discovery-guide</tool_call>",
            function_name: "repo-discovery-guide",
            arguments_json: EMPTY_ARGUMENTS_JSON,
        },
        LoggedLiveAbortCase {
            marker_defect: "read with missing < on arg_key open",
            poolside_fragment: "<tool_call>read\narg_key>path</arg_key><arg_value>romeo-and-juliet.md</arg_value></tool_call>",
            function_name: "read",
            arguments_json: r#"{"path":"romeo-and-juliet.md"}"#,
        },
        LoggedLiveAbortCase {
            marker_defect: "read with _key slop then a well-formed value pair",
            poolside_fragment: "<tool_call>read\n<_key>argument-key</arg_key><arg_value>path</arg_value><arg_key>value</arg_key><arg_value>romeo-and-juliet.md</arg_value></tool_call>",
            function_name: "read",
            arguments_json: r#"{"value":"romeo-and-juliet.md"}"#,
        },
        LoggedLiveAbortCase {
            marker_defect: "jammed well-formed read without a newline after the function name",
            poolside_fragment: "<tool_call>read<arg_key>path</arg_key><arg_value>romeo-and-juliet.md</arg_value></tool_call>",
            function_name: "read",
            arguments_json: r#"{"path":"romeo-and-juliet.md"}"#,
        },
    ]
}

struct MarkerDefectCase {
    marker_defect: &'static str,
    poolside_fragment: &'static str,
    expectation: ClosedEnvelopeExpectation,
}

struct UnclosedEnvelopeCase {
    marker_defect: &'static str,
    poolside_fragment: &'static str,
    expectation: ClosedEnvelopeExpectation,
}

fn closed_envelope_cases() -> Vec<MarkerDefectCase> {
    vec![
        MarkerDefectCase {
            marker_defect: "canonical declared call",
            poolside_fragment: "<tool_call>find_character<arg_key>name</arg_key><arg_value>Romeo</arg_value></tool_call>",
            expectation: tool_call(DECLARED_FUNCTION_NAME, ROMEO_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "newline between declared name and arguments",
            poolside_fragment: "<tool_call>find_character\n<arg_key>name</arg_key><arg_value>Romeo</arg_value></tool_call>",
            expectation: tool_call(DECLARED_FUNCTION_NAME, ROMEO_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "spaces between declared argument tags",
            poolside_fragment: "<tool_call>find_character<arg_key>name</arg_key> <arg_value>Romeo</arg_value></tool_call>",
            expectation: tool_call(DECLARED_FUNCTION_NAME, ROMEO_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "missing < on declared arg_key open",
            poolside_fragment: "<tool_call>find_characterarg_key>name</arg_key><arg_value>Romeo</arg_value></tool_call>",
            expectation: tool_call(DECLARED_FUNCTION_NAME, ROMEO_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "missing < on declared arg_value open",
            poolside_fragment: "<tool_call>find_character<arg_key>name</arg_key>arg_value>Romeo</arg_value></tool_call>",
            expectation: tool_call(DECLARED_FUNCTION_NAME, ROMEO_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "missing < on both declared argument opens",
            poolside_fragment: "<tool_call>find_characterarg_key>name</arg_key>arg_value>Romeo</arg_value></tool_call>",
            expectation: tool_call(DECLARED_FUNCTION_NAME, ROMEO_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "newline and missing < on declared arg_key open",
            poolside_fragment: "<tool_call>find_character\narg_key>name</arg_key><arg_value>Romeo</arg_value></tool_call>",
            expectation: tool_call(DECLARED_FUNCTION_NAME, ROMEO_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "missing > on declared arg_key open",
            poolside_fragment: "<tool_call>find_character<arg_keyname</arg_key><arg_value>Romeo</arg_value></tool_call>",
            expectation: tool_call(DECLARED_FUNCTION_NAME, EMPTY_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "missing declared arg_key closing tag",
            poolside_fragment: "<tool_call>find_character<arg_key>name<arg_value>Romeo</arg_value></tool_call>",
            expectation: tool_call(DECLARED_FUNCTION_NAME, EMPTY_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "missing declared arg_value closing tag",
            poolside_fragment: "<tool_call>find_character<arg_key>name</arg_key><arg_value>Romeo</tool_call>",
            expectation: tool_call(DECLARED_FUNCTION_NAME, EMPTY_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "missing declared arg_key opening tag glues the argument name",
            poolside_fragment: "<tool_call>find_charactername</arg_key><arg_value>Romeo</arg_value></tool_call>",
            expectation: tool_call("find_charactername", EMPTY_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "missing declared arg_value opening tag",
            poolside_fragment: "<tool_call>find_character<arg_key>name</arg_key>Romeo</arg_value></tool_call>",
            expectation: tool_call(DECLARED_FUNCTION_NAME, EMPTY_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "missing both declared argument opening tags glues the argument name",
            poolside_fragment: "<tool_call>find_charactername</arg_key>Romeo</arg_value></tool_call>",
            expectation: tool_call("find_charactername", EMPTY_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "missing > on declared arg_key open and missing arg_value close",
            poolside_fragment: "<tool_call>find_character<arg_keyname</arg_key><arg_value>Romeo</tool_call>",
            expectation: tool_call(DECLARED_FUNCTION_NAME, EMPTY_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "canonical undeclared call",
            poolside_fragment: "<tool_call>inspect_verse<arg_key>name</arg_key><arg_value>Romeo</arg_value></tool_call>",
            expectation: tool_call(UNDECLARED_FUNCTION_NAME, ROMEO_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "undeclared name only",
            poolside_fragment: "<tool_call>inspect_verse</tool_call>",
            expectation: tool_call(UNDECLARED_FUNCTION_NAME, EMPTY_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "undeclared with missing < on arg_key open",
            poolside_fragment: "<tool_call>inspect_verse\narg_key>name</arg_key><arg_value>Romeo</arg_value></tool_call>",
            expectation: tool_call(UNDECLARED_FUNCTION_NAME, ROMEO_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "undeclared with missing < on both argument opens",
            poolside_fragment: "<tool_call>inspect_versearg_key>name</arg_key>arg_value>Romeo</arg_value></tool_call>",
            expectation: tool_call(UNDECLARED_FUNCTION_NAME, ROMEO_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "undeclared with missing > on arg_key open",
            poolside_fragment: "<tool_call>inspect_verse<arg_keyname</arg_key><arg_value>Romeo</arg_value></tool_call>",
            expectation: tool_call(UNDECLARED_FUNCTION_NAME, EMPTY_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "undeclared with missing arg_key close",
            poolside_fragment: "<tool_call>inspect_verse<arg_key>name<arg_value>Romeo</arg_value></tool_call>",
            expectation: tool_call(UNDECLARED_FUNCTION_NAME, EMPTY_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "undeclared with missing arg_value close",
            poolside_fragment: "<tool_call>inspect_verse<arg_key>name</arg_key><arg_value>Romeo</tool_call>",
            expectation: tool_call(UNDECLARED_FUNCTION_NAME, EMPTY_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "undeclared with missing arg_key open tag glues the argument name",
            poolside_fragment: "<tool_call>inspect_versename</arg_key><arg_value>Romeo</arg_value></tool_call>",
            expectation: tool_call("inspect_versename", EMPTY_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "undeclared with missing arg_value open tag",
            poolside_fragment: "<tool_call>inspect_verse<arg_key>name</arg_key>Romeo</arg_value></tool_call>",
            expectation: tool_call(UNDECLARED_FUNCTION_NAME, EMPTY_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "undeclared with missing both argument opens glues the argument name",
            poolside_fragment: "<tool_call>inspect_versename</arg_key>Romeo</arg_value></tool_call>",
            expectation: tool_call("inspect_versename", EMPTY_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "undeclared with broken _key slop",
            poolside_fragment: "<tool_call>inspect_verse\n<_key>argument-key</arg_key><arg_value>path</arg_value></tool_call>",
            expectation: tool_call(UNDECLARED_FUNCTION_NAME, EMPTY_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "empty function name",
            poolside_fragment: "<tool_call></tool_call>",
            expectation: ClosedEnvelopeExpectation::VisibleText,
        },
        MarkerDefectCase {
            marker_defect: "arguments without a function name",
            poolside_fragment: "<tool_call><arg_key>name</arg_key><arg_value>Romeo</arg_value></tool_call>",
            expectation: ClosedEnvelopeExpectation::VisibleText,
        },
        MarkerDefectCase {
            marker_defect: "nested argument marker in the key",
            poolside_fragment: "<tool_call>find_character<arg_key>name<arg_key>nested</arg_key></arg_key><arg_value>Romeo</arg_value></tool_call>",
            expectation: tool_call(DECLARED_FUNCTION_NAME, EMPTY_ARGUMENTS_JSON),
        },
        MarkerDefectCase {
            marker_defect: "duplicate declared argument",
            poolside_fragment: "<tool_call>find_character<arg_key>name</arg_key><arg_value>Romeo</arg_value><arg_key>name</arg_key><arg_value>Juliet</arg_value></tool_call>",
            expectation: tool_call(DECLARED_FUNCTION_NAME, r#"{"name":"Juliet"}"#),
        },
    ]
}

fn unclosed_envelope_cases() -> Vec<UnclosedEnvelopeCase> {
    vec![
        UnclosedEnvelopeCase {
            marker_defect: "missing tool_call close",
            poolside_fragment: "<tool_call>find_character<arg_key>name</arg_key><arg_value>Romeo</arg_value>",
            expectation: tool_call(DECLARED_FUNCTION_NAME, ROMEO_ARGUMENTS_JSON),
        },
        UnclosedEnvelopeCase {
            marker_defect: "missing < on tool_call close",
            poolside_fragment: "<tool_call>find_character<arg_key>name</arg_key><arg_value>Romeo</arg_value>/tool_call>",
            expectation: tool_call(DECLARED_FUNCTION_NAME, ROMEO_ARGUMENTS_JSON),
        },
        UnclosedEnvelopeCase {
            marker_defect: "missing > on tool_call close",
            poolside_fragment: "<tool_call>find_character<arg_key>name</arg_key><arg_value>Romeo</arg_value></tool_call",
            expectation: tool_call(DECLARED_FUNCTION_NAME, ROMEO_ARGUMENTS_JSON),
        },
        UnclosedEnvelopeCase {
            marker_defect: "truncated tool_call close",
            poolside_fragment: "<tool_call>inspect_verse<arg_key>name</arg_key><arg_value>Romeo</arg_value></tool_c",
            expectation: tool_call(UNDECLARED_FUNCTION_NAME, ROMEO_ARGUMENTS_JSON),
        },
        UnclosedEnvelopeCase {
            marker_defect: "unclosed undeclared call",
            poolside_fragment: "<tool_call>inspect_verse",
            expectation: tool_call(UNDECLARED_FUNCTION_NAME, EMPTY_ARGUMENTS_JSON),
        },
        UnclosedEnvelopeCase {
            marker_defect: "nameless unclosed arguments",
            poolside_fragment: "<tool_call><arg_key>name</arg_key><arg_value>Romeo</arg_value>",
            expectation: ClosedEnvelopeExpectation::VisibleText,
        },
    ]
}

fn missing_tool_call_open_cases() -> Vec<UnclosedEnvelopeCase> {
    vec![
        UnclosedEnvelopeCase {
            marker_defect: "missing tool_call open entirely",
            poolside_fragment: "find_character<arg_key>name</arg_key><arg_value>Romeo</arg_value></tool_call>",
            expectation: ClosedEnvelopeExpectation::VisibleText,
        },
        UnclosedEnvelopeCase {
            marker_defect: "missing < on tool_call open",
            poolside_fragment: "tool_call>inspect_verse<arg_key>name</arg_key><arg_value>Romeo</arg_value></tool_call>",
            expectation: ClosedEnvelopeExpectation::VisibleText,
        },
        UnclosedEnvelopeCase {
            marker_defect: "missing > on tool_call open",
            poolside_fragment: "<tool_call find_character<arg_key>name</arg_key><arg_value>Romeo</arg_value></tool_call>",
            expectation: ClosedEnvelopeExpectation::VisibleText,
        },
    ]
}

fn tool_call(
    function_name: &'static str,
    arguments_json: &'static str,
) -> ClosedEnvelopeExpectation {
    ClosedEnvelopeExpectation::ToolCall {
        function_name,
        arguments_json,
    }
}

fn assert_unclosed_finish_outcome(
    marker_defect: &str,
    poolside_fragment: &str,
    finish_events: Vec<LagunaOutputEvent>,
    expectation: ClosedEnvelopeExpectation,
) {
    match expectation {
        ClosedEnvelopeExpectation::VisibleText => {
            assert!(
                !finish_events
                    .iter()
                    .any(|output_event| matches!(output_event, LagunaOutputEvent::ToolCall { .. })),
                "defect={marker_defect}; fragment={poolside_fragment} emitted a tool call: {finish_events:?}"
            );
            assert!(
                finish_events.iter().any(|output_event| {
                    matches!(output_event, LagunaOutputEvent::TextDelta(text) if !text.is_empty())
                }),
                "defect={marker_defect}; fragment={poolside_fragment} dropped nameless tool-call text: {finish_events:?}"
            );
        }
        tool_call_expectation => {
            assert_closed_envelope_outcome(
                marker_defect,
                poolside_fragment,
                Ok(finish_events),
                tool_call_expectation,
            );
        }
    }
}

fn assert_closed_envelope_outcome(
    marker_defect: &str,
    poolside_fragment: &str,
    parse_outcome: Result<Vec<LagunaOutputEvent>, LagunaOutputParserError>,
    expectation: ClosedEnvelopeExpectation,
) {
    let failure_context = format!("defect={marker_defect}; fragment={poolside_fragment}");
    match expectation {
        ClosedEnvelopeExpectation::ToolCall {
            function_name,
            arguments_json,
        } => {
            let output_events = parse_outcome.unwrap_or_else(|parser_error| {
                panic!("{failure_context} should emit a tool call, got {parser_error}")
            });
            assert_eq!(
                output_events,
                vec![LagunaOutputEvent::ToolCall {
                    index: 0,
                    function_name: function_name.to_owned(),
                    arguments_json: arguments_json.to_owned(),
                }],
                "{failure_context}"
            );
        }
        ClosedEnvelopeExpectation::VisibleText => {
            let output_events = parse_outcome.unwrap_or_else(|parser_error| {
                panic!("{failure_context} should not abort generation, got {parser_error}")
            });
            assert!(
                !output_events
                    .iter()
                    .any(|output_event| matches!(output_event, LagunaOutputEvent::ToolCall { .. })),
                "{failure_context} emitted a tool call: {output_events:?}"
            );
        }
    }
}
