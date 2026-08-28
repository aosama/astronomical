//! Qwen3.5 marker-defect permutations for dense and MoE.
//!
//! Closed envelopes with a usable function name must reach the harness. Salvage
//! a missing `<` on `<function=` / `<parameter=`. Unclosed envelopes stay
//! incomplete. Empty names stay malformed.

use astronomical_model_serving::{Qwen3_5OutputEvent, Qwen3_5OutputParserError, Qwen3_5ToolCall};

use super::support::{
    DECLARED_CHARACTER_FUNCTION, EMPTY_ARGUMENTS_JSON, ROMEO_ARGUMENTS_JSON,
    UNDECLARED_FUNCTION_NAME, literary_output_parser,
};
use super::{TOOL_CALL_END, TOOL_CALL_START};

#[derive(Clone, Copy, Debug)]
enum ClosedEnvelopeExpectation {
    ToolCall {
        function_name: &'static str,
        arguments_json: &'static str,
    },
    VisibleText,
}

struct MarkerDefectCase {
    marker_defect: &'static str,
    qwen_envelope: String,
    expectation: ClosedEnvelopeExpectation,
}

struct UnclosedEnvelopeCase {
    marker_defect: &'static str,
    qwen_fragment: String,
}

#[test]
fn should_honor_the_fail_open_contract_for_every_closed_qwen_marker_defect() {
    for closed_envelope_case in closed_envelope_cases() {
        let mut output_parser = literary_output_parser();
        let parse_outcome = output_parser.push_fragment(&closed_envelope_case.qwen_envelope);
        assert_closed_envelope_outcome(
            closed_envelope_case.marker_defect,
            &closed_envelope_case.qwen_envelope,
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
fn should_keep_unclosed_qwen_envelopes_incomplete_instead_of_inventing_a_tool_call() {
    for unclosed_envelope_case in unclosed_envelope_cases() {
        let mut output_parser = literary_output_parser();
        let push_outcome = output_parser
            .push_fragment(&unclosed_envelope_case.qwen_fragment)
            .unwrap_or_else(|parser_error| {
                panic!(
                    "unclosed defect {} should not fail during push: {parser_error}; fragment={}",
                    unclosed_envelope_case.marker_defect, unclosed_envelope_case.qwen_fragment
                )
            });
        assert!(
            !push_outcome
                .iter()
                .any(|output_event| matches!(output_event, Qwen3_5OutputEvent::ToolCall(_))),
            "unclosed defect {} emitted a tool call during push: {push_outcome:?}",
            unclosed_envelope_case.marker_defect
        );
        let finish_error = output_parser.finish().expect_err(&format!(
            "unclosed defect {} must stay incomplete at finish; fragment={}",
            unclosed_envelope_case.marker_defect, unclosed_envelope_case.qwen_fragment
        ));
        assert!(
            matches!(
                finish_error,
                Qwen3_5OutputParserError::UnclosedToolCall
                    | Qwen3_5OutputParserError::IncompleteControlMarker
            ),
            "unclosed defect {} finished with {finish_error}",
            unclosed_envelope_case.marker_defect
        );
    }
}

#[test]
fn should_treat_missing_qwen_tool_call_open_as_visible_text_not_a_tool_call() {
    for visible_text_case in missing_tool_call_open_cases() {
        let mut output_parser = literary_output_parser();
        let output_events = output_parser
            .push_fragment(&visible_text_case.qwen_fragment)
            .unwrap_or_else(|parser_error| {
                panic!(
                    "missing-open defect {} should stream as text: {parser_error}; fragment={}",
                    visible_text_case.marker_defect, visible_text_case.qwen_fragment
                )
            });
        assert!(
            !output_events
                .iter()
                .any(|output_event| matches!(output_event, Qwen3_5OutputEvent::ToolCall(_))),
            "missing-open defect {} emitted a tool call: {output_events:?}",
            visible_text_case.marker_defect
        );
        assert!(
            output_events.iter().any(|output_event| matches!(
                output_event,
                Qwen3_5OutputEvent::TextDelta(text)
                    if text.contains(DECLARED_CHARACTER_FUNCTION)
                        || text.contains(UNDECLARED_FUNCTION_NAME)
                        || text.contains("parameter")
                        || text.contains("function=")
            )),
            "missing-open defect {} did not stream the model text: {output_events:?}",
            visible_text_case.marker_defect
        );
    }
}

fn closed_envelope_cases() -> Vec<MarkerDefectCase> {
    vec![
        closed_case(
            "canonical declared call",
            "<function=find_character><parameter=name>Romeo</parameter></function>",
            tool_call(DECLARED_CHARACTER_FUNCTION, ROMEO_ARGUMENTS_JSON),
        ),
        closed_case(
            "missing < on function open",
            "function=find_character><parameter=name>Romeo</parameter></function>",
            tool_call(DECLARED_CHARACTER_FUNCTION, ROMEO_ARGUMENTS_JSON),
        ),
        closed_case(
            "missing < on parameter open",
            "<function=find_character>parameter=name>Romeo</parameter></function>",
            tool_call(DECLARED_CHARACTER_FUNCTION, ROMEO_ARGUMENTS_JSON),
        ),
        closed_case(
            "missing < on function and parameter opens",
            "function=find_character>parameter=name>Romeo</parameter></function>",
            tool_call(DECLARED_CHARACTER_FUNCTION, ROMEO_ARGUMENTS_JSON),
        ),
        closed_case(
            "missing > on function open",
            "<function=find_character\n<parameter=name>Romeo</parameter></function>",
            tool_call(DECLARED_CHARACTER_FUNCTION, ROMEO_ARGUMENTS_JSON),
        ),
        closed_case(
            "missing function close",
            "<function=find_character><parameter=name>Romeo</parameter>",
            tool_call(DECLARED_CHARACTER_FUNCTION, ROMEO_ARGUMENTS_JSON),
        ),
        closed_case(
            "missing parameter close",
            "<function=find_character><parameter=name>Romeo</function>",
            tool_call(DECLARED_CHARACTER_FUNCTION, ROMEO_ARGUMENTS_JSON),
        ),
        closed_case(
            "undeclared with missing < on parameter open",
            "<function=inspect_verse>\nparameter=name>Romeo</parameter></function>",
            tool_call(UNDECLARED_FUNCTION_NAME, ROMEO_ARGUMENTS_JSON),
        ),
        closed_case(
            "undeclared name only",
            "<function=inspect_verse></function>",
            tool_call(UNDECLARED_FUNCTION_NAME, EMPTY_ARGUMENTS_JSON),
        ),
        closed_case(
            "undeclared with _key slop then a well-formed value pair",
            "<function=read>\n<_key>argument-key</parameter><parameter=value>romeo-and-juliet.md</parameter></function>",
            tool_call("read", r#"{"value":"romeo-and-juliet.md"}"#),
        ),
        closed_case(
            "empty function name",
            "<function=></function>",
            ClosedEnvelopeExpectation::VisibleText,
        ),
        closed_case(
            "arguments without a function marker",
            "<parameter=name>Romeo</parameter>",
            ClosedEnvelopeExpectation::VisibleText,
        ),
    ]
}

fn unclosed_envelope_cases() -> Vec<UnclosedEnvelopeCase> {
    vec![
        UnclosedEnvelopeCase {
            marker_defect: "missing tool_call close",
            qwen_fragment: format!(
                "{TOOL_CALL_START}<function=find_character><parameter=name>Romeo</parameter></function>"
            ),
        },
        UnclosedEnvelopeCase {
            marker_defect: "missing < on tool_call close",
            qwen_fragment: format!(
                "{TOOL_CALL_START}<function=find_character><parameter=name>Romeo</parameter></function>/tool_call>"
            ),
        },
        UnclosedEnvelopeCase {
            marker_defect: "truncated tool_call close",
            qwen_fragment: format!(
                "{TOOL_CALL_START}<function=inspect_verse><parameter=name>Romeo</parameter></function></tool_c"
            ),
        },
        UnclosedEnvelopeCase {
            marker_defect: "unclosed undeclared call",
            qwen_fragment: format!("{TOOL_CALL_START}<function=inspect_verse>"),
        },
    ]
}

fn missing_tool_call_open_cases() -> Vec<UnclosedEnvelopeCase> {
    vec![
        UnclosedEnvelopeCase {
            marker_defect: "missing tool_call open entirely",
            qwen_fragment: format!(
                "<function=find_character><parameter=name>Romeo</parameter></function>{TOOL_CALL_END}"
            ),
        },
        UnclosedEnvelopeCase {
            marker_defect: "missing < on tool_call open",
            qwen_fragment: format!(
                "tool_call><function=inspect_verse><parameter=name>Romeo</parameter></function>{TOOL_CALL_END}"
            ),
        },
    ]
}

fn closed_case(
    marker_defect: &'static str,
    qwen_function_body: &str,
    expectation: ClosedEnvelopeExpectation,
) -> MarkerDefectCase {
    MarkerDefectCase {
        marker_defect,
        qwen_envelope: format!("{TOOL_CALL_START}{qwen_function_body}{TOOL_CALL_END}"),
        expectation,
    }
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

fn assert_closed_envelope_outcome(
    marker_defect: &str,
    qwen_envelope: &str,
    parse_outcome: Result<Vec<Qwen3_5OutputEvent>, Qwen3_5OutputParserError>,
    expectation: ClosedEnvelopeExpectation,
) {
    let failure_context = format!("defect={marker_defect}; envelope={qwen_envelope}");
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
                vec![Qwen3_5OutputEvent::ToolCall(Qwen3_5ToolCall {
                    index: 0,
                    function_name: function_name.to_owned(),
                    arguments_json: arguments_json.to_owned(),
                })],
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
                    .any(|output_event| matches!(output_event, Qwen3_5OutputEvent::ToolCall(_))),
                "{failure_context} emitted a tool call: {output_events:?}"
            );
        }
    }
}
