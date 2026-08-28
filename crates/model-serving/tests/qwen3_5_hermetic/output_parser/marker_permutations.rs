//! Qwen3.5 marker-defect permutations for dense and MoE.
//!
//! Closed envelopes with a usable function name must reach the harness. Salvage
//! a missing `<` on `<function=` / `<parameter=`. Unclosed envelopes stay pending
//! while tokens can still arrive, then salvage at generation end. Empty names
//! stream as text instead of aborting generation.

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
    expectation: ClosedEnvelopeExpectation,
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
fn should_forward_unclosed_qwen_envelopes_when_generation_finishes() {
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
        let finish_events = output_parser.finish().unwrap_or_else(|parser_error| {
            panic!(
                "unclosed defect {} must not abort generation: {parser_error}; fragment={}",
                unclosed_envelope_case.marker_defect, unclosed_envelope_case.qwen_fragment
            )
        });
        assert_unclosed_finish_outcome(
            unclosed_envelope_case.marker_defect,
            &unclosed_envelope_case.qwen_fragment,
            finish_events,
            unclosed_envelope_case.expectation,
        );
    }
}

#[test]
fn should_recover_tool_calls_that_omit_the_envelope_open() {
    // The model sometimes writes Qwen function tags without the surrounding
    // envelope. The call must still reach the harness, and any slop before the
    // function open streams as visible text.
    let mut declared_open = literary_output_parser();
    let declared_events = declared_open
        .push_fragment(&format!(
            "<function={DECLARED_CHARACTER_FUNCTION}><parameter=name>Romeo</parameter></function>{TOOL_CALL_END}"
        ))
        .expect("a bare declared function must not abort generation");
    assert_eq!(
        declared_events,
        vec![Qwen3_5OutputEvent::ToolCall(Qwen3_5ToolCall {
            index: 0,
            function_name: DECLARED_CHARACTER_FUNCTION.to_owned(),
            arguments_json: ROMEO_ARGUMENTS_JSON.to_owned(),
        })]
    );

    let mut undeclared_open = literary_output_parser();
    let undeclared_events = undeclared_open
        .push_fragment(&format!(
            "tool_call><function={UNDECLARED_FUNCTION_NAME}><parameter=name>Romeo</parameter></function>{TOOL_CALL_END}"
        ))
        .expect("slop before a bare function must stream as text");
    assert_eq!(
        undeclared_events,
        vec![
            Qwen3_5OutputEvent::TextDelta("tool_call>".to_owned()),
            Qwen3_5OutputEvent::ToolCall(Qwen3_5ToolCall {
                index: 0,
                function_name: UNDECLARED_FUNCTION_NAME.to_owned(),
                arguments_json: ROMEO_ARGUMENTS_JSON.to_owned(),
            }),
        ]
    );
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
            expectation: tool_call(DECLARED_CHARACTER_FUNCTION, ROMEO_ARGUMENTS_JSON),
        },
        UnclosedEnvelopeCase {
            marker_defect: "missing < on tool_call close",
            qwen_fragment: format!(
                "{TOOL_CALL_START}<function=find_character><parameter=name>Romeo</parameter></function>/tool_call>"
            ),
            expectation: tool_call(DECLARED_CHARACTER_FUNCTION, ROMEO_ARGUMENTS_JSON),
        },
        UnclosedEnvelopeCase {
            marker_defect: "truncated tool_call close",
            qwen_fragment: format!(
                "{TOOL_CALL_START}<function=inspect_verse><parameter=name>Romeo</parameter></function></tool_c"
            ),
            expectation: tool_call(UNDECLARED_FUNCTION_NAME, ROMEO_ARGUMENTS_JSON),
        },
        UnclosedEnvelopeCase {
            marker_defect: "unclosed undeclared call",
            qwen_fragment: format!("{TOOL_CALL_START}<function=inspect_verse>"),
            expectation: tool_call(UNDECLARED_FUNCTION_NAME, EMPTY_ARGUMENTS_JSON),
        },
        UnclosedEnvelopeCase {
            marker_defect: "nameless unclosed arguments",
            qwen_fragment: format!("{TOOL_CALL_START}<parameter=name>Romeo</parameter>"),
            expectation: ClosedEnvelopeExpectation::VisibleText,
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

#[test]
fn should_salvage_an_oversized_qwen_tool_call_fragment_without_aborting_generation() {
    let mut output_parser = literary_output_parser();
    assert!(
        output_parser
            .push_fragment(&format!(
                "{TOOL_CALL_START}<function=find_character><parameter=name>Romeo"
            ))
            .expect("an open Qwen tool call should remain pending")
            .is_empty()
    );
    let oversized_fragment = "R".repeat(20 * 1024);
    let salvaged_events = output_parser
        .push_fragment(&oversized_fragment)
        .expect("an oversized fragment in tool-call state must salvage instead of aborting");
    assert_eq!(
        salvaged_events,
        vec![Qwen3_5OutputEvent::ToolCall(Qwen3_5ToolCall {
            index: 0,
            function_name: DECLARED_CHARACTER_FUNCTION.to_owned(),
            arguments_json: ROMEO_ARGUMENTS_JSON.to_owned(),
        })]
    );
    assert!(
        output_parser
            .finish()
            .expect("salvage should leave the parser able to complete the request")
            .is_empty()
    );
}

fn assert_unclosed_finish_outcome(
    marker_defect: &str,
    qwen_fragment: &str,
    finish_events: Vec<Qwen3_5OutputEvent>,
    expectation: ClosedEnvelopeExpectation,
) {
    match expectation {
        ClosedEnvelopeExpectation::VisibleText => {
            assert!(
                !finish_events
                    .iter()
                    .any(|output_event| matches!(output_event, Qwen3_5OutputEvent::ToolCall(_))),
                "defect={marker_defect}; fragment={qwen_fragment} emitted a tool call: {finish_events:?}"
            );
            assert!(
                finish_events.iter().any(|output_event| {
                    matches!(output_event, Qwen3_5OutputEvent::TextDelta(text) if !text.is_empty())
                }),
                "defect={marker_defect}; fragment={qwen_fragment} dropped nameless tool-call text: {finish_events:?}"
            );
        }
        tool_call_expectation => {
            assert_closed_envelope_outcome(
                marker_defect,
                qwen_fragment,
                Ok(finish_events),
                tool_call_expectation,
            );
        }
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
