//! Fail-open coverage for non-canonical tool-call dialects and stream-end flushes.
//!
//! Quantized models sometimes emit Claude-style invoke/parameter attribute tags,
//! bare Qwen function tags without the envelope, or end the stream while a
//! control marker prefix is still buffered. Every one of those must reach the
//! harness as a tool call or visible text; generation must never abort.

use astronomical_model_serving::{Qwen3_5OutputEvent, Qwen3_5OutputParser, Qwen3_5ToolCall};

use super::support::{
    BALCONY_ARGUMENTS_JSON, DECLARED_CHARACTER_FUNCTION, DECLARED_SCENE_FUNCTION,
    ROMEO_ARGUMENTS_JSON, UNDECLARED_FUNCTION_NAME, literary_declared_tools,
    literary_output_parser,
};
use super::{TC_CLOSE, TC_OPEN};

// Marker literals are assembled so this suite never spells a live invoke block.
const INVOKE_OPEN: &str = concat!("<", "invoke name=\"");
const PARAM_ATTR_OPEN: &str = concat!("<", "parameter name=\"");
const PARAM_CLOSE: &str = concat!("<", "/parameter", ">");
const FUNC_OPEN: &str = concat!("<", "function=");
const FUNC_CLOSE: &str = concat!("<", "/function", ">");
const INVOKE_CLOSE: &str = concat!("<", "/invoke", ">");

fn invoke_style_envelope(
    function_name: &str,
    parameter_name: &str,
    parameter_value: &str,
) -> String {
    format!(
        "{INVOKE_OPEN}{function_name}\">\n{PARAM_ATTR_OPEN}{parameter_name}\">\n{parameter_value}\n{PARAM_CLOSE}\n{FUNC_CLOSE}\n{TC_CLOSE}"
    )
}

fn bare_qwen_function(
    function_name: &str,
    parameter_name: &str,
    parameter_value: &str,
    trailing_text: &str,
) -> String {
    format!(
        "{FUNC_OPEN}{function_name}>\n{PARAM_ATTR_OPEN}{parameter_name}\">\n{parameter_value}\n{PARAM_CLOSE}\n{FUNC_CLOSE}\n{trailing_text}"
    )
}

#[test]
fn should_parse_a_claude_style_invoke_envelope_as_a_tool_call() {
    let mut output_parser = literary_output_parser();
    let output_events = output_parser
        .push_fragment(&invoke_style_envelope(
            DECLARED_CHARACTER_FUNCTION,
            "name",
            "Romeo",
        ))
        .expect("a closed invoke-dialect envelope must not abort generation");
    assert_eq!(
        output_events,
        vec![Qwen3_5OutputEvent::ToolCall(Qwen3_5ToolCall {
            index: 0,
            function_name: DECLARED_CHARACTER_FUNCTION.to_owned(),
            arguments_json: ROMEO_ARGUMENTS_JSON.to_owned(),
        })]
    );
    assert!(
        output_parser
            .finish()
            .expect("finish after a parsed invoke envelope must stay fail-open")
            .is_empty()
    );
}

#[test]
fn should_parse_a_bare_qwen_function_and_resume_visible_text() {
    let mut output_parser = literary_output_parser();
    let output_events = output_parser
        .push_fragment(&bare_qwen_function(
            DECLARED_SCENE_FUNCTION,
            "scene",
            "balcony",
            "Juliet waits below.",
        ))
        .expect("a bare function without the envelope must not abort generation");
    assert_eq!(
        output_events,
        vec![
            Qwen3_5OutputEvent::ToolCall(Qwen3_5ToolCall {
                index: 0,
                function_name: DECLARED_SCENE_FUNCTION.to_owned(),
                arguments_json: BALCONY_ARGUMENTS_JSON.to_owned(),
            }),
            Qwen3_5OutputEvent::TextDelta("\nJuliet waits below.".to_owned()),
        ]
    );
}

#[test]
fn should_salvage_an_unclosed_invoke_envelope_when_generation_finishes() {
    let mut output_parser = literary_output_parser();
    let unclosed_invoke = format!(
        "{INVOKE_OPEN}{UNDECLARED_FUNCTION_NAME}\">\n{PARAM_ATTR_OPEN}name\">\nRomeo\n{PARAM_CLOSE}"
    );
    assert!(
        output_parser
            .push_fragment(&unclosed_invoke)
            .expect("an unclosed invoke envelope must stay pending")
            .is_empty()
    );
    let finish_events = output_parser
        .finish()
        .expect("an unclosed invoke envelope must salvage instead of aborting");
    assert_eq!(
        finish_events,
        vec![Qwen3_5OutputEvent::ToolCall(Qwen3_5ToolCall {
            index: 0,
            function_name: UNDECLARED_FUNCTION_NAME.to_owned(),
            arguments_json: ROMEO_ARGUMENTS_JSON.to_owned(),
        })]
    );
}

#[test]
fn should_flush_partial_marker_prefixes_as_text_when_generation_ends() {
    for partial_marker in ["<", "<t", "<inv", "<tool", "<func"] {
        let mut output_parser = literary_output_parser();
        let fragment = format!("Romeo seeks the friar. {partial_marker}");
        let push_events = output_parser
            .push_fragment(&fragment)
            .expect("a partial marker prefix must hold back instead of aborting");
        let finish_events = output_parser.finish().unwrap_or_else(|parser_error| {
            panic!("partial marker {partial_marker} must flush as text, not abort: {parser_error}")
        });
        let streamed_text = push_events
            .into_iter()
            .chain(finish_events)
            .map(|output_event| match output_event {
                Qwen3_5OutputEvent::TextDelta(text) => text,
                other_event => panic!(
                    "partial marker {partial_marker} emitted a non-text event: {other_event:?}"
                ),
            })
            .collect::<String>();
        assert_eq!(
            streamed_text, fragment,
            "partial marker {partial_marker} must flush verbatim instead of aborting"
        );
    }
}

#[test]
fn should_flush_an_oversized_text_frame_instead_of_aborting_generation() {
    let mut output_parser = literary_output_parser();
    let oversized_verse = format!("{}<", "Romeo ".repeat(4 * 1024));
    assert!(oversized_verse.len() > 16 * 1024);
    let output_events = output_parser
        .push_fragment(&oversized_verse)
        .expect("an oversized text frame must flush instead of aborting");
    assert_eq!(
        output_events,
        vec![Qwen3_5OutputEvent::TextDelta(oversized_verse)]
    );
    assert!(
        output_parser
            .finish()
            .expect("finish after an oversized flush must stay open")
            .is_empty()
    );
}

#[test]
fn should_keep_streaming_text_after_the_pending_cap_is_crossed() {
    let mut output_parser = literary_output_parser();
    let verse_block = "Juliet ".repeat(8 * 1024);
    let mut streamed_bytes = 0usize;
    for _ in 0..20 {
        let output_events = output_parser
            .push_fragment(&verse_block)
            .expect("crossing the pending cap must flush instead of aborting");
        streamed_bytes += output_events
            .iter()
            .map(|output_event| match output_event {
                Qwen3_5OutputEvent::TextDelta(text) => text.len(),
                other_event => panic!("unexpected event while flushing: {other_event:?}"),
            })
            .sum::<usize>();
    }
    let finish_events = output_parser
        .finish()
        .expect("finish after crossing the pending cap must stay open");
    streamed_bytes += finish_events
        .iter()
        .map(|output_event| match output_event {
            Qwen3_5OutputEvent::TextDelta(text) => text.len(),
            other_event => panic!("unexpected finish event: {other_event:?}"),
        })
        .sum::<usize>();
    assert_eq!(
        streamed_bytes,
        verse_block.len() * 20,
        "no generated text may be dropped while flushing"
    );
}

#[test]
fn should_forward_a_nameless_invoke_body_as_visible_text_not_an_abort() {
    let mut output_parser = literary_output_parser();
    let nameless_invoke = format!(
        "{TC_OPEN}{INVOKE_OPEN}\">\n{PARAM_ATTR_OPEN}name\">\nRomeo\n{PARAM_CLOSE}\n{FUNC_CLOSE}\n{TC_CLOSE}"
    );
    let output_events = output_parser
        .push_fragment(&nameless_invoke)
        .expect("a nameless invoke body must not abort generation");
    assert!(
        !output_events
            .iter()
            .any(|output_event| matches!(output_event, Qwen3_5OutputEvent::ToolCall(_))),
        "a nameless invoke body must not invent a tool call: {output_events:?}"
    );
    assert!(
        output_parser
            .finish()
            .expect("finish after a nameless invoke must stay open")
            .is_empty()
    );
}

fn invoke_closed_with_invoke_end(
    function_name: &str,
    parameter_name: &str,
    parameter_value: &str,
) -> String {
    format!(
        "{INVOKE_OPEN}{function_name}\">\n{PARAM_ATTR_OPEN}{parameter_name}\">\n{parameter_value}\n{PARAM_CLOSE}\n{INVOKE_CLOSE}"
    )
}

#[test]
fn should_parse_an_invoke_block_that_closes_with_the_invoke_end_tag() {
    let mut output_parser = literary_output_parser();
    let output_events = output_parser
        .push_fragment(&invoke_closed_with_invoke_end(
            DECLARED_CHARACTER_FUNCTION,
            "name",
            "Romeo",
        ))
        .expect("an invoke-end-closed call must not abort generation");
    assert_eq!(
        output_events,
        vec![Qwen3_5OutputEvent::ToolCall(Qwen3_5ToolCall {
            index: 0,
            function_name: DECLARED_CHARACTER_FUNCTION.to_owned(),
            arguments_json: ROMEO_ARGUMENTS_JSON.to_owned(),
        })]
    );
}

#[test]
fn should_promote_an_invoke_inside_prompt_opened_reasoning_to_a_tool_call() {
    let mut output_parser =
        Qwen3_5OutputParser::new_after_thinking_prefix(&literary_declared_tools())
            .expect("literary tools should construct a prompt-opened reasoning parser");
    let fragment = format!(
        "Romeo waits on the balcony.\n{}",
        invoke_closed_with_invoke_end(DECLARED_CHARACTER_FUNCTION, "name", "Romeo")
    );
    let output_events = output_parser
        .push_fragment(&fragment)
        .expect("a tool attempt inside the open think channel must reach the harness");
    assert_eq!(
        output_events,
        vec![
            Qwen3_5OutputEvent::ReasoningDelta("Romeo waits on the balcony.\n".to_owned()),
            Qwen3_5OutputEvent::ToolCall(Qwen3_5ToolCall {
                index: 0,
                function_name: DECLARED_CHARACTER_FUNCTION.to_owned(),
                arguments_json: ROMEO_ARGUMENTS_JSON.to_owned(),
            }),
        ]
    );
}
