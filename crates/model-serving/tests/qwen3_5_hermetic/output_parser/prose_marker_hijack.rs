//! Tool-call markers quoted inside model prose must not hijack the parser.
//!
//! A quantized coding model that discusses tool-calling infrastructure quotes
//! markers like `<invoke name=...>` inside its reasoning. The parser must treat
//! the quoted opener as prose, stream the reasoning, and still deliver the
//! model's real tool call that follows the think close.

use super::support::{DECLARED_CHARACTER_FUNCTION, ROMEO_ARGUMENTS_JSON, literary_declared_tools};
use super::{THINK_END, TOOL_CALL_END, TOOL_CALL_START};
use astronomical_model_serving::{Qwen3_5OutputEvent, Qwen3_5OutputParser};

#[test]
fn should_resync_when_quoted_prose_markers_hijack_the_tool_call_state() {
    let mut parser = Qwen3_5OutputParser::new_after_thinking_prefix(&literary_declared_tools())
        .expect("Romeo and Juliet literary tools should construct a Qwen3.5 parser");

    let reasoning_prose = "The diffstat shows the consolidation. The normalizer rewrites \
markers like `<invoke name=...>` into the Qwen grammar before parsing. Let me verify the wiring.";
    let mut events = parser
        .push_fragment(reasoning_prose)
        .expect("reasoning prose must not abort");

    let remainder = format!(
        "{THINK_END}\nNow let me isolate the shipped changes.\n\
{TOOL_CALL_START}\n<function=find_character>\n<parameter=name>Romeo</parameter>\n{TOOL_CALL_END}\n"
    );
    events.extend(parser.push_fragment(&remainder).expect("remainder"));
    events.extend(parser.finish().expect("finish"));

    let tool_calls: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            Qwen3_5OutputEvent::ToolCall(tool_call) => Some(tool_call),
            _ => None,
        })
        .collect();
    assert_eq!(
        tool_calls.len(),
        1,
        "exactly the real tool call must reach the harness as a structured call; events: {events:?}"
    );
    assert_eq!(tool_calls[0].function_name, DECLARED_CHARACTER_FUNCTION);
    assert_eq!(tool_calls[0].arguments_json, ROMEO_ARGUMENTS_JSON);

    for event in &events {
        if let Qwen3_5OutputEvent::TextDelta(text) = event {
            assert!(
                !text.contains(TOOL_CALL_START),
                "tool-call marker leaked as visible text: {text:?}"
            );
            assert!(
                !text.contains(THINK_END),
                "think-close marker leaked as visible text: {text:?}"
            );
        }
    }
    assert!(
        events.iter().any(|event| matches!(
            event,
            Qwen3_5OutputEvent::ReasoningDelta(text) if text.contains("<invoke name=...>")
        )),
        "the hijacked reasoning prose must still stream as reasoning; events: {events:?}"
    );
}
