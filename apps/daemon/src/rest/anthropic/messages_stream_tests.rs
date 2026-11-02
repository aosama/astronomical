use astronomical_ipc_protocol::{ChatGenerationCompletionReason, ChatGenerationFailureReason};
use astronomical_rest_contract::{
    AnthropicContentBlockDelta, AnthropicMessageStreamEvent, AnthropicStopReason,
};

use super::messages_stream::AnthropicMessagesStreamEncoder;
use crate::ChatGenerationStreamEvent;

fn completed_event(reason: ChatGenerationCompletionReason) -> ChatGenerationStreamEvent {
    ChatGenerationStreamEvent::Completed {
        prompt_token_count: 120,
        generated_token_count: 9,
        reasoning_token_count: 3,
        cached_token_count: 100,
        reason,
    }
}

fn encoded_stream(
    worker_events: Vec<ChatGenerationStreamEvent>,
) -> (Vec<AnthropicMessageStreamEvent>, bool) {
    let mut stream_encoder =
        AnthropicMessagesStreamEncoder::new("msg_test-1".to_owned(), "local-model".to_owned());
    let mut stream_events = Vec::from(stream_encoder.initial_events());
    for worker_event in worker_events {
        stream_events.extend(stream_encoder.encode(worker_event));
    }
    (stream_events, stream_encoder.is_terminal())
}

fn event_types(stream_events: &[AnthropicMessageStreamEvent]) -> Vec<&'static str> {
    let mut event_type_names = Vec::with_capacity(stream_events.len());
    for stream_event in stream_events {
        event_type_names.push(stream_event.event_type());
    }
    event_type_names
}

#[test]
fn should_open_one_block_per_kind_and_close_it_before_the_next() {
    let (stream_events, is_terminal) = encoded_stream(vec![
        ChatGenerationStreamEvent::ReasoningFragment("Let me ".to_owned()),
        ChatGenerationStreamEvent::ReasoningFragment("think.".to_owned()),
        ChatGenerationStreamEvent::TextFragment("To be".to_owned()),
        ChatGenerationStreamEvent::TextFragment(", or not.".to_owned()),
        completed_event(ChatGenerationCompletionReason::EndOfSequence),
    ]);

    assert_eq!(
        event_types(&stream_events),
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    assert!(is_terminal);
    assert!(matches!(
        &stream_events[6],
        AnthropicMessageStreamEvent::ContentBlockDelta {
            index: 1,
            delta: AnthropicContentBlockDelta::TextDelta { text },
        } if text == "To be"
    ));
}

#[test]
fn should_report_cached_input_separately_in_the_final_usage() {
    let (stream_events, _) = encoded_stream(vec![completed_event(
        ChatGenerationCompletionReason::MaximumOutputTokens,
    )]);

    let AnthropicMessageStreamEvent::MessageDelta { delta, usage } = &stream_events[1] else {
        panic!("an empty completion should go straight to message_delta");
    };
    assert_eq!(delta.stop_reason, AnthropicStopReason::MaxTokens);
    assert_eq!(usage.input_tokens, 20);
    assert_eq!(usage.cache_read_input_tokens, 100);
    assert_eq!(usage.output_tokens, 9);
}

#[test]
fn should_emit_each_tool_call_as_a_complete_tool_use_block() {
    let (stream_events, _) = encoded_stream(vec![
        ChatGenerationStreamEvent::TextFragment("Reading.".to_owned()),
        ChatGenerationStreamEvent::ToolCall {
            tool_call_index: 0,
            function_name: "read_play".to_owned(),
            arguments_json: r#"{"act":3}"#.to_owned(),
        },
        completed_event(ChatGenerationCompletionReason::ToolCalls),
    ]);

    assert_eq!(
        event_types(&stream_events)[3..8],
        [
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta"
        ]
    );
    assert!(matches!(
        &stream_events[5],
        AnthropicMessageStreamEvent::ContentBlockDelta {
            index: 1,
            delta: AnthropicContentBlockDelta::InputJsonDelta { partial_json },
        } if partial_json == r#"{"act":3}"#
    ));
    assert!(matches!(
        &stream_events[7],
        AnthropicMessageStreamEvent::MessageDelta { delta, .. }
            if delta.stop_reason == AnthropicStopReason::ToolUse
    ));
}

#[test]
fn should_end_with_a_prompt_too_long_error_on_context_overflow() {
    let (stream_events, is_terminal) = encoded_stream(vec![ChatGenerationStreamEvent::Failed {
        reason: ChatGenerationFailureReason::ContextLengthExceeded {
            actual_total_context_tokens: 262_145,
            maximum_context_tokens: 262_144,
        },
    }]);

    assert!(is_terminal);
    assert!(matches!(
        &stream_events[1],
        AnthropicMessageStreamEvent::Error { error }
            if error.error_type == "invalid_request_error"
                && error.message == "prompt is too long: 262145 tokens > 262144 maximum"
    ));
}

#[test]
fn should_refuse_a_tool_call_whose_arguments_are_not_json() {
    let (stream_events, is_terminal) = encoded_stream(vec![ChatGenerationStreamEvent::ToolCall {
        tool_call_index: 0,
        function_name: "read_play".to_owned(),
        arguments_json: "{not json".to_owned(),
    }]);

    assert!(is_terminal);
    assert_eq!(event_types(&stream_events), vec!["message_start", "error"]);
}

#[test]
fn should_end_with_an_error_when_the_request_is_cancelled() {
    let (stream_events, is_terminal) = encoded_stream(vec![
        ChatGenerationStreamEvent::TextFragment("Partial".to_owned()),
        completed_event(ChatGenerationCompletionReason::Cancelled),
    ]);

    assert!(is_terminal);
    assert_eq!(event_types(&stream_events).last(), Some(&"error"));
}
