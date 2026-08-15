use astronomical_ipc_protocol::{ChatGenerationCompletionReason, ChatGenerationFailureReason};
use astronomical_supervisor::{ChatGenerationStreamEvent, OpenAiResponsesStreamEncoder};

#[test]
fn should_emit_the_exact_fragmented_text_lifecycle_and_skip_prefill_telemetry() {
    let mut encoder = OpenAiResponsesStreamEncoder::new(
        "resp_instance-11".to_owned(),
        1_753_000_000,
        "mlx-community/Ornith-1.0-35B-OptiQ-4bit".to_owned(),
        None,
        Default::default(),
    );
    let mut encoded_events = encoder.initial_events();
    encoded_events.extend(
        encoder
            .encode(ChatGenerationStreamEvent::PrefillProgress {
                processed_tokens: 2_048,
                total_tokens: 4_096,
                elapsed_millis: 1_000,
                forward_prefill_chunk_elapsed_millis: Some(900),
                completed_prefill_chunk_tokens: Some(2_048),
                mlx_active_memory_bytes: Some(20_000),
                mlx_allocator_cache_memory_bytes: Some(0),
                mlx_peak_memory_bytes: Some(22_000),
            })
            .expect("prefill telemetry encoding should succeed"),
    );
    encoded_events.extend(
        encoder
            .encode(ChatGenerationStreamEvent::TextFragment("Hel".to_owned()))
            .expect("the first text fragment should encode"),
    );
    encoded_events.extend(
        encoder
            .encode(ChatGenerationStreamEvent::TextFragment("lo".to_owned()))
            .expect("the second text fragment should encode"),
    );
    encoded_events.extend(
        encoder
            .encode(ChatGenerationStreamEvent::Completed {
                prompt_token_count: 10,
                generated_token_count: 2,
                reasoning_token_count: 0,
                cached_token_count: 0,
                reason: ChatGenerationCompletionReason::EndOfSequence,
            })
            .expect("completion should encode"),
    );

    let event_types = encoded_events
        .iter()
        .map(|encoded_event| encoded_event.event_type())
        .collect::<Vec<_>>();
    assert_eq!(
        event_types,
        vec![
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    let sequence_numbers = encoded_events
        .iter()
        .map(|encoded_event| encoded_event.sequence_number())
        .collect::<Vec<_>>();
    assert_eq!(sequence_numbers, (0_u64..10).collect::<Vec<_>>());
    let serialized_events = encoded_events
        .iter()
        .map(|encoded_event| serde_json::to_string(encoded_event).expect("event should serialize"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(serialized_events.contains(r#""delta":"Hel""#));
    assert!(serialized_events.contains(r#""delta":"lo""#));
    assert!(!serialized_events.contains("[DONE]"));
    assert_no_response_id_fields(&encoded_events);
}

#[test]
fn should_stream_raw_reasoning_as_summary_text_without_encrypted_content() {
    let mut encoder = OpenAiResponsesStreamEncoder::new(
        "resp_instance-12".to_owned(),
        1_753_000_000,
        "mlx-community/Ornith-1.0-35B-OptiQ-4bit".to_owned(),
        None,
        Default::default(),
    );
    let mut encoded_events = encoder.initial_events();
    encoded_events.extend(
        encoder
            .encode(ChatGenerationStreamEvent::ReasoningFragment(
                "Inspect first.".to_owned(),
            ))
            .expect("reasoning should encode"),
    );
    encoded_events.extend(
        encoder
            .encode(ChatGenerationStreamEvent::ToolCall {
                tool_call_index: 0,
                function_name: "read".to_owned(),
                arguments_json: r#"{"filePath":"README.md"}"#.to_owned(),
            })
            .expect("the function call should encode"),
    );
    encoded_events.extend(
        encoder
            .encode(ChatGenerationStreamEvent::Completed {
                prompt_token_count: 10,
                generated_token_count: 2,
                reasoning_token_count: 2,
                cached_token_count: 0,
                reason: ChatGenerationCompletionReason::ToolCalls,
            })
            .expect("tool completion should encode"),
    );

    let event_types = encoded_events
        .iter()
        .map(|encoded_event| encoded_event.event_type())
        .collect::<Vec<_>>();
    assert_eq!(
        event_types,
        vec![
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    let serialized_events = encoded_events
        .iter()
        .map(|encoded_event| serde_json::to_value(encoded_event).expect("event should serialize"))
        .collect::<Vec<_>>();
    assert_eq!(serialized_events[3]["summary_index"], 0);
    assert_eq!(serialized_events[3]["delta"], "Inspect first.");
    assert_eq!(serialized_events[4]["summary_index"], 0);
    assert_eq!(serialized_events[4]["text"], "Inspect first.");
    assert_eq!(
        serialized_events[5]["item"]["summary"][0]["type"],
        "summary_text"
    );
    assert_eq!(
        serialized_events[5]["item"]["summary"][0]["text"],
        "Inspect first."
    );
    assert!(
        !serde_json::to_string(&serialized_events)
            .expect("the event documents should serialize together")
            .contains("encrypted_content")
    );
    assert_eq!(serialized_events[7]["delta"], r#"{"filePath":"README.md"}"#);
    assert_eq!(serialized_events[8]["name"], "read");
    assert_eq!(serialized_events[10]["response"]["output_text"], "");
    assert_no_response_id_fields(&encoded_events);
}

#[test]
fn should_close_open_text_before_emitting_an_incomplete_terminal_event() {
    let mut encoder = OpenAiResponsesStreamEncoder::new(
        "resp_instance-13".to_owned(),
        1_753_000_000,
        "mlx-community/Ornith-1.0-35B-OptiQ-4bit".to_owned(),
        None,
        Default::default(),
    );
    let mut encoded_events = encoder.initial_events();
    encoded_events.extend(
        encoder
            .encode(ChatGenerationStreamEvent::TextFragment(
                "Partial".to_owned(),
            ))
            .expect("partial text should encode"),
    );
    encoded_events.extend(
        encoder
            .encode(ChatGenerationStreamEvent::Completed {
                prompt_token_count: 10,
                generated_token_count: 2,
                reasoning_token_count: 0,
                cached_token_count: 0,
                reason: ChatGenerationCompletionReason::MaximumOutputTokens,
            })
            .expect("the token ceiling should encode"),
    );

    let serialized_terminal_event = serde_json::to_value(
        encoded_events
            .back()
            .expect("the stream should contain one terminal event"),
    )
    .expect("the terminal event should serialize");
    assert_eq!(serialized_terminal_event["type"], "response.incomplete");
    assert_eq!(
        serialized_terminal_event["response"]["status"],
        "incomplete"
    );
    assert_eq!(
        serialized_terminal_event["response"]["incomplete_details"]["reason"],
        "max_output_tokens"
    );
    assert!(encoder.is_terminal());
}

#[test]
fn should_emit_a_terminal_failed_response_for_a_worker_reported_context_failure() {
    let mut encoder = OpenAiResponsesStreamEncoder::new(
        "resp_instance-14".to_owned(),
        1_753_000_000,
        "mlx-community/Ornith-1.0-35B-OptiQ-4bit".to_owned(),
        None,
        Default::default(),
    );
    let mut encoded_events = encoder.initial_events();
    encoded_events.extend(
        encoder
            .encode(ChatGenerationStreamEvent::Failed {
                reason: ChatGenerationFailureReason::ContextLengthExceeded {
                    actual_total_context_tokens: 262_145,
                    maximum_context_tokens: 262_144,
                },
            })
            .expect("the context failure should encode"),
    );

    let terminal_event = serde_json::to_value(
        encoded_events
            .back()
            .expect("the stream should contain one terminal event"),
    )
    .expect("the terminal event should serialize");
    assert_eq!(terminal_event["type"], "response.failed");
    assert_eq!(terminal_event["response"]["status"], "failed");
    assert_eq!(
        terminal_event["response"]["error"]["code"],
        "context_length_exceeded"
    );
    assert!(encoder.is_terminal());
}

#[test]
fn should_mark_partially_emitted_output_incomplete_when_the_worker_fails() {
    let mut encoder = OpenAiResponsesStreamEncoder::new(
        "resp_instance-15".to_owned(),
        1_753_000_000,
        "mlx-community/Ornith-1.0-35B-OptiQ-4bit".to_owned(),
        None,
        Default::default(),
    );
    let mut encoded_events = encoder.initial_events();
    encoded_events.extend(
        encoder
            .encode(ChatGenerationStreamEvent::TextFragment(
                "Partial".to_owned(),
            ))
            .expect("the partial output should encode"),
    );
    encoded_events.extend(
        encoder
            .encode(ChatGenerationStreamEvent::Failed {
                reason: ChatGenerationFailureReason::InvalidRequest {
                    reason: "the tool result was invalid".to_owned(),
                },
            })
            .expect("the request failure should encode"),
    );

    let terminal_event = serde_json::to_value(
        encoded_events
            .back()
            .expect("the stream should contain one terminal event"),
    )
    .expect("the terminal event should serialize");
    assert_eq!(terminal_event["type"], "response.failed");
    assert_eq!(
        terminal_event["response"]["output"][0]["status"],
        "incomplete"
    );
    assert_eq!(
        terminal_event["response"]["error"]["code"],
        "response_generation_failed"
    );
}

fn assert_no_response_id_fields(
    encoded_events: &std::collections::VecDeque<
        astronomical_rest_contract::OpenAiResponseStreamEvent,
    >,
) {
    for encoded_event in encoded_events {
        let event_document = serde_json::to_value(encoded_event).expect("event should serialize");
        assert!(
            event_document.get("response_id").is_none(),
            "{} must not include response_id",
            encoded_event.event_type()
        );
    }
}
