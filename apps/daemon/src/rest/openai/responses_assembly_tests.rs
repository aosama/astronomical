//! Unit tests for `crate::rest::openai::responses_assembly`: how stream
//! events collect into one non-streaming Responses payload.

use crate::{ChatGenerationStreamEvent, OpenAiResponsesCollector};
use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatGenerationFailureReason, OUT_OF_GRAPHICS_MEMORY_ADVICE,
};

#[test]
fn should_collect_reasoning_summary_text_and_function_calls_into_one_response() {
    let mut collector = OpenAiResponsesCollector::new(
        "resp_instance-9".to_owned(),
        1_753_000_000,
        "astronomical/fake-mixture-of-experts".to_owned(),
        Some("Be precise.".to_owned()),
        Default::default(),
    );
    collector
        .ingest_event(ChatGenerationStreamEvent::ReasoningFragment(
            "Inspect first.".to_owned(),
        ))
        .expect("reasoning should be collectable");
    collector
        .ingest_event(ChatGenerationStreamEvent::PrefillProgress {
            processed_tokens: 2_048,
            total_tokens: 4_096,
            elapsed_millis: 1_000,
            forward_prefill_chunk_elapsed_millis: Some(900),
            completed_prefill_chunk_tokens: Some(2_048),
            mlx_active_memory_bytes: Some(20_000),
            mlx_allocator_cache_memory_bytes: Some(0),
            mlx_peak_memory_bytes: Some(22_000),
        })
        .expect("internal prefill telemetry should be ignored");
    collector
        .ingest_event(ChatGenerationStreamEvent::TextFragment("Done.".to_owned()))
        .expect("text should be collectable");
    collector
        .ingest_event(ChatGenerationStreamEvent::ToolCall {
            tool_call_index: 0,
            function_name: "read".to_owned(),
            arguments_json: r#"{"filePath":"README.md"}"#.to_owned(),
        })
        .expect("function calls should be collectable");

    let response = collector
        .into_response(
            100,
            20,
            64,
            7,
            ChatGenerationCompletionReason::EndOfSequence,
        )
        .expect("the completed response should assemble");
    let response_document =
        serde_json::to_value(response).expect("the assembled response should serialize");

    assert_eq!(response_document["output"][0]["type"], "reasoning");
    assert_eq!(
        response_document["output"][0]["summary"][0],
        serde_json::json!({"type":"summary_text","text":"Inspect first."})
    );
    assert_eq!(
        response_document["output"][0]["content"],
        serde_json::json!([])
    );
    assert!(response_document["output"][0]["encrypted_content"].is_null());
    assert_eq!(response_document["output"][1]["type"], "message");
    assert_eq!(response_document["output"][2]["type"], "function_call");
    assert_eq!(response_document["output_text"], "Done.");
    assert!(
        response_document["completed_at"]
            .as_u64()
            .expect("completed_at should be a Unix timestamp")
            > response_document["created_at"]
                .as_u64()
                .expect("created_at should be a Unix timestamp")
    );
    assert_eq!(
        response_document["usage"]["output_tokens_details"]["reasoning_tokens"],
        7
    );
    assert_eq!(response_document["usage"]["total_tokens"], 120);
}

#[test]
fn should_mark_a_maximum_output_token_response_incomplete() {
    let mut collector = OpenAiResponsesCollector::new(
        "resp_instance-10".to_owned(),
        1_753_000_000,
        "astronomical/fake-mixture-of-experts".to_owned(),
        None,
        Default::default(),
    );
    collector
        .ingest_event(ChatGenerationStreamEvent::TextFragment(
            "Partial answer".to_owned(),
        ))
        .expect("partial text should be collectable");

    let response = collector
        .into_response(
            100,
            20,
            0,
            0,
            ChatGenerationCompletionReason::MaximumOutputTokens,
        )
        .expect("the token limit should produce an incomplete response");
    let response_document =
        serde_json::to_value(response).expect("the incomplete response should serialize");

    assert_eq!(response_document["status"], "incomplete");
    assert_eq!(
        response_document["incomplete_details"]["reason"],
        "max_output_tokens"
    );
    assert_eq!(response_document["output"][0]["status"], "incomplete");
}

fn failed_response_error(failure_reason: ChatGenerationFailureReason) -> serde_json::Value {
    let collector = OpenAiResponsesCollector::new(
        "resp_instance-11".to_owned(),
        1_753_000_000,
        "astronomical/fake-mixture-of-experts".to_owned(),
        None,
        Default::default(),
    );
    let failed_response = collector.into_failed_response(failure_reason);
    let mut response_document =
        serde_json::to_value(failed_response).expect("a failed response should serialize");
    assert_eq!(response_document["status"], "failed");
    response_document["error"].take()
}

/// A client can only tell someone to free memory if this failure has a code of
/// its own; under the generic code it reads as an engine that broke.
#[test]
fn should_report_out_of_graphics_memory_with_its_own_code_and_advice() {
    let out_of_memory_error =
        failed_response_error(ChatGenerationFailureReason::OutOfGraphicsMemory);

    assert_eq!(out_of_memory_error["code"], "chat_out_of_memory");
    assert_eq!(
        out_of_memory_error["message"],
        OUT_OF_GRAPHICS_MEMORY_ADVICE
    );
}

#[test]
fn should_keep_the_generic_code_for_a_failure_that_is_not_out_of_memory() {
    for other_failure_reason in [
        ChatGenerationFailureReason::EngineBusy,
        ChatGenerationFailureReason::FatalExecution {
            reason: "model execution failed inside the local worker".to_owned(),
        },
    ] {
        let other_error = failed_response_error(other_failure_reason);

        assert_eq!(other_error["code"], "response_generation_failed");
        assert_ne!(other_error["message"], OUT_OF_GRAPHICS_MEMORY_ADVICE);
    }
}
