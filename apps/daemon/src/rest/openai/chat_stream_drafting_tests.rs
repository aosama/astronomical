//! A streamed reply's drafting figures in its final chunk's usage.

use ipc_protocol::{ChatGenerationCompletionReason, RequestDraftingStatistics};

use super::OpenAiChatStreamEncoder;
use crate::ChatGenerationStreamEvent;

fn final_chunk_text(stream_events: Vec<ChatGenerationStreamEvent>) -> String {
    let stream_encoder = OpenAiChatStreamEncoder::new(
        1,
        "chatcmpl-test".to_owned(),
        1,
        "test-model".to_owned(),
        true,
        false,
    );
    for stream_event in stream_events {
        let encoded_events = stream_encoder
            .encode(stream_event)
            .expect("the event encodes");
        assert!(encoded_events.is_empty(), "a report alone sends no chunk");
    }
    let completion_events = stream_encoder
        .encode(ChatGenerationStreamEvent::Completed {
            prompt_token_count: 28,
            generated_token_count: 99,
            reasoning_token_count: 0,
            astro_ssd_prompt_cache_restored_token_count: 0,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        })
        .expect("the completion encodes");
    format!("{completion_events:?}")
}

#[test]
fn should_put_a_drafting_report_into_the_final_chunks_usage() {
    let final_chunk = final_chunk_text(vec![ChatGenerationStreamEvent::DraftingReport(
        RequestDraftingStatistics {
            round_count: 26,
            drafted_token_count: 78,
            accepted_token_count: 74,
        },
    )]);
    assert!(
        final_chunk.contains("accepted_prediction_tokens"),
        "{final_chunk}"
    );
}

#[test]
fn should_leave_the_final_chunks_usage_unchanged_without_a_drafting_report() {
    let final_chunk = final_chunk_text(Vec::new());
    assert!(final_chunk.contains("completion_tokens"), "{final_chunk}");
    assert!(
        !final_chunk.contains("completion_tokens_details"),
        "{final_chunk}"
    );
}
