//! A reply's drafting figures in its OpenAI usage.

use ipc_protocol::RequestDraftingStatistics;
use rest_contract::OpenAiTokenUsage;

use super::{OpenAiChatCompletionCollector, with_drafted_tokens};

fn serialized_usage(drafting: Option<RequestDraftingStatistics>) -> String {
    let token_usage = OpenAiTokenUsage::new(28, 99).expect("small counts do not overflow");
    serde_json::to_string(&with_drafted_tokens(token_usage, drafting)).expect("usage serializes")
}

#[test]
fn should_report_kept_and_thrown_away_guesses_for_a_reply_that_drafted() {
    let usage_json = serialized_usage(Some(RequestDraftingStatistics {
        round_count: 26,
        drafted_token_count: 78,
        accepted_token_count: 74,
    }));
    assert!(
        usage_json.contains(
            r#""completion_tokens_details":{"accepted_prediction_tokens":74,"rejected_prediction_tokens":4}"#
        ),
        "{usage_json}"
    );
}

#[test]
fn should_leave_the_usage_unchanged_for_a_reply_that_did_not_draft() {
    assert!(!serialized_usage(None).contains("completion_tokens_details"));
}

fn collected_reply_json(stream_events: Vec<crate::ChatGenerationStreamEvent>) -> String {
    let mut chat_completion_collector = OpenAiChatCompletionCollector::new(
        "chatcmpl-test".to_owned(),
        1,
        "test-model".to_owned(),
        false,
    );
    for stream_event in stream_events {
        assert!(
            chat_completion_collector
                .ingest_event(stream_event)
                .is_none()
        );
    }
    let chat_completion_response = chat_completion_collector
        .into_response(
            28,
            99,
            0,
            ipc_protocol::ChatGenerationCompletionReason::EndOfSequence,
        )
        .expect("the reply assembles");
    serde_json::to_string(&chat_completion_response).expect("the reply serializes")
}

#[test]
fn should_put_a_drafting_report_into_the_complete_replys_usage() {
    let reply_json = collected_reply_json(vec![
        crate::ChatGenerationStreamEvent::TextFragment("def merge(a, b):".to_owned()),
        crate::ChatGenerationStreamEvent::DraftingReport(RequestDraftingStatistics {
            round_count: 26,
            drafted_token_count: 78,
            accepted_token_count: 74,
        }),
    ]);
    assert!(
        reply_json.contains(r#""accepted_prediction_tokens":74"#),
        "{reply_json}"
    );
}

#[test]
fn should_report_no_prediction_tokens_for_a_reply_without_a_drafting_report() {
    let reply_json = collected_reply_json(vec![crate::ChatGenerationStreamEvent::TextFragment(
        "Shall I compare thee".to_owned(),
    )]);
    assert!(
        !reply_json.contains("completion_tokens_details"),
        "{reply_json}"
    );
}
