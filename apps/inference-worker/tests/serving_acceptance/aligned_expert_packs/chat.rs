//! Chat Completions must answer a Romeo and Juliet household question.

use serde_json::{Value, json};
use tokio::time::timeout;

use crate::serving_acceptance::chat::openai_rest::{
    assert_successful_streaming_chat_response, post_chat_completion,
};

use super::support::{
    JOURNEY_TIMEOUT, STREAMING_MODEL_ID, assert_streaming_model_is_advertised, households_prompt,
    launch_streaming_model_rest_server_with_attribution, names_the_households,
    pack_streamed_source_summaries, read_generation_attribution_reports,
    stop_streaming_model_rest_server, streamed_chat_text,
};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "streams Romeo and Juliet households through Chat Completions for the converted expert-streaming model"]
async fn should_name_romeo_and_juliet_households_over_chat_completions() {
    timeout(JOURNEY_TIMEOUT, run_chat_journey())
        .await
        .expect("the expert-streaming Chat Completions journey must finish within 115 seconds");
}

async fn run_chat_journey() {
    let (isolated_home, rest_server) = launch_streaming_model_rest_server_with_attribution().await;
    let server_address = rest_server.server_address;
    assert_streaming_model_is_advertised(server_address).await;
    eprintln!("[aligned-expert-packs] phase=chat model={STREAMING_MODEL_ID}");
    let chat_response = post_chat_completion(
        server_address,
        json!({
            "model": STREAMING_MODEL_ID,
            "messages": [{
                "role": "user",
                "content": households_prompt(),
            }],
            "stream": true,
            "temperature": 1,
            "max_tokens": 512,
        })
        .to_string(),
    )
    .await;
    assert_successful_streaming_chat_response(&chat_response);
    let streamed_text = streamed_chat_text(&chat_response);
    eprintln!("[aligned-expert-packs] chat_streamed_text={streamed_text}");
    assert!(
        names_the_households(&streamed_text),
        "Chat Completions must name Montague or Capulet, got {streamed_text:?}"
    );
    let reports = read_generation_attribution_reports(isolated_home.path());
    assert!(
        !reports.is_empty(),
        "the chat journey must produce worker generation attribution reports"
    );
    let pack_streamed_summaries = pack_streamed_source_summaries(&reports);
    assert!(
        !pack_streamed_summaries.is_empty(),
        "attribution must record expert pages streamed through per-expert pack files, reports: {reports:?}"
    );
    for summary in &pack_streamed_summaries {
        // Per-expert packs mean one source file per routed expert: a summary
        // that streamed through packs must not group experts into shards.
        let streamed_expert_count = summary
            .get("total_streamed_expert_count")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let source_shard_count = summary
            .get("total_source_shard_count")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        assert_eq!(
            streamed_expert_count, source_shard_count,
            "pack-streamed pages must open one file per expert"
        );
    }
    eprintln!(
        "[aligned-expert-packs] phase=attribution pack_streamed_summaries={}",
        pack_streamed_summaries.len()
    );
    stop_streaming_model_rest_server(rest_server).await;
}
