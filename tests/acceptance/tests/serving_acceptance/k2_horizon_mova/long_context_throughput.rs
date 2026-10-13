//! Multi-turn decode rates must be measured at realistic accumulated context.
//!
//! Turn one prefills the complete Romeo and Juliet source (~7k tokens) and
//! turn two appends a second copy so the same conversation reaches ~13k
//! tokens while the persistent prompt cache restores the shared prefix.
//! Decode tokens per second come from the engine attribution log, never a
//! client wall clock.

use serde_json::json;
use tokio::time::timeout;

use crate::serving_acceptance::chat::openai_rest::{
    assert_successful_streaming_chat_response, post_chat_completion,
};

use super::support::{
    JOURNEY_TIMEOUT, assert_k2_is_advertised, decode_throughput_from_report,
    full_romeo_and_juliet_source, launch_k2_rest_server_with_quantized_kv, public_model_id,
    read_generation_attribution_reports, stop_k2_rest_server,
};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "measures K2 Horizon MoVA decode across accumulated long-context turns"]
async fn should_measure_long_context_decode_throughput_across_turns() {
    timeout(JOURNEY_TIMEOUT, run_long_context_journey())
        .await
        .expect("the K2 Horizon MoVA long-context journey must finish within 115 seconds");
}

async fn run_long_context_journey() {
    let (isolated_development_home, rest_server) = launch_k2_rest_server_with_quantized_kv().await;
    let server_address = rest_server.server_address;
    assert_k2_is_advertised(server_address).await;
    let model_id = public_model_id();
    let source = full_romeo_and_juliet_source();

    eprintln!("[k2-long-context] phase=turn-1 model={model_id}");
    let first_turn = post_chat_completion(
        server_address,
        json!({
            "model": model_id,
            "messages": [{
                "role": "user",
                "content": format!(
                    "Here is the play. Reply with one short sentence naming the two households.\n\n{source}",
                ),
            }],
            "stream": true,
            "temperature": 1,
            "max_tokens": 64,
        })
        .to_string(),
    )
    .await;
    assert_successful_streaming_chat_response(&first_turn);

    eprintln!("[k2-long-context] phase=turn-2 model={model_id}");
    let second_turn = post_chat_completion(
        server_address,
        json!({
            "model": model_id,
            "messages": [{
                "role": "user",
                "content": format!(
                    "Here is the play. Reply with one short sentence naming the two households.\n\n{source}\nHere is the play again for reference. Name the household that Romeo belongs to.\n\n{source}",
                ),
            }],
            "stream": true,
            "temperature": 1,
            "max_tokens": 128,
        })
        .to_string(),
    )
    .await;
    assert_successful_streaming_chat_response(&second_turn);
    stop_k2_rest_server(rest_server).await;

    let reports = read_generation_attribution_reports(isolated_development_home.path());
    assert!(
        reports.len() >= 2,
        "both long-context turns must produce generation attribution reports, found {}",
        reports.len()
    );
    for (turn_index, report) in reports.iter().enumerate().take(2) {
        let (decode_tokens, decode_seconds, prefill_spans, prefill_seconds) =
            decode_throughput_from_report(report);
        let decode_tokens_per_second = if decode_seconds > 0.0 {
            decode_tokens as f64 / decode_seconds
        } else {
            0.0
        };
        eprintln!(
            "[k2-long-context] turn={} decode_tokens={decode_tokens} decode_seconds={decode_seconds:.2} decode_tok_per_second={decode_tokens_per_second:.2} prefill_spans={prefill_spans} prefill_seconds={prefill_seconds:.2}",
            turn_index + 1,
        );
        assert!(
            decode_tokens > 0 && decode_seconds > 0.0,
            "turn {} must record engine decode spans: {report}",
            turn_index + 1,
        );
        assert!(
            decode_tokens_per_second >= 40.0,
            "turn {} decode throughput must reach 40 tok/s at accumulated context for a 4B-active MoE, got {decode_tokens_per_second:.2}",
            turn_index + 1,
        );
    }
}
