//! Chat Completions must summarize the play excerpt with the folded ternary pack.

use serde_json::json;
use tokio::time::timeout;

use crate::serving_acceptance::chat::openai_rest::{
    assert_successful_streaming_chat_response, post_chat_completion,
};

use crate::support::romeo_and_juliet;

use super::support::{
    JOURNEY_TIMEOUT, launch_prism_rest_server, public_model_id, server_address,
    stop_prism_rest_server,
};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the real roughly 8 GB folded ternary pack and streams greedy and sampled replies"]
async fn should_serve_greedy_and_sampled_chat_completions_from_the_folded_pack() {
    timeout(JOURNEY_TIMEOUT, run_chat_journey())
        .await
        .expect("the folded-ternary Chat Completions journey must finish within 115 seconds");
}

async fn run_chat_journey() {
    let (isolated_development_home, rest_server) = launch_prism_rest_server().await;
    let address = server_address(&rest_server);
    let model_id = public_model_id();
    let user_prompt = romeo_and_juliet::summary_prompt(120);
    eprintln!("[prism-hadamard] phase=chat model={model_id}");
    // The one-token case guards the prefill-to-output boundary: the first
    // sampled token must still be emitted when it exhausts the request budget.
    for (temperature, maximum_output_tokens) in [(0.0_f64, 64_u16), (1.0, 64), (0.0, 1)] {
        let chat_response = post_chat_completion(
            address,
            json!({
                "model": model_id,
                "messages": [{
                    "role": "user",
                    "content": user_prompt.clone(),
                }],
                "stream": true,
                "thinking_budget": 0,
                "temperature": temperature,
                "max_tokens": maximum_output_tokens,
            })
            .to_string(),
        )
        .await;
        assert_successful_streaming_chat_response(&chat_response);
        let streamed_text = streamed_chat_text(&chat_response);
        eprintln!(
            "[prism-hadamard] temperature={temperature} max_tokens={maximum_output_tokens} chat_streamed_text={streamed_text}"
        );
        assert!(
            !streamed_text.trim().is_empty(),
            "Chat Completions must stream visible text at temperature {temperature} and max_tokens={maximum_output_tokens}"
        );
    }
    stop_prism_rest_server(rest_server).await;

    let performance_attribution_log_path = isolated_development_home
        .path()
        .join(".astronomical-dev/logs/performance-attribution.jsonl");
    let performance_reports = std::fs::read_to_string(performance_attribution_log_path)
        .expect("the enabled model attribution log should be written");
    let parsed_performance_reports = performance_reports
        .lines()
        .map(|report_line| {
            serde_json::from_str::<serde_json::Value>(report_line)
                .expect("each attribution line should be valid JSON")
        })
        .collect::<Vec<_>>();
    assert!(
        parsed_performance_reports
            .iter()
            .any(|report| report["report_kind"] == "model_loading"),
        "the enabled attribution log should include model loading: {performance_reports}"
    );
    let generation_report = parsed_performance_reports
        .iter()
        .find(|report| report["report_kind"] == "generation")
        .expect("the completed request should write a generation attribution report");
    let generation_operations = generation_report["operations"]
        .as_array()
        .expect("the generation report should include operation measurements");
    assert!(
        generation_operations
            .iter()
            .any(|operation| operation["operation"] == "prompt_prefill_advance_span")
    );
    assert!(
        generation_operations
            .iter()
            .any(|operation| operation["operation"] == "decode_sampling_span")
    );
}

/// Joins the streamed deltas into the visible reply text.
fn streamed_chat_text(chat_response: &str) -> String {
    chat_response
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str::<serde_json::Value>(data).ok())
        .filter_map(|delta| {
            delta["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect()
}
