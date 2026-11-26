//! The one end-to-end thinking-budget journey, run once per budget field name.
//!
//! The canonical `thinking_budget` journey and the coding-agent field journey
//! differ only in the JSON field that carries the budget and in their log
//! label, so the launch, request and enforcement assertions live here once.

use model_serving::{Qwen3_5ArtifactValidator, Qwen3_5Tokenizer};
use serde_json::{Map, Value, json};

use super::openai_rest::{
    E2E_TIMEOUT, launch_serving_rest_server_for_model, post_chat_completion,
    stop_serving_rest_server,
};
use super::thinking_budget_support::{
    MAXIMUM_OUTPUT_TOKEN_COUNT, MODEL_OWNED_TRANSITION_TEXT, ROMEO_AND_JULIET_SOURCE,
    StreamedCompletion, THINKING_BUDGET_TOKEN_COUNT, assert_forced_transition_attribution,
    parse_streamed_completion, write_thinking_budget_acceptance_config,
};
use crate::small_dense_model;

pub(super) struct ThinkingBudgetJourney {
    /// Prefix of the progress lines, so each journey stays greppable in logs.
    pub(super) log_label: &'static str,
    /// The request field that carries the reasoning budget.
    pub(super) budget_field_name: &'static str,
}

pub(super) async fn run_thinking_budget_journey(journey: &ThinkingBudgetJourney) {
    tokio::time::timeout(E2E_TIMEOUT, run_journey_without_deadline(journey))
        .await
        .unwrap_or_else(|_| {
            panic!(
                "the {} REST journey must finish within 115 seconds",
                journey.log_label
            )
        });
}

async fn run_journey_without_deadline(journey: &ThinkingBudgetJourney) {
    let selected_model = small_dense_model::configured_deployment_litmus_model();
    let expected_forced_transition_token_count =
        forced_transition_token_count(&selected_model.model_directory);
    let isolated_worker_home = tempfile::tempdir()
        .expect("the thinking-budget REST journey should create an isolated worker home");
    write_thinking_budget_acceptance_config(
        isolated_worker_home.path(),
        &selected_model.model_id,
        &selected_model.model_directory,
        MAXIMUM_OUTPUT_TOKEN_COUNT,
    );
    let performance_log_directory = tempfile::tempdir()
        .expect("the thinking-budget REST journey should create a performance-log directory");
    let rest_server = launch_serving_rest_server_for_model(
        &selected_model.model_id,
        selected_model.model_directory,
        Some(isolated_worker_home.path()),
        Some(performance_log_directory.path()),
    )
    .await;

    eprintln!(
        "[{}] status=progress phase=request model={} budget_field={} budget_tokens={THINKING_BUDGET_TOKEN_COUNT}",
        journey.log_label, selected_model.model_id, journey.budget_field_name
    );
    let chat_response = post_chat_completion(
        rest_server.server_address,
        thinking_budget_request_body(&selected_model.model_id, journey.budget_field_name),
    )
    .await;
    let streamed_completion = parse_streamed_completion(&chat_response);
    stop_serving_rest_server(rest_server).await;

    assert_budget_enforced_before_visible_answer(&streamed_completion);
    assert_forced_transition_attribution(
        isolated_worker_home.path(),
        expected_forced_transition_token_count,
    );
    eprintln!(
        "[{}] status=success budget_field={} forced_transition_tokens={expected_forced_transition_token_count} visible_characters={}",
        journey.log_label,
        journey.budget_field_name,
        streamed_completion.visible_content.len()
    );
}

fn forced_transition_token_count(model_directory: &std::path::Path) -> u64 {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, u32::from(MAXIMUM_OUTPUT_TOKEN_COUNT))
        .expect("the smallest configured Qwen3.5 artifact should validate");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the smallest configured Qwen3.5 tokenizer should load");
    u64::try_from(tokenizer.forced_thinking_transition_token_ids().len())
        .expect("the forced-transition token count should fit in u64")
}

fn assert_budget_enforced_before_visible_answer(streamed_completion: &StreamedCompletion) {
    assert!(
        streamed_completion
            .reasoning_content
            .contains(MODEL_OWNED_TRANSITION_TEXT),
        "the thinking budget must be enforced: the public reasoning stream must contain the complete model-owned transition: {:?}",
        streamed_completion.reasoning_content
    );
    assert!(
        !streamed_completion.visible_content.trim().is_empty(),
        "visible answer content must follow the committed reasoning transition under the remaining output budget"
    );
    assert!(
        !streamed_completion.reasoning_arrived_after_visible_content,
        "reasoning content must not resume after visible answer streaming begins"
    );
}

fn thinking_budget_request_body(model_id: &str, budget_field_name: &str) -> String {
    let mut request_fields = Map::new();
    request_fields.insert("model".to_owned(), json!(model_id));
    request_fields.insert(
        "messages".to_owned(),
        json!([{
            "role": "user",
            "content": format!(
                "Summarize this Romeo and Juliet excerpt in one sentence.\n\n{}",
                ROMEO_AND_JULIET_SOURCE.chars().take(512).collect::<String>()
            ),
        }]),
    );
    request_fields.insert("stream".to_owned(), json!(true));
    request_fields.insert("temperature".to_owned(), json!(1));
    request_fields.insert(
        budget_field_name.to_owned(),
        json!(THINKING_BUDGET_TOKEN_COUNT),
    );
    request_fields.insert("max_tokens".to_owned(), json!(MAXIMUM_OUTPUT_TOKEN_COUNT));
    Value::Object(request_fields).to_string()
}
