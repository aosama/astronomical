//! Public REST acceptance that a coding-agent thinking-budget field is enforced.
//!
//! Coding agents running on the OpenAI-compatible surface transmit the
//! reasoning budget under a different field name than the canonical
//! `thinking_budget`: the agent resolves its budget field from its provider
//! compatibility configuration, falling back to `thinking_token_budget` when
//! the server declares thinking-budget support. This journey mirrors that
//! exact request shape — the budget under the coding-agent field name, with
//! streaming enabled — and demands the identical hard enforcement contract
//! the canonical journey already proves. Before the field is accepted, the
//! budget is silently dropped by lenient JSON parsing, the model is free to
//! reason for the whole output budget, and the forced-transition attribution
//! never appears; that failing run is the recorded reproduction of the
//! reported defect, and the same journey guards the fix afterwards.

use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5Tokenizer};
use serde_json::json;

use super::openai_rest::{
    E2E_TIMEOUT, launch_serving_rest_server_for_model, post_chat_completion,
    stop_serving_rest_server,
};
use super::thinking_budget_support::{
    MAXIMUM_OUTPUT_TOKEN_COUNT, MODEL_OWNED_TRANSITION_TEXT, ROMEO_AND_JULIET_SOURCE,
    THINKING_BUDGET_TOKEN_COUNT, assert_forced_transition_attribution, parse_streamed_completion,
    write_thinking_budget_acceptance_config,
};
use crate::small_dense_model::configured_deployment_litmus_model;

/// The field name a coding agent sends by default when it believes the server
/// supports thinking budgets, per the agent's compatibility resolution order.
const CLIENT_THINKING_BUDGET_FIELD_NAME: &str = "thinking_token_budget";

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST surface and smallest configured Qwen3.5 model"]
async fn should_enforce_the_coding_agent_thinking_budget_field_before_visible_answer_content() {
    tokio::time::timeout(E2E_TIMEOUT, async {
        let selected_model = configured_deployment_litmus_model();
        let validated_artifact = Qwen3_5ArtifactValidator::new()
            .validate(
                &selected_model.model_directory,
                u32::from(MAXIMUM_OUTPUT_TOKEN_COUNT),
            )
            .expect("the smallest configured Qwen3.5 artifact should validate");
        let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
            .expect("the smallest configured Qwen3.5 tokenizer should load");
        let expected_forced_transition_token_count = u64::try_from(
            tokenizer
                .forced_thinking_transition_token_ids()
                .len(),
        )
        .expect("the forced-transition token count should fit in u64");
        let isolated_worker_home = tempfile::tempdir()
            .expect("the client-budget REST journey should create an isolated worker home");
        write_thinking_budget_acceptance_config(
            isolated_worker_home.path(),
            &selected_model.model_id,
            &selected_model.model_directory,
            MAXIMUM_OUTPUT_TOKEN_COUNT,
        );
        let performance_log_directory = tempfile::tempdir()
            .expect("the client-budget REST journey should create a performance-log directory");
        let rest_server = launch_serving_rest_server_for_model(
            &selected_model.model_id,
            selected_model.model_directory,
            Some(isolated_worker_home.path()),
            Some(performance_log_directory.path()),
        )
        .await;

        eprintln!(
            "[client-thinking-budget-rest] status=progress phase=request model={} budget_field={CLIENT_THINKING_BUDGET_FIELD_NAME} budget_tokens={THINKING_BUDGET_TOKEN_COUNT}",
            selected_model.model_id
        );
        let chat_response = post_chat_completion(
            rest_server.server_address,
            client_thinking_budget_request_body(&selected_model.model_id),
        )
        .await;
        let streamed_completion = parse_streamed_completion(&chat_response);
        stop_serving_rest_server(rest_server).await;

        assert!(
            streamed_completion
                .reasoning_content
                .contains(MODEL_OWNED_TRANSITION_TEXT),
            "the coding-agent thinking budget must be enforced: the public reasoning stream must contain the complete model-owned transition: {:?}",
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
        assert_forced_transition_attribution(
            isolated_worker_home.path(),
            expected_forced_transition_token_count,
        );
        eprintln!(
            "[client-thinking-budget-rest] status=success budget_field={CLIENT_THINKING_BUDGET_FIELD_NAME} forced_transition_tokens={expected_forced_transition_token_count} visible_characters={}",
            streamed_completion.visible_content.len()
        );
    })
    .await
    .expect("the client thinking-budget REST journey must finish within 115 seconds");
}

fn client_thinking_budget_request_body(model_id: &str) -> String {
    json!({
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": format!(
                "Summarize this Romeo and Juliet excerpt in one sentence.\n\n{}",
                ROMEO_AND_JULIET_SOURCE.chars().take(512).collect::<String>()
            ),
        }],
        "stream": true,
        "temperature": 1,
        (CLIENT_THINKING_BUDGET_FIELD_NAME): THINKING_BUDGET_TOKEN_COUNT,
        "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
    })
    .to_string()
}
