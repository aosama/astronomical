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

use super::thinking_budget_journey::{ThinkingBudgetJourney, run_thinking_budget_journey};

/// The field name a coding agent sends by default when it believes the server
/// supports thinking budgets, per the agent's compatibility resolution order.
const CLIENT_THINKING_BUDGET_FIELD_NAME: &str = "thinking_token_budget";

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST surface and smallest configured Qwen3.5 model"]
async fn should_enforce_the_coding_agent_thinking_budget_field_before_visible_answer_content() {
    run_thinking_budget_journey(&ThinkingBudgetJourney {
        log_label: "client-thinking-budget-rest",
        budget_field_name: CLIENT_THINKING_BUDGET_FIELD_NAME,
    })
    .await;
}
