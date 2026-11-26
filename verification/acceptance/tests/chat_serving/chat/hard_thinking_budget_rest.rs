//! Public REST acceptance of Qwen's model-owned hard reasoning transition.

use super::thinking_budget_journey::{ThinkingBudgetJourney, run_thinking_budget_journey};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST surface and smallest configured Qwen3.5 model"]
async fn should_commit_the_hard_thinking_budget_before_visible_answer_content() {
    run_thinking_budget_journey(&ThinkingBudgetJourney {
        log_label: "hard-thinking-budget-rest",
        budget_field_name: "thinking_budget",
    })
    .await;
}
