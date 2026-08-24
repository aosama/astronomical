use astronomical_model_serving::{Qwen3_5ThinkingBudgetError, Qwen3_5ThinkingBudgetState};

const THINK_END_TOKEN_ID: u32 = 90;
const TOOL_CALL_START_TOKEN_ID: u32 = 91;

#[test]
fn should_commit_the_complete_forced_transition_after_preserving_the_reasoning_allowance() {
    let forced_transition_token_ids = vec![70, 71, THINK_END_TOKEN_ID];
    let mut thinking_budget_state = Qwen3_5ThinkingBudgetState::new(
        true,
        Some(3),
        forced_transition_token_ids.clone(),
        vec![THINK_END_TOKEN_ID, TOOL_CALL_START_TOKEN_ID],
    )
    .expect("a bounded thinking transition should be valid");

    for reasoning_token_id in [10, 11, 12] {
        assert!(
            thinking_budget_state
                .observe_committed_token(reasoning_token_id)
                .expect("an ordinary reasoning token should commit")
        );
    }
    assert_eq!(thinking_budget_state.thinking_token_count(), 3);

    for forced_transition_token_id in forced_transition_token_ids {
        assert_eq!(
            thinking_budget_state
                .next_forced_transition_token_id()
                .expect("the next forced token should resolve"),
            Some(forced_transition_token_id)
        );
        let is_reasoning_token = thinking_budget_state
            .observe_committed_token(forced_transition_token_id)
            .expect("the forced token should commit");
        assert_eq!(
            is_reasoning_token,
            forced_transition_token_id != THINK_END_TOKEN_ID
        );
    }

    assert!(!thinking_budget_state.is_inside_thinking());
    assert_eq!(
        thinking_budget_state
            .next_forced_transition_token_id()
            .expect("the completed transition should remain inactive"),
        None
    );
    assert!(
        !thinking_budget_state
            .observe_committed_token(20)
            .expect("visible answer generation should continue")
    );
}

#[test]
fn should_preserve_natural_and_implicit_reasoning_exits_without_forcing() {
    for natural_reasoning_end_token_id in [THINK_END_TOKEN_ID, TOOL_CALL_START_TOKEN_ID] {
        let mut thinking_budget_state = Qwen3_5ThinkingBudgetState::new(
            true,
            Some(8),
            vec![70, THINK_END_TOKEN_ID],
            vec![THINK_END_TOKEN_ID, TOOL_CALL_START_TOKEN_ID],
        )
        .expect("the reasoning boundary should be valid");

        assert!(
            thinking_budget_state
                .observe_committed_token(10)
                .expect("reasoning should begin normally")
        );
        assert!(
            !thinking_budget_state
                .observe_committed_token(natural_reasoning_end_token_id)
                .expect("the natural reasoning exit should commit")
        );
        assert!(!thinking_budget_state.is_inside_thinking());
        assert_eq!(
            thinking_budget_state
                .next_forced_transition_token_id()
                .expect("a natural exit should not force a transition"),
            None
        );
    }
}

#[test]
fn should_start_in_visible_answer_mode_for_a_zero_budget() {
    let mut thinking_budget_state =
        Qwen3_5ThinkingBudgetState::new(true, Some(0), Vec::new(), vec![THINK_END_TOKEN_ID])
            .expect("a zero budget should not require a forced transition");

    assert!(!thinking_budget_state.is_inside_thinking());
    assert_eq!(thinking_budget_state.thinking_budget(), None);
    assert!(
        !thinking_budget_state
            .observe_committed_token(10)
            .expect("a visible token should remain visible")
    );
}

#[test]
fn should_reject_a_forced_transition_with_an_early_reasoning_boundary() {
    let configuration_error = Qwen3_5ThinkingBudgetState::new(
        true,
        Some(3),
        vec![70, THINK_END_TOKEN_ID, 71, THINK_END_TOKEN_ID],
        vec![THINK_END_TOKEN_ID, TOOL_CALL_START_TOKEN_ID],
    )
    .expect_err("a forced transition must not end reasoning before its final token");

    assert_eq!(
        configuration_error,
        Qwen3_5ThinkingBudgetError::TransitionEndsReasoningEarly
    );
}
