#[allow(dead_code)]
#[path = "../../src/qwen3_5/inference_execution/prefill_execution_context.rs"]
mod prefill_execution_context;
#[path = "../../src/qwen3_5/inference_execution/speculative_prefill_eligibility.rs"]
mod speculative_prefill_eligibility;
#[path = "speculative_prefill_policy.rs"]
mod speculative_prefill_policy;

use prefill_execution_context::{
    CAPACITY_REDUCED_CONTEXT_FLAG, Qwen3_5PrefillExecutionContext,
    SPECULATIVE_PREFILL_TARGET_ONLY_PREFIX_CONTEXT_FLAG,
};
use speculative_prefill_eligibility::{
    Qwen3_5SpeculativePrefillRequestEligibility, qwen3_5_speculative_prefill_request_eligibility,
};
use speculative_prefill_policy::{
    Qwen3_5SpeculativePrefillChunckMode,
    qwen3_5_merge_speculative_prefill_selection_with_image_pad_positions,
    qwen3_5_select_speculative_prefill_token_positions,
    qwen3_5_selected_speculative_prefill_positions_for_range,
    qwen3_5_speculative_prefill_chunck_mode, qwen3_5_speculative_prefill_scoring_plan,
    qwen3_5_speculative_prefill_selectable_importance_score_range,
};

#[test]
fn should_report_each_speculative_prefill_request_eligibility_outcome() {
    assert_eq!(
        qwen3_5_speculative_prefill_request_eligibility(
            false, true, false, 8_192, 8_192, 0, false, false,
        ),
        Qwen3_5SpeculativePrefillRequestEligibility::DisabledByConfiguration,
    );
    assert_eq!(
        qwen3_5_speculative_prefill_request_eligibility(
            true, false, false, 8_192, 8_192, 0, false, false,
        ),
        Qwen3_5SpeculativePrefillRequestEligibility::DraftModelUnavailable,
    );
    assert_eq!(
        qwen3_5_speculative_prefill_request_eligibility(
            true, true, false, 40_000, 8_192, 35_000, false, false,
        ),
        Qwen3_5SpeculativePrefillRequestEligibility::PromptBelowMinimum,
    );
    assert_eq!(
        qwen3_5_speculative_prefill_request_eligibility(
            true, true, false, 8_192, 8_192, 8_191, false, false,
        ),
        Qwen3_5SpeculativePrefillRequestEligibility::PromptAlreadyRestored,
    );
    assert_eq!(
        qwen3_5_speculative_prefill_request_eligibility(
            true, true, false, 8_192, 8_192, 0, true, false,
        ),
        Qwen3_5SpeculativePrefillRequestEligibility::PrecomputedVisualEmbeddingsPresent,
    );
    assert_eq!(
        qwen3_5_speculative_prefill_request_eligibility(
            true, true, false, 8_192, 8_192, 0, false, true,
        ),
        Qwen3_5SpeculativePrefillRequestEligibility::DraftModelDoesNotSupportProcessedVisualImages,
    );
    let eligible_request_eligibility = qwen3_5_speculative_prefill_request_eligibility(
        true, true, false, 8_192, 8_192, 0, false, false,
    );
    assert_eq!(
        eligible_request_eligibility,
        Qwen3_5SpeculativePrefillRequestEligibility::Eligible,
    );
    assert!(eligible_request_eligibility.is_eligible());
    assert_eq!(eligible_request_eligibility.identifier(), "eligible");
    assert_eq!(
        qwen3_5_speculative_prefill_request_eligibility(true, true, false, 1, 1, 0, false, false,),
        Qwen3_5SpeculativePrefillRequestEligibility::PromptAlreadyRestored,
    );
}

#[test]
fn should_apply_the_speculative_prefill_threshold_to_the_uncached_follow_up_suffix() {
    assert_eq!(
        qwen3_5_speculative_prefill_request_eligibility(
            true, true, false, 50_000, 8_192, 40_000, false, false,
        ),
        Qwen3_5SpeculativePrefillRequestEligibility::Eligible,
    );
    assert_eq!(
        qwen3_5_speculative_prefill_request_eligibility(
            true, true, false, 47_000, 8_192, 40_000, false, false,
        ),
        Qwen3_5SpeculativePrefillRequestEligibility::PromptBelowMinimum,
    );
}

#[test]
fn should_keep_speculative_prefill_eligible_for_a_40k_plus_10k_follow_up_after_cache_restore() {
    assert_eq!(
        qwen3_5_speculative_prefill_request_eligibility(
            true, true, false, 51_200, 8_192, 40_960, false, false,
        ),
        Qwen3_5SpeculativePrefillRequestEligibility::Eligible,
    );
    assert_eq!(
        qwen3_5_speculative_prefill_request_eligibility(
            true, true, false, 40_960, 8_192, 0, false, false,
        ),
        Qwen3_5SpeculativePrefillRequestEligibility::Eligible,
    );
    assert_eq!(
        qwen3_5_speculative_prefill_request_eligibility(
            true, true, false, 48_959, 8_192, 40_960, false, false,
        ),
        Qwen3_5SpeculativePrefillRequestEligibility::PromptBelowMinimum,
    );
}

#[test]
fn should_allow_processed_images_when_the_draft_supports_their_input_contract() {
    assert_eq!(
        qwen3_5_speculative_prefill_request_eligibility(
            true, true, true, 8_192, 8_192, 0, false, true,
        ),
        Qwen3_5SpeculativePrefillRequestEligibility::Eligible,
    );
}

#[test]
fn should_score_the_complete_prompt_while_reserving_the_final_target_token() {
    let (draft_scoring_token_range, selectable_importance_score_count) =
        qwen3_5_speculative_prefill_scoring_plan(2, 5, 6)
            .expect("the suffix should produce a valid scoring plan");

    assert_eq!(draft_scoring_token_range, 2..6);
    assert_eq!(selectable_importance_score_count, 3);
}

#[test]
fn should_select_the_target_suffix_from_full_visual_draft_scores_after_cache_restore() {
    let selectable_importance_score_range =
        qwen3_5_speculative_prefill_selectable_importance_score_range(0, 46_341, 45_056, 1_284)
            .expect("a full visual draft score vector must contain the target cache-miss suffix");

    assert_eq!(selectable_importance_score_range, 45_056..46_340);
}

#[test]
fn should_select_every_nonfinal_score_from_a_cache_deleted_visual_prompt() {
    let selectable_importance_score_range =
        qwen3_5_speculative_prefill_selectable_importance_score_range(0, 47_047, 0, 47_046)
            .expect("a cache-deleted visual prompt must select all scores except its final token");

    assert_eq!(selectable_importance_score_range, 0..47_046);
}

#[test]
fn should_use_ordinary_target_prefill_when_mtp_is_disabled_for_a_nonterminal_chunck() {
    assert_eq!(
        qwen3_5_speculative_prefill_chunck_mode(false, 3, 5),
        Qwen3_5SpeculativePrefillChunckMode::OrdinaryTarget,
    );
}

#[test]
fn should_use_ordinary_target_prefill_when_mtp_is_disabled_for_a_terminal_chunck() {
    assert_eq!(
        qwen3_5_speculative_prefill_chunck_mode(false, 5, 5),
        Qwen3_5SpeculativePrefillChunckMode::OrdinaryTarget,
    );
}

#[test]
fn should_use_target_only_mtp_prefix_prefill_for_an_active_nonterminal_chunck() {
    assert_eq!(
        qwen3_5_speculative_prefill_chunck_mode(true, 3, 5),
        Qwen3_5SpeculativePrefillChunckMode::TargetOnlyPrefix,
    );
}

#[test]
fn should_use_terminal_mtp_capture_for_an_active_terminal_chunck() {
    assert_eq!(
        qwen3_5_speculative_prefill_chunck_mode(true, 5, 5),
        Qwen3_5SpeculativePrefillChunckMode::TerminalAdditionalHistoryCapture,
    );
}

#[test]
fn should_keep_target_only_mtp_prefix_and_capacity_reduced_context_flags_distinct() {
    let target_only_mtp_prefix_context =
        Qwen3_5PrefillExecutionContext::new(false, true, false, false)
            .with_target_only_prefix(true);

    assert_ne!(
        SPECULATIVE_PREFILL_TARGET_ONLY_PREFIX_CONTEXT_FLAG,
        CAPACITY_REDUCED_CONTEXT_FLAG,
    );
    assert_eq!(
        target_only_mtp_prefix_context.context_identifier_flags()
            & SPECULATIVE_PREFILL_TARGET_ONLY_PREFIX_CONTEXT_FLAG,
        SPECULATIVE_PREFILL_TARGET_ONLY_PREFIX_CONTEXT_FLAG,
    );
    assert_eq!(
        target_only_mtp_prefix_context.context_identifier_flags() & CAPACITY_REDUCED_CONTEXT_FLAG,
        0,
    );
}

#[test]
fn should_select_highest_scoring_chunks_and_always_retain_the_trailing_window() {
    let importance_scores = [
        0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3,
    ];

    let selected_token_positions =
        qwen3_5_select_speculative_prefill_token_positions(&importance_scores, 50, 4, 4)
            .expect("selection should succeed");

    assert_eq!(selected_token_positions, vec![4, 5, 6, 7, 12, 13, 14, 15]);
}

#[test]
fn should_let_the_mandatory_trailing_window_take_precedence_for_short_prompts() {
    let importance_scores = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];

    let selected_token_positions =
        qwen3_5_select_speculative_prefill_token_positions(&importance_scores, 10, 4, 8)
            .expect("selection should succeed");

    assert_eq!(selected_token_positions, (0..8).collect::<Vec<_>>());
}

#[test]
fn should_retain_every_prompt_position_when_keep_percentage_is_full() {
    let importance_scores = [0.1, 0.2, 0.3, 0.4, 0.5];

    let selected_token_positions =
        qwen3_5_select_speculative_prefill_token_positions(&importance_scores, 100, 3, 1)
            .expect("selection should succeed");

    assert_eq!(selected_token_positions, (0..5).collect::<Vec<_>>());
}

#[test]
fn should_reject_invalid_speculative_prefill_selection_inputs() {
    assert!(qwen3_5_select_speculative_prefill_token_positions(&[], 20, 32, 512).is_err());
    assert!(qwen3_5_select_speculative_prefill_token_positions(&[0.1], 0, 32, 512).is_err());
    assert!(qwen3_5_select_speculative_prefill_token_positions(&[0.1], 20, 0, 512).is_err());
    assert!(qwen3_5_select_speculative_prefill_token_positions(&[f32::NAN], 20, 32, 512).is_err());
}

#[test]
fn should_partition_selected_prompt_positions_by_original_prefill_range() {
    let selected_token_positions = [1, 4, 7, 10];

    assert_eq!(
        qwen3_5_selected_speculative_prefill_positions_for_range(&selected_token_positions, 3, 8,),
        vec![4, 7],
    );
    assert!(
        qwen3_5_selected_speculative_prefill_positions_for_range(&selected_token_positions, 8, 8,)
            .is_empty()
    );
}

#[test]
fn should_retain_every_image_pad_position_alongside_draft_selected_positions() {
    let image_pad_positions = [1, 2, 5, 6, 7];
    let visual_embedding_token_count = 1;

    let selected_prompt_positions =
        qwen3_5_merge_speculative_prefill_selection_with_image_pad_positions(
            vec![0, 3, 8],
            &image_pad_positions,
            visual_embedding_token_count,
        );

    assert_eq!(selected_prompt_positions, vec![0, 1, 2, 3, 5, 6, 7, 8],);
}

#[test]
fn should_retain_only_selectable_image_pad_positions_when_the_final_token_is_reserved() {
    let image_pad_positions = [1, 2];
    let visual_embedding_token_count = 1;

    let selected_prompt_positions =
        qwen3_5_merge_speculative_prefill_selection_with_image_pad_positions(
            vec![0, 1],
            &image_pad_positions,
            visual_embedding_token_count,
        );

    assert_eq!(selected_prompt_positions, vec![0, 1, 2]);
}
