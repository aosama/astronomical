#[allow(dead_code)]
#[path = "../../src/qwen3_5/inference_execution/prefill_execution_context.rs"]
mod prefill_execution_context;
#[path = "../../src/qwen3_5/inference_execution/speculative_prefill_eligibility.rs"]
mod speculative_prefill_eligibility;
#[path = "../../src/qwen3_5/inference_execution/speculative_prefill_selection.rs"]
mod speculative_prefill_policy;
#[path = "../../src/qwen3_5/inference_execution/speculative_prefill_control_span.rs"]
mod speculative_prefill_control_span;
#[path = "../../src/qwen3_5/inference_execution/speculative_prefill_failure.rs"]
mod speculative_prefill_failure;
#[path = "../../src/qwen3_5/inference_execution/speculative_prefill.rs"]
mod speculative_prefill_chunck_policy;

use prefill_execution_context::{
    CAPACITY_REDUCED_CONTEXT_FLAG, Qwen3_5PrefillExecutionContext,
    SPECULATIVE_PREFILL_TARGET_ONLY_PREFIX_CONTEXT_FLAG,
};
use astronomical_model_serving::InferenceEngineError;
use speculative_prefill_eligibility::{
    Qwen3_5SpeculativePrefillRequestEligibility, qwen3_5_speculative_prefill_request_eligibility,
};
use speculative_prefill_policy::{
    qwen3_5_merge_speculative_prefill_selection_with_image_pad_positions,
    qwen3_5_select_speculative_prefill_token_positions,
    qwen3_5_selected_speculative_prefill_positions_for_range,
    qwen3_5_speculative_prefill_scoring_plan,
    qwen3_5_speculative_prefill_selectable_importance_score_range,
};
use speculative_prefill_chunck_policy::{
    Qwen3_5SpeculativePrefillChunckMode, qwen3_5_speculative_prefill_chunck_mode,
};
use speculative_prefill_control_span::{
    qwen3_5_prefill_chunck_end_at_ordinary_target_control_span_boundary,
    qwen3_5_speculative_prefill_sparse_target_is_active,
};
use speculative_prefill_failure::{
    configured_speculative_prefill_activation_failure, configured_speculative_prefill_failure,
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
fn should_apply_keep_percentage_only_after_the_ordinary_target_control_span() {
    let ordinary_target_control_span_end_position = 1_024;
    let final_generation_kickoff_position = 11_025;
    let complete_prompt_token_count = 11_026;

    let (selectable_conversation_and_kickoff_range, selectable_conversation_token_count) =
        qwen3_5_speculative_prefill_scoring_plan(
            ordinary_target_control_span_end_position,
            final_generation_kickoff_position,
            complete_prompt_token_count,
        )
        .expect("the conversation after the control span should be selectable");

    assert_eq!(
        selectable_conversation_and_kickoff_range,
        ordinary_target_control_span_end_position..complete_prompt_token_count
    );
    assert_eq!(selectable_conversation_token_count, 10_001);
}

#[test]
fn should_end_ordinary_target_prefill_exactly_at_the_control_span_boundary() {
    assert_eq!(
        qwen3_5_prefill_chunck_end_at_ordinary_target_control_span_boundary(0, 2_048, 1_200),
        Some(1_200)
    );
    assert_eq!(
        qwen3_5_prefill_chunck_end_at_ordinary_target_control_span_boundary(
            0, 512, 1_200
        ),
        Some(512)
    );
    assert_eq!(
        qwen3_5_prefill_chunck_end_at_ordinary_target_control_span_boundary(
            1_200, 3_248, 1_200
        ),
        Some(3_248)
    );
}

#[test]
fn should_activate_sparse_target_processing_only_after_the_control_span() {
    assert!(!qwen3_5_speculative_prefill_sparse_target_is_active(
        true, 0, 1_200
    ));
    assert!(!qwen3_5_speculative_prefill_sparse_target_is_active(
        true, 1_199, 1_200
    ));
    assert!(qwen3_5_speculative_prefill_sparse_target_is_active(
        true, 1_200, 1_200
    ));
    assert!(!qwen3_5_speculative_prefill_sparse_target_is_active(
        false, 1_200, 1_200
    ));
}

#[test]
fn should_stop_a_configured_speculative_prefill_failure_without_target_only_retry() {
    let generation_failure = configured_speculative_prefill_failure(
        astronomical_ipc_protocol::RequestId::new(50),
        "draft scoring",
        "forced acceptance-test failure",
    );

    assert!(matches!(
        generation_failure,
        astronomical_model_serving::InferenceEngineError::InvalidRequest { ref reason }
            if reason.contains("request was stopped")
                && reason.contains("without a target-only retry")
    ));
}

#[test]
fn should_bound_every_configured_speculative_prefill_execution_failure_stage() {
    for failure_stage in [
        "drafter loading",
        "drafter memory admission",
        "drafter visual input assembly",
        "draft scoring or selection",
        "selection restoration",
        "sparse target input assembly",
        "sparse target execution",
        "drafter prompt-state persistence",
        "selection persistence",
        "exact target prompt-state persistence",
        "sparse target-state persistence",
    ] {
        let generation_failure = configured_speculative_prefill_failure(
            astronomical_ipc_protocol::RequestId::new(50),
            failure_stage,
            "private implementation details",
        );
        assert!(matches!(
            generation_failure,
            InferenceEngineError::InvalidRequest { ref reason }
                if reason.contains(failure_stage)
                    && reason.contains("without a target-only retry")
                    && !reason.contains("private implementation details")
        ));
    }
}

#[test]
fn should_reject_speculative_prefill_activation_when_a_required_purge_fails() {
    let model_loading_failure = configured_speculative_prefill_activation_failure(
        "target keep-percentage state purge",
        "private storage details that must not reach the user",
    );

    assert!(matches!(
        model_loading_failure,
        InferenceEngineError::Fatal { ref reason }
            if reason.contains("target keep-percentage state purge")
                && reason.contains("model use was stopped")
                && !reason.contains("private storage details")
    ));
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
fn should_retain_an_unaligned_mandatory_trailing_window_in_full() {
    let importance_scores = [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];

    let selected_token_positions =
        qwen3_5_select_speculative_prefill_token_positions(&importance_scores, 10, 4, 5)
            .expect("selection should retain every mandatory trailing token");

    assert_eq!(selected_token_positions, (4..10).collect::<Vec<_>>());
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
    let prompt_token_ids = [10, 99, 99, 20, 30, 99, 99, 99, 40];

    let selected_prompt_positions =
        qwen3_5_merge_speculative_prefill_selection_with_image_pad_positions(
            vec![0, 3, 8],
            &prompt_token_ids,
            0,
            prompt_token_ids.len(),
            99,
        )
        .expect("valid image-pad positions should merge");

    assert_eq!(selected_prompt_positions, vec![0, 1, 2, 3, 5, 6, 7, 8],);
}

#[test]
fn should_retain_only_selectable_image_pad_positions_when_the_final_token_is_reserved() {
    let prompt_token_ids = [10, 99, 99];

    let selected_prompt_positions =
        qwen3_5_merge_speculative_prefill_selection_with_image_pad_positions(
            vec![0, 1],
            &prompt_token_ids,
            0,
            2,
            99,
        )
        .expect("the reserved final prompt token should remain outside selection");

    assert_eq!(selected_prompt_positions, vec![0, 1]);
}
