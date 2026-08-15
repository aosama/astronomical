use astronomical_model_serving::{
    Qwen3_5PrefillExecutionContext, Qwen3_5PromptProcessingChunkSizer,
};

fn optimized_prompt_processing_chunk_sizer(
    maximum_prompt_processing_chunk_size_tokens: u32,
    configured_candidate_chunk_size_token_counts: Vec<u32>,
) -> Result<
    Qwen3_5PromptProcessingChunkSizer,
    astronomical_model_serving::Qwen3_5PromptProcessingChunkSizerError,
> {
    Qwen3_5PromptProcessingChunkSizer::for_optimized_with_behavior(
        maximum_prompt_processing_chunk_size_tokens,
        configured_candidate_chunk_size_token_counts,
        5,
        32_768,
    )
}

#[test]
fn should_report_position_boundaries_at_the_start_middle_and_end_of_a_configured_range() {
    for chunk_start_token_position in [0_usize, 512, 1_023] {
        let mut prompt_processing_chunk_sizer =
            Qwen3_5PromptProcessingChunkSizer::for_optimized_with_behavior(
                1_024,
                vec![128, 256, 512, 1_024],
                5,
                1_024,
            )
            .expect("the position-range boundary test should initialize the optimizer");
        prompt_processing_chunk_sizer.start_prompt_processing_request(0);
        let selected_chunk_end_token_position_exclusive = prompt_processing_chunk_sizer
            .next_prompt_processing_chunk_end(
                chunk_start_token_position,
                chunk_start_token_position.saturating_add(2_048),
            );
        assert!(selected_chunk_end_token_position_exclusive > chunk_start_token_position);
        prompt_processing_chunk_sizer.record_prompt_processing_chunk_elapsed_millis(128, 1);

        let chunk_outcome = prompt_processing_chunk_sizer
            .take_latest_prompt_processing_chunk_optimization_outcome()
            .expect("a completed optimized chunk should publish one outcome");
        assert_eq!(
            chunk_outcome.measurement_context.chunk_start_token_position,
            chunk_start_token_position
        );
        assert_eq!(
            chunk_outcome
                .measurement_context
                .position_range_start_token_position,
            0
        );
        assert_eq!(
            chunk_outcome
                .measurement_context
                .position_range_end_token_position_exclusive,
            1_024
        );
    }
}

#[test]
fn should_use_only_configured_candidates_below_the_validated_model_context_maximum() {
    const VALIDATED_MODEL_MAXIMUM_POSITION_COUNT: u32 = 8_193;
    let mut prompt_processing_chunk_sizer = optimized_prompt_processing_chunk_sizer(
        VALIDATED_MODEL_MAXIMUM_POSITION_COUNT,
        vec![1_024, 2_048, 4_096, 8_192, 16_384],
    )
    .expect("configured candidates below the model maximum should initialize");

    assert_eq!(
        prompt_processing_chunk_sizer.maximum_prompt_processing_chunk_size_tokens(),
        VALIDATED_MODEL_MAXIMUM_POSITION_COUNT as usize
    );
    prompt_processing_chunk_sizer.start_prompt_processing_request(0);
    let mut chunk_start_token_position = 0_usize;
    for expected_chunk_size_tokens in [8_192, 4_096, 2_048, 1_024] {
        let chunk_end_token_position_exclusive = prompt_processing_chunk_sizer
            .next_prompt_processing_chunk_end(chunk_start_token_position, 100_000);
        assert_eq!(
            chunk_end_token_position_exclusive - chunk_start_token_position,
            expected_chunk_size_tokens
        );
        prompt_processing_chunk_sizer
            .record_prompt_processing_chunk_elapsed_millis(expected_chunk_size_tokens, 1_000);
        chunk_start_token_position = chunk_end_token_position_exclusive;
    }
}

#[test]
fn should_explore_configured_chunk_sizes_in_non_persisted_optimized_mode() {
    let mut prompt_processing_chunk_sizer =
        optimized_prompt_processing_chunk_sizer(4_096, vec![1_024, 2_048, 4_096])
            .expect("configured optimizer candidates should initialize");

    assert_full_candidate_exploration_for_one_context(&mut prompt_processing_chunk_sizer);
}

#[test]
fn should_explore_configured_chunk_sizes_when_the_persisted_optimizer_is_enabled() {
    let optimizer_state_directory = tempfile::tempdir()
        .expect("the persisted optimizer test should create a temporary state directory");
    let mut prompt_processing_chunk_sizer =
        Qwen3_5PromptProcessingChunkSizer::for_optimized_production_with_persisted_state_and_behavior(
            4_096,
            vec![1_024, 2_048, 4_096],
            optimizer_state_directory.path().to_path_buf(),
            "test-model".to_owned(),
            "test-revision".to_owned(),
            5,
            32_768,
        )
        .expect("the persisted production optimizer should initialize");
    assert_full_candidate_exploration_for_one_context(&mut prompt_processing_chunk_sizer);
}

fn assert_full_candidate_exploration_for_one_context(
    prompt_processing_chunk_sizer: &mut Qwen3_5PromptProcessingChunkSizer,
) {
    prompt_processing_chunk_sizer.start_prompt_processing_request(0);
    let mut chunk_start_token_position = 0_usize;
    for expected_chunk_size_tokens in [4_096, 2_048, 1_024] {
        let chunk_end_token_position_exclusive = prompt_processing_chunk_sizer
            .next_prompt_processing_chunk_end(chunk_start_token_position, 100_000);
        assert_eq!(
            chunk_end_token_position_exclusive - chunk_start_token_position,
            expected_chunk_size_tokens
        );
        prompt_processing_chunk_sizer
            .record_prompt_processing_chunk_elapsed_millis(expected_chunk_size_tokens, 1_000);
        chunk_start_token_position = chunk_end_token_position_exclusive;
    }
}

#[test]
fn should_bound_each_prompt_processing_chunk_by_maximum_size_and_prompt_end() {
    let mut prompt_processing_chunk_sizer =
        optimized_prompt_processing_chunk_sizer(2_048, vec![128, 256, 512, 1_024, 2_048])
            .expect("the explicit maximum prompt-processing chunk size should be valid");

    prompt_processing_chunk_sizer.start_prompt_processing_request(0);

    assert_eq!(
        prompt_processing_chunk_sizer.next_prompt_processing_chunk_end(0, 5_000),
        2_048
    );
    assert_eq!(
        prompt_processing_chunk_sizer.next_prompt_processing_chunk_end(4_096, 4_100),
        4_100
    );
}

#[test]
fn should_reject_a_zero_maximum_prompt_processing_chunk_size() {
    let prompt_processing_chunk_sizer_error =
        optimized_prompt_processing_chunk_sizer(0, vec![1_024])
            .expect_err("the maximum prompt-processing chunk size must be positive");

    assert_eq!(
        prompt_processing_chunk_sizer_error.to_string(),
        "prompt-processing chunk size must be positive"
    );
}

#[test]
fn should_use_the_explicit_optimizer_prompt_processing_chunk_size_maximum() {
    let prompt_processing_chunk_sizer =
        optimized_prompt_processing_chunk_sizer(4_096, vec![1_024, 2_048, 4_096])
            .expect("maximum prompt-processing chunk size should be valid");

    assert_eq!(
        prompt_processing_chunk_sizer.maximum_prompt_processing_chunk_size_tokens(),
        4_096
    );
}

#[test]
fn should_explore_unobserved_candidates_from_largest_to_smallest() {
    let mut prompt_processing_chunk_sizer =
        optimized_prompt_processing_chunk_sizer(2_048, vec![128, 256, 512, 1_024, 2_048])
            .expect("the explicit maximum prompt-processing chunk size should be valid");

    prompt_processing_chunk_sizer.start_prompt_processing_request(0);

    assert_eq!(
        prompt_processing_chunk_sizer.next_prompt_processing_chunk_end(0, 5_000),
        2_048
    );
    prompt_processing_chunk_sizer.record_prompt_processing_chunk_elapsed_millis(2_048, 1_000);
    assert_eq!(
        prompt_processing_chunk_sizer.active_prompt_processing_chunk_size_tokens(),
        2_048,
        "the completed 2,048-token chunk should remain active until the next decision"
    );

    assert_eq!(
        prompt_processing_chunk_sizer.next_prompt_processing_chunk_end(2_048, 5_000),
        3_072
    );
    prompt_processing_chunk_sizer.record_prompt_processing_chunk_elapsed_millis(1_024, 1_000);
    assert_eq!(
        prompt_processing_chunk_sizer.active_prompt_processing_chunk_size_tokens(),
        1_024,
        "the completed chunk should remain active until the next decision"
    );

    assert_eq!(
        prompt_processing_chunk_sizer.next_prompt_processing_chunk_end(3_072, 5_000),
        3_584,
        "the next largest candidate that fits should be selected"
    );
}

#[test]
fn should_not_consume_the_next_optimizer_decision_when_recording_a_completed_chunk() {
    let mut prompt_processing_chunk_sizer =
        optimized_prompt_processing_chunk_sizer(2_048, vec![128, 256, 512, 1_024, 2_048])
            .expect("the explicit maximum prompt-processing chunk size should be valid");

    prompt_processing_chunk_sizer.start_prompt_processing_request(0);

    assert_eq!(
        prompt_processing_chunk_sizer.next_prompt_processing_chunk_end(0, 5_000),
        2_048
    );
    prompt_processing_chunk_sizer.record_prompt_processing_chunk_elapsed_millis(2_048, 1_000);
    assert_eq!(
        prompt_processing_chunk_sizer.active_prompt_processing_chunk_size_tokens(),
        2_048,
        "recording a completed chunk should keep reporting the chunk that actually ran"
    );

    assert_eq!(
        prompt_processing_chunk_sizer.next_prompt_processing_chunk_end(2_048, 5_000),
        3_072
    );
    prompt_processing_chunk_sizer.record_prompt_processing_chunk_elapsed_millis(1_024, 1_000);
    assert_eq!(
        prompt_processing_chunk_sizer.active_prompt_processing_chunk_size_tokens(),
        1_024,
        "recording the completed chunk should not pre-ask the optimizer"
    );

    assert_eq!(
        prompt_processing_chunk_sizer.next_prompt_processing_chunk_end(3_072, 5_000),
        3_584
    );
    prompt_processing_chunk_sizer.record_prompt_processing_chunk_elapsed_millis(512, 1_000);

    assert_eq!(
        prompt_processing_chunk_sizer.active_prompt_processing_chunk_size_tokens(),
        512,
        "recording the completed chunk should not consume the next decision"
    );
}

#[test]
fn should_retain_final_prompt_tail_transitions() {
    let mut prompt_processing_chunk_sizer =
        optimized_prompt_processing_chunk_sizer(2_048, vec![128, 256, 512, 1_024, 2_048])
            .expect("the explicit maximum prompt-processing chunk size should be valid");

    prompt_processing_chunk_sizer.start_prompt_processing_request(0);
    assert_eq!(
        prompt_processing_chunk_sizer.next_prompt_processing_chunk_end(0, 64),
        64
    );
    prompt_processing_chunk_sizer.record_prompt_processing_chunk_elapsed_millis(64, 11_000);

    assert_eq!(
        prompt_processing_chunk_sizer.active_prompt_processing_chunk_size_tokens(),
        128,
        "the minimum requested candidate remains active for a short prompt tail"
    );
}

#[test]
fn should_skip_exploration_candidates_larger_than_the_remaining_prompt() {
    let mut prompt_processing_chunk_sizer =
        optimized_prompt_processing_chunk_sizer(1_024, vec![128, 256, 512, 1_024])
            .expect("the optimizer maximum should be valid");

    prompt_processing_chunk_sizer.start_prompt_processing_request(0);
    assert_eq!(
        prompt_processing_chunk_sizer.next_prompt_processing_chunk_end(0, 700),
        512
    );
    prompt_processing_chunk_sizer.record_prompt_processing_chunk_elapsed_millis(512, 100);
    assert_eq!(
        prompt_processing_chunk_sizer.next_prompt_processing_chunk_end(512, 700),
        640
    );
    prompt_processing_chunk_sizer.record_prompt_processing_chunk_elapsed_millis(128, 100);

    assert_eq!(
        prompt_processing_chunk_sizer.next_prompt_processing_chunk_end(640, 700),
        700,
        "the remaining prompt should execute as a minimum-candidate tail"
    );
}

#[test]
fn should_isolate_execution_modes_and_clear_first_after_restore() {
    let mut prompt_processing_chunk_sizer =
        optimized_prompt_processing_chunk_sizer(1_024, vec![128, 256, 512, 1_024])
            .expect("the optimizer maximum should be valid");
    let text_execution_context = Qwen3_5PrefillExecutionContext::default();
    let visual_execution_context = Qwen3_5PrefillExecutionContext::new(true, false, false, false);

    prompt_processing_chunk_sizer.start_prompt_processing_request(128);
    assert_eq!(
        prompt_processing_chunk_sizer.next_prompt_processing_chunk_end_for_execution_context(
            128,
            10_000,
            text_execution_context,
        ),
        1_152
    );
    prompt_processing_chunk_sizer.record_prompt_processing_chunk_transition(
        1_024,
        500,
        false,
        text_execution_context,
    );
    assert_eq!(
        prompt_processing_chunk_sizer.next_prompt_processing_chunk_end_for_execution_context(
            1_152,
            10_000,
            text_execution_context,
        ),
        2_176,
        "the first-after-restore context should not reuse first-chunk evidence"
    );
    assert_eq!(
        prompt_processing_chunk_sizer.next_prompt_processing_chunk_end_for_execution_context(
            1_152,
            10_000,
            visual_execution_context,
        ),
        2_176,
        "visual prefill should not reuse text observations"
    );
}

#[test]
fn should_enter_a_capacity_reduced_context_after_an_admission_retry() {
    let mut prompt_processing_chunk_sizer =
        optimized_prompt_processing_chunk_sizer(1_024, vec![128, 256, 512, 1_024])
            .expect("the optimizer maximum should be valid");
    let execution_context = Qwen3_5PrefillExecutionContext::default();
    prompt_processing_chunk_sizer.start_prompt_processing_request(0);
    assert_eq!(
        prompt_processing_chunk_sizer.next_prompt_processing_chunk_end_for_execution_context(
            0,
            10_000,
            execution_context,
        ),
        1_024
    );
    prompt_processing_chunk_sizer.record_prompt_processing_chunk_transition(
        512,
        2_000,
        true,
        execution_context,
    );
    assert_eq!(
        prompt_processing_chunk_sizer.next_prompt_processing_chunk_end_for_execution_context(
            512,
            10_000,
            execution_context,
        ),
        1_536,
        "capacity-reduced execution should begin independent largest-first discovery"
    );
}

#[test]
fn should_keep_fixed_prompt_processing_chunk_size_after_recorded_elapsed_time() {
    let mut prompt_processing_chunk_sizer =
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(2_048)
            .expect("fixed prompt-processing chunk size should be valid");

    prompt_processing_chunk_sizer.start_prompt_processing_request(0);

    assert_eq!(
        prompt_processing_chunk_sizer.next_prompt_processing_chunk_end(0, 10_000),
        2_048
    );
    prompt_processing_chunk_sizer.record_prompt_processing_chunk_elapsed_millis(2_048, 10_000);
    assert_eq!(
        prompt_processing_chunk_sizer.active_prompt_processing_chunk_size_tokens(),
        2_048,
        "fixed mode must not adapt after an elapsed-time observation"
    );
    assert_eq!(
        prompt_processing_chunk_sizer.next_prompt_processing_chunk_end(2_048, 10_000),
        4_096,
        "fixed mode must retain its configured size for the next full chunk"
    );
}

#[test]
fn should_use_fixed_ssd_streaming_chunk_size_only_while_experts_are_paged() {
    let mut prompt_processing_chunk_sizer =
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(
            2_048,
            Some(256),
        )
        .expect("fixed complete-resident and SSD streaming sizes should be valid");
    let paged_execution_context = Qwen3_5PrefillExecutionContext::new(false, false, true, false);
    let complete_resident_execution_context =
        Qwen3_5PrefillExecutionContext::new(false, false, false, false);

    prompt_processing_chunk_sizer.start_prompt_processing_request(0);

    assert_eq!(
        prompt_processing_chunk_sizer.next_prompt_processing_chunk_end_for_execution_context(
            0,
            10_000,
            paged_execution_context,
        ),
        256,
        "paged expert residency must use the configured SSD streaming chunk size"
    );
    assert_eq!(
        prompt_processing_chunk_sizer.active_prompt_processing_chunk_size_tokens(),
        256,
        "active fixed size should reflect the SSD streaming selection"
    );
    assert_eq!(
        prompt_processing_chunk_sizer.next_prompt_processing_chunk_end_for_execution_context(
            0,
            10_000,
            complete_resident_execution_context,
        ),
        2_048,
        "complete-resident execution must keep the larger fixed chunk size"
    );
}

#[test]
fn should_coalesce_a_terminal_remainder_between_optimizer_candidates() {
    let mut prompt_processing_chunk_sizer =
        optimized_prompt_processing_chunk_sizer(4_096, vec![512, 1_024, 2_048, 4_096])
            .expect("terminal-tail candidates should be valid");
    let chunk_start_token_position = 26_624;
    let final_prompt_end_token_position_exclusive = 28_511;

    prompt_processing_chunk_sizer.start_prompt_processing_request(chunk_start_token_position);

    assert_eq!(
        prompt_processing_chunk_sizer
            .next_prompt_processing_chunk_end_for_execution_context_with_terminal_coalescing(
                chunk_start_token_position,
                final_prompt_end_token_position_exclusive,
                Qwen3_5PrefillExecutionContext::default(),
                true,
            ),
        final_prompt_end_token_position_exclusive,
        "the exact 1,887-token tail should execute in one forward"
    );
    assert_eq!(
        prompt_processing_chunk_sizer.active_prompt_processing_chunk_size_tokens(),
        2_048,
        "optimizer evidence should retain the smallest candidate containing the tail"
    );
}

#[test]
fn should_execute_a_small_terminal_remainder_exactly_under_its_candidate_label() {
    let mut prompt_processing_chunk_sizer =
        optimized_prompt_processing_chunk_sizer(4_096, vec![512, 1_024, 2_048, 4_096])
            .expect("terminal-tail candidates should be valid");
    let chunk_start_token_position = 10_000;
    let final_prompt_end_token_position_exclusive = 10_351;

    prompt_processing_chunk_sizer.start_prompt_processing_request(chunk_start_token_position);

    assert_eq!(
        prompt_processing_chunk_sizer
            .next_prompt_processing_chunk_end_for_execution_context_with_terminal_coalescing(
                chunk_start_token_position,
                final_prompt_end_token_position_exclusive,
                Qwen3_5PrefillExecutionContext::default(),
                true,
            ),
        final_prompt_end_token_position_exclusive
    );
    assert_eq!(
        prompt_processing_chunk_sizer.active_prompt_processing_chunk_size_tokens(),
        512
    );
}

#[test]
fn should_leave_fixed_terminal_chunk_selection_unchanged() {
    let mut prompt_processing_chunk_sizer =
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(1_024)
            .expect("fixed prefill chunk size should be valid");

    assert_eq!(
        prompt_processing_chunk_sizer
            .next_prompt_processing_chunk_end_for_execution_context_with_terminal_coalescing(
                26_624,
                28_511,
                Qwen3_5PrefillExecutionContext::default(),
                true,
            ),
        27_648,
        "terminal coalescing must not override an explicitly fixed chunk size"
    );
}
