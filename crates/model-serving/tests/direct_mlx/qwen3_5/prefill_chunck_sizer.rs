use astronomical_model_serving::{Qwen3_5PrefillChunckSizer, Qwen3_5PrefillExecutionContext};

fn optimized_prefill_chunck_sizer(
    maximum_prefill_chunck_tokens: u32,
    optimizer_prefill_chunck_token_candidates: Vec<u32>,
) -> Result<Qwen3_5PrefillChunckSizer, astronomical_model_serving::Qwen3_5PrefillChunckSizerError> {
    Qwen3_5PrefillChunckSizer::for_optimized_with_behavior(
        maximum_prefill_chunck_tokens,
        optimizer_prefill_chunck_token_candidates,
        5,
        32_768,
    )
}

#[test]
fn should_use_only_configured_candidates_below_the_validated_model_context_maximum() {
    const VALIDATED_MODEL_MAXIMUM_POSITION_COUNT: u32 = 8_193;
    let mut prefill_chunck_sizer = optimized_prefill_chunck_sizer(
        VALIDATED_MODEL_MAXIMUM_POSITION_COUNT,
        vec![1_024, 2_048, 4_096, 8_192, 16_384],
    )
    .expect("configured candidates below the model maximum should initialize");

    assert_eq!(
        prefill_chunck_sizer.prefill_chunck_tokens(),
        VALIDATED_MODEL_MAXIMUM_POSITION_COUNT as usize
    );
    prefill_chunck_sizer.start_prompt_processing_request(0);
    let mut prefill_chunck_start = 0_usize;
    for expected_prefill_chunck_tokens in [8_192, 4_096, 2_048, 1_024] {
        let prefill_chunck_end =
            prefill_chunck_sizer.next_prefill_chunck_end(prefill_chunck_start, 100_000);
        assert_eq!(
            prefill_chunck_end - prefill_chunck_start,
            expected_prefill_chunck_tokens
        );
        prefill_chunck_sizer
            .record_prefill_chunck_elapsed_millis(expected_prefill_chunck_tokens, 1_000);
        prefill_chunck_start = prefill_chunck_end;
    }
}

#[test]
fn should_explore_configured_prefill_chunck_tokens_in_non_persisted_optimized_mode() {
    let mut prefill_chunck_sizer = optimized_prefill_chunck_sizer(4_096, vec![1_024, 2_048, 4_096])
        .expect("configured optimizer candidates should initialize");

    assert_full_candidate_exploration_for_one_context(&mut prefill_chunck_sizer);
}

#[test]
fn should_explore_configured_prefill_chunck_tokens_when_the_persisted_optimizer_is_enabled() {
    let optimizer_state_directory = tempfile::tempdir()
        .expect("the persisted optimizer test should create a temporary state directory");
    let mut prefill_chunck_sizer =
        Qwen3_5PrefillChunckSizer::for_optimized_production_with_persisted_state_and_behavior(
            4_096,
            vec![1_024, 2_048, 4_096],
            optimizer_state_directory.path().to_path_buf(),
            "test-model".to_owned(),
            "test-revision".to_owned(),
            5,
            32_768,
        )
        .expect("the persisted production optimizer should initialize");
    assert_full_candidate_exploration_for_one_context(&mut prefill_chunck_sizer);
}

fn assert_full_candidate_exploration_for_one_context(
    prefill_chunck_sizer: &mut Qwen3_5PrefillChunckSizer,
) {
    prefill_chunck_sizer.start_prompt_processing_request(0);
    let mut prefill_chunck_start = 0_usize;
    for expected_prefill_chunck_tokens in [4_096, 2_048, 1_024] {
        let prefill_chunck_end =
            prefill_chunck_sizer.next_prefill_chunck_end(prefill_chunck_start, 100_000);
        assert_eq!(
            prefill_chunck_end - prefill_chunck_start,
            expected_prefill_chunck_tokens
        );
        prefill_chunck_sizer
            .record_prefill_chunck_elapsed_millis(expected_prefill_chunck_tokens, 1_000);
        prefill_chunck_start = prefill_chunck_end;
    }
}

#[test]
fn should_bound_each_prompt_processing_chunck_by_prefill_chunck_tokens_and_prompt_end() {
    let mut prefill_chunck_sizer =
        optimized_prefill_chunck_sizer(2_048, vec![128, 256, 512, 1_024, 2_048])
            .expect("the explicit maximum prefill_chunck_tokens should be valid");

    prefill_chunck_sizer.start_prompt_processing_request(0);

    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end(0, 5_000),
        2_048
    );
    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end(4_096, 4_100),
        4_100
    );
}

#[test]
fn should_reject_zero_prefill_chunck_tokens() {
    let prefill_chunck_sizer_error = optimized_prefill_chunck_sizer(0, vec![1_024])
        .expect_err("prefill_chunck_tokens must contain at least one token");

    assert_eq!(
        prefill_chunck_sizer_error.to_string(),
        "prefill_chunck_tokens must be positive"
    );
}

#[test]
fn should_use_the_explicit_optimizer_prefill_chunck_tokens_maximum() {
    let prefill_chunck_sizer = optimized_prefill_chunck_sizer(4_096, vec![1_024, 2_048, 4_096])
        .expect("maximum prefill_chunck_tokens should be valid");

    assert_eq!(prefill_chunck_sizer.prefill_chunck_tokens(), 4_096);
}

#[test]
fn should_explore_unobserved_candidates_from_largest_to_smallest() {
    let mut prefill_chunck_sizer =
        optimized_prefill_chunck_sizer(2_048, vec![128, 256, 512, 1_024, 2_048])
            .expect("the explicit maximum prefill_chunck_tokens should be valid");

    prefill_chunck_sizer.start_prompt_processing_request(0);

    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end(0, 5_000),
        2_048
    );
    prefill_chunck_sizer.record_prefill_chunck_elapsed_millis(2_048, 1_000);
    assert_eq!(
        prefill_chunck_sizer.active_prefill_chunck_tokens(),
        2_048,
        "the completed 2,048-token chunk should remain active until the next decision"
    );

    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end(2_048, 5_000),
        3_072
    );
    prefill_chunck_sizer.record_prefill_chunck_elapsed_millis(1_024, 1_000);
    assert_eq!(
        prefill_chunck_sizer.active_prefill_chunck_tokens(),
        1_024,
        "the completed chunk should remain active until the next decision"
    );

    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end(3_072, 5_000),
        3_584,
        "the next largest candidate that fits should be selected"
    );
}

#[test]
fn should_not_consume_the_next_optimizer_decision_when_recording_a_completed_prefill_chunck() {
    let mut prefill_chunck_sizer =
        optimized_prefill_chunck_sizer(2_048, vec![128, 256, 512, 1_024, 2_048])
            .expect("the explicit maximum prefill_chunck_tokens should be valid");

    prefill_chunck_sizer.start_prompt_processing_request(0);

    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end(0, 5_000),
        2_048
    );
    prefill_chunck_sizer.record_prefill_chunck_elapsed_millis(2_048, 1_000);
    assert_eq!(
        prefill_chunck_sizer.active_prefill_chunck_tokens(),
        2_048,
        "recording a completed chunk should keep reporting the chunk that actually ran"
    );

    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end(2_048, 5_000),
        3_072
    );
    prefill_chunck_sizer.record_prefill_chunck_elapsed_millis(1_024, 1_000);
    assert_eq!(
        prefill_chunck_sizer.active_prefill_chunck_tokens(),
        1_024,
        "recording the completed chunk should not pre-ask the optimizer"
    );

    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end(3_072, 5_000),
        3_584
    );
    prefill_chunck_sizer.record_prefill_chunck_elapsed_millis(512, 1_000);

    assert_eq!(
        prefill_chunck_sizer.active_prefill_chunck_tokens(),
        512,
        "recording the completed chunk should not consume the next decision"
    );
}

#[test]
fn should_retain_final_prompt_tail_transitions() {
    let mut prefill_chunck_sizer =
        optimized_prefill_chunck_sizer(2_048, vec![128, 256, 512, 1_024, 2_048])
            .expect("the explicit maximum prefill_chunck_tokens should be valid");

    prefill_chunck_sizer.start_prompt_processing_request(0);
    assert_eq!(prefill_chunck_sizer.next_prefill_chunck_end(0, 64), 64);
    prefill_chunck_sizer.record_prefill_chunck_elapsed_millis(64, 11_000);

    assert_eq!(
        prefill_chunck_sizer.active_prefill_chunck_tokens(),
        128,
        "the minimum requested candidate remains active for a short prompt tail"
    );
}

#[test]
fn should_skip_exploration_candidates_larger_than_the_remaining_prompt() {
    let mut prefill_chunck_sizer =
        optimized_prefill_chunck_sizer(1_024, vec![128, 256, 512, 1_024])
            .expect("the optimizer maximum should be valid");

    prefill_chunck_sizer.start_prompt_processing_request(0);
    assert_eq!(prefill_chunck_sizer.next_prefill_chunck_end(0, 700), 512);
    prefill_chunck_sizer.record_prefill_chunck_elapsed_millis(512, 100);
    assert_eq!(prefill_chunck_sizer.next_prefill_chunck_end(512, 700), 640);
    prefill_chunck_sizer.record_prefill_chunck_elapsed_millis(128, 100);

    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end(640, 700),
        700,
        "the remaining prompt should execute as a minimum-candidate tail"
    );
}

#[test]
fn should_isolate_execution_modes_and_clear_first_after_restore() {
    let mut prefill_chunck_sizer =
        optimized_prefill_chunck_sizer(1_024, vec![128, 256, 512, 1_024])
            .expect("the optimizer maximum should be valid");
    let text_execution_context = Qwen3_5PrefillExecutionContext::default();
    let visual_execution_context = Qwen3_5PrefillExecutionContext::new(true, false, false, false);

    prefill_chunck_sizer.start_prompt_processing_request(128);
    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end_for_execution_context(
            128,
            10_000,
            text_execution_context,
        ),
        1_152
    );
    prefill_chunck_sizer.record_prefill_chunck_transition(
        1_024,
        500,
        false,
        text_execution_context,
    );
    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end_for_execution_context(
            1_152,
            10_000,
            text_execution_context,
        ),
        2_176,
        "the first-after-restore context should not reuse first-chunk evidence"
    );
    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end_for_execution_context(
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
    let mut prefill_chunck_sizer =
        optimized_prefill_chunck_sizer(1_024, vec![128, 256, 512, 1_024])
            .expect("the optimizer maximum should be valid");
    let execution_context = Qwen3_5PrefillExecutionContext::default();
    prefill_chunck_sizer.start_prompt_processing_request(0);
    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end_for_execution_context(
            0,
            10_000,
            execution_context,
        ),
        1_024
    );
    prefill_chunck_sizer.record_prefill_chunck_transition(512, 2_000, true, execution_context);
    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end_for_execution_context(
            512,
            10_000,
            execution_context,
        ),
        1_536,
        "capacity-reduced execution should begin independent largest-first discovery"
    );
}

#[test]
fn should_keep_fixed_prefill_chunck_tokens_after_recorded_elapsed_time() {
    let mut prefill_chunck_sizer =
        Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens(2_048)
            .expect("fixed prefill_chunck_tokens should be valid");

    prefill_chunck_sizer.start_prompt_processing_request(0);

    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end(0, 10_000),
        2_048
    );
    prefill_chunck_sizer.record_prefill_chunck_elapsed_millis(2_048, 10_000);
    assert_eq!(
        prefill_chunck_sizer.active_prefill_chunck_tokens(),
        2_048,
        "fixed mode must not adapt after an elapsed-time observation"
    );
    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end(2_048, 10_000),
        4_096,
        "fixed mode must retain its configured size for the next full chunk"
    );
}

#[test]
fn should_use_fixed_ssd_streaming_prefill_chunck_tokens_only_while_experts_are_paged() {
    let mut prefill_chunck_sizer =
        Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens_with_ssd_streaming(
            2_048,
            Some(256),
        )
        .expect("fixed complete-resident and SSD streaming sizes should be valid");
    let paged_execution_context = Qwen3_5PrefillExecutionContext::new(false, false, true, false);
    let complete_resident_execution_context =
        Qwen3_5PrefillExecutionContext::new(false, false, false, false);

    prefill_chunck_sizer.start_prompt_processing_request(0);

    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end_for_execution_context(
            0,
            10_000,
            paged_execution_context,
        ),
        256,
        "paged expert residency must use the configured SSD streaming chunk size"
    );
    assert_eq!(
        prefill_chunck_sizer.active_prefill_chunck_tokens(),
        256,
        "active fixed size should reflect the SSD streaming selection"
    );
    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end_for_execution_context(
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
    let mut prefill_chunck_sizer =
        optimized_prefill_chunck_sizer(4_096, vec![512, 1_024, 2_048, 4_096])
            .expect("terminal-tail candidates should be valid");
    let prefill_cursor = 26_624;
    let final_prompt_index = 28_511;

    prefill_chunck_sizer.start_prompt_processing_request(prefill_cursor);

    assert_eq!(
        prefill_chunck_sizer
            .next_prefill_chunck_end_for_execution_context_with_terminal_coalescing(
                prefill_cursor,
                final_prompt_index,
                Qwen3_5PrefillExecutionContext::default(),
                true,
            ),
        final_prompt_index,
        "the exact 1,887-token tail should execute in one forward"
    );
    assert_eq!(
        prefill_chunck_sizer.active_prefill_chunck_tokens(),
        2_048,
        "optimizer evidence should retain the smallest candidate containing the tail"
    );
}

#[test]
fn should_execute_a_small_terminal_remainder_exactly_under_its_candidate_label() {
    let mut prefill_chunck_sizer =
        optimized_prefill_chunck_sizer(4_096, vec![512, 1_024, 2_048, 4_096])
            .expect("terminal-tail candidates should be valid");
    let prefill_cursor = 10_000;
    let final_prompt_index = 10_351;

    prefill_chunck_sizer.start_prompt_processing_request(prefill_cursor);

    assert_eq!(
        prefill_chunck_sizer
            .next_prefill_chunck_end_for_execution_context_with_terminal_coalescing(
                prefill_cursor,
                final_prompt_index,
                Qwen3_5PrefillExecutionContext::default(),
                true,
            ),
        final_prompt_index
    );
    assert_eq!(prefill_chunck_sizer.active_prefill_chunck_tokens(), 512);
}

#[test]
fn should_leave_fixed_terminal_chunk_selection_unchanged() {
    let mut prefill_chunck_sizer =
        Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens(1_024)
            .expect("fixed prefill chunk size should be valid");

    assert_eq!(
        prefill_chunck_sizer
            .next_prefill_chunck_end_for_execution_context_with_terminal_coalescing(
                26_624,
                28_511,
                Qwen3_5PrefillExecutionContext::default(),
                true,
            ),
        27_648,
        "terminal coalescing must not override an explicitly fixed chunk size"
    );
}
