use astronomical_model_serving::Qwen3_5MoEPrefillChunckSizer;

#[test]
fn should_include_the_exact_validated_model_context_maximum_in_production_candidates() {
    const VALIDATED_MODEL_MAXIMUM_POSITION_COUNT: u32 = 8_193;
    let mut prefill_chunck_sizer =
        Qwen3_5MoEPrefillChunckSizer::production(VALIDATED_MODEL_MAXIMUM_POSITION_COUNT)
            .expect("the validated model context maximum should configure the optimizer");

    assert_eq!(
        prefill_chunck_sizer.prefill_chunck_tokens(),
        VALIDATED_MODEL_MAXIMUM_POSITION_COUNT as usize
    );
    for _observation_round in 1..=3 {
        prefill_chunck_sizer.start_prompt_processing_request(0);
        let mut prefill_chunck_start = 0_usize;
        for expected_prefill_chunck_tokens in [128, 256, 512, 1_024, 2_048, 4_096, 8_192, 8_193] {
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
}

#[test]
fn should_explore_128_through_4096_prefill_chunck_tokens_in_non_persisted_optimized_mode() {
    let mut prefill_chunck_sizer = Qwen3_5MoEPrefillChunckSizer::production(4_096)
        .expect("the validated model context maximum should configure the optimizer");

    assert_full_candidate_exploration_for_one_context(&mut prefill_chunck_sizer);
}

#[test]
fn should_explore_128_through_4096_prefill_chunck_tokens_when_the_persisted_optimizer_is_enabled() {
    let optimizer_state_directory = tempfile::tempdir()
        .expect("the persisted optimizer test should create a temporary state directory");
    let mut prefill_chunck_sizer =
        Qwen3_5MoEPrefillChunckSizer::for_optimized_production_with_persisted_state(
            4_096,
            optimizer_state_directory.path().to_path_buf(),
            "test-model".to_owned(),
            "test-revision".to_owned(),
        )
        .expect("the persisted production optimizer should initialize");
    assert_full_candidate_exploration_for_one_context(&mut prefill_chunck_sizer);
}

fn assert_full_candidate_exploration_for_one_context(
    prefill_chunck_sizer: &mut Qwen3_5MoEPrefillChunckSizer,
) {
    for _observation_round in 1..=3 {
        prefill_chunck_sizer.start_prompt_processing_request(0);
        let mut prefill_chunck_start = 0_usize;
        for expected_prefill_chunck_tokens in [128, 256, 512, 1_024, 2_048, 4_096] {
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
}

#[test]
fn should_bound_each_prompt_processing_chunck_by_prefill_chunck_tokens_and_prompt_end() {
    let mut prefill_chunck_sizer =
        Qwen3_5MoEPrefillChunckSizer::for_optimized_with_maximum_prefill_chunck_tokens(2_048)
            .expect("the explicit maximum prefill_chunck_tokens should be valid");

    prefill_chunck_sizer.start_prompt_processing_request(0);

    assert_eq!(prefill_chunck_sizer.next_prefill_chunck_end(0, 5_000), 128);
    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end(4_096, 4_100),
        4_100
    );
}

#[test]
fn should_reject_zero_prefill_chunck_tokens() {
    let prefill_chunck_sizer_error =
        Qwen3_5MoEPrefillChunckSizer::for_optimized_with_maximum_prefill_chunck_tokens(0)
            .expect_err("prefill_chunck_tokens must contain at least one token");

    assert_eq!(
        prefill_chunck_sizer_error.to_string(),
        "prefill_chunck_tokens must be positive"
    );
}

#[test]
fn should_use_the_explicit_optimizer_prefill_chunck_tokens_maximum() {
    let prefill_chunck_sizer =
        Qwen3_5MoEPrefillChunckSizer::for_optimized_with_maximum_prefill_chunck_tokens(4_096)
            .expect("maximum prefill_chunck_tokens should be valid");

    assert_eq!(prefill_chunck_sizer.prefill_chunck_tokens(), 4_096);
}

#[test]
fn should_interleave_candidates_before_any_candidate_reaches_three_observations() {
    let mut prefill_chunck_sizer =
        Qwen3_5MoEPrefillChunckSizer::for_optimized_with_maximum_prefill_chunck_tokens(2_048)
            .expect("the explicit maximum prefill_chunck_tokens should be valid");

    prefill_chunck_sizer.start_prompt_processing_request(0);

    assert_eq!(prefill_chunck_sizer.next_prefill_chunck_end(0, 5_000), 128);
    prefill_chunck_sizer.record_prefill_chunck_elapsed_millis(128, 1_000);
    assert_eq!(
        prefill_chunck_sizer.active_prefill_chunck_tokens(),
        128,
        "the completed 128-token chunk should remain active until the next decision"
    );

    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end(128, 5_000),
        384
    );
    prefill_chunck_sizer.record_prefill_chunck_elapsed_millis(256, 1_000);
    assert_eq!(
        prefill_chunck_sizer.active_prefill_chunck_tokens(),
        256,
        "the completed 256-token chunk should remain active until the next decision"
    );

    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end(384, 5_000),
        896,
        "the first exploration round should continue with 512 tokens"
    );
}

#[test]
fn should_not_consume_the_next_optimizer_decision_when_recording_a_completed_prefill_chunck() {
    let mut prefill_chunck_sizer =
        Qwen3_5MoEPrefillChunckSizer::for_optimized_with_maximum_prefill_chunck_tokens(2_048)
            .expect("the explicit maximum prefill_chunck_tokens should be valid");

    prefill_chunck_sizer.start_prompt_processing_request(0);

    assert_eq!(prefill_chunck_sizer.next_prefill_chunck_end(0, 5_000), 128);
    prefill_chunck_sizer.record_prefill_chunck_elapsed_millis(128, 1_000);
    assert_eq!(
        prefill_chunck_sizer.active_prefill_chunck_tokens(),
        128,
        "recording a completed chunk should keep reporting the chunk that actually ran"
    );

    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end(128, 5_000),
        384
    );
    prefill_chunck_sizer.record_prefill_chunck_elapsed_millis(256, 1_000);
    assert_eq!(
        prefill_chunck_sizer.active_prefill_chunck_tokens(),
        256,
        "recording the 256-token chunk should not pre-ask the optimizer"
    );

    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end(384, 5_000),
        896
    );
    prefill_chunck_sizer.record_prefill_chunck_elapsed_millis(512, 1_000);

    assert_eq!(
        prefill_chunck_sizer.active_prefill_chunck_tokens(),
        512,
        "recording the 512-token chunk should not consume the next decision"
    );
    assert_eq!(
        prefill_chunck_sizer.next_prefill_chunck_end(896, 5_000),
        1_920,
        "the next real chunk should be the first 1,024-token exploration chunk"
    );
}

#[test]
fn should_ignore_final_partial_prefill_chuncks_when_optimizing_future_sizes() {
    let mut prefill_chunck_sizer =
        Qwen3_5MoEPrefillChunckSizer::for_optimized_with_maximum_prefill_chunck_tokens(2_048)
            .expect("the explicit maximum prefill_chunck_tokens should be valid");

    prefill_chunck_sizer.start_prompt_processing_request(0);
    assert_eq!(prefill_chunck_sizer.next_prefill_chunck_end(0, 64), 64);
    prefill_chunck_sizer.record_prefill_chunck_elapsed_millis(64, 11_000);

    assert_eq!(
        prefill_chunck_sizer.active_prefill_chunck_tokens(),
        128,
        "a partial chunk should not count as an observation"
    );
}

#[test]
fn should_keep_fixed_prefill_chunck_tokens_after_recorded_elapsed_time() {
    let mut prefill_chunck_sizer =
        Qwen3_5MoEPrefillChunckSizer::for_fixed_prefill_chunck_tokens(2_048)
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
