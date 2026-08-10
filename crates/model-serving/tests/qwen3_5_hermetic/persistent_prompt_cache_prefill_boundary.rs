use astronomical_model_serving::persistent_prompt_cache_boundary_completed_prefill_chunck_tokens;

#[test]
fn should_report_every_persistent_prompt_cache_boundary_crossed_by_one_prefill_chunck() {
    for (
        prefill_chunck_start,
        prefill_chunck_end,
        persistent_prompt_cache_block_token_count,
        expected_completed_prefill_chunck_tokens,
    ) in [
        (0, 128, 2_048, Vec::new()),
        (0, 2_048, 2_048, vec![2_048]),
        (0, 4_096, 2_048, vec![2_048, 4_096]),
        (128, 4_096, 2_048, vec![1_920, 3_968]),
        (22_528, 26_624, 2_048, vec![2_048, 4_096]),
        (22_528, 25_000, 2_048, vec![2_048]),
        (0, 1_536, 512, vec![512, 1_024, 1_536]),
        (2_048, 2_048, 2_048, Vec::new()),
        (4_096, 2_048, 2_048, Vec::new()),
        (0, 4_096, 0, Vec::new()),
        (usize::MAX - 1_024, usize::MAX, 2_048, Vec::new()),
    ] {
        assert_eq!(
            expected_completed_prefill_chunck_tokens,
            persistent_prompt_cache_boundary_completed_prefill_chunck_tokens(
                prefill_chunck_start,
                prefill_chunck_end,
                persistent_prompt_cache_block_token_count,
            ),
            "unexpected local boundary counts for [{prefill_chunck_start}, {prefill_chunck_end})"
        );
    }
}
