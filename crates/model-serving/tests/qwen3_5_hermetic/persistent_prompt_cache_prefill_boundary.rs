use astronomical_model_serving::persistent_prompt_cache_boundary_completed_prefill_chunck_tokens;

#[test]
fn should_report_every_persistent_prompt_cache_boundary_crossed_by_one_prefill_chunck() {
    for (prefill_chunck_start, prefill_chunck_end, expected_completed_prefill_chunck_tokens) in [
        (0, 128, Vec::new()),
        (0, 2_048, vec![2_048]),
        (0, 4_096, vec![2_048, 4_096]),
        (128, 4_096, vec![1_920, 3_968]),
        (22_528, 26_624, vec![2_048, 4_096]),
        (22_528, 25_000, vec![2_048]),
        (2_048, 2_048, Vec::new()),
        (4_096, 2_048, Vec::new()),
        (usize::MAX - 1_024, usize::MAX, Vec::new()),
    ] {
        assert_eq!(
            expected_completed_prefill_chunck_tokens,
            persistent_prompt_cache_boundary_completed_prefill_chunck_tokens(
                prefill_chunck_start,
                prefill_chunck_end,
            ),
            "unexpected local boundary counts for [{prefill_chunck_start}, {prefill_chunck_end})"
        );
    }
}
