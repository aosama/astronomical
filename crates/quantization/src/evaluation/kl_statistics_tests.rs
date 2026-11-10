use super::KlEvaluationSummary;

#[test]
fn should_average_prompt_means_so_a_long_prompt_does_not_outweigh_a_short_one() {
    let per_prompt_divergences = vec![vec![1.0], vec![0.0, 0.0, 0.0]];

    let summary = KlEvaluationSummary::from_per_prompt_divergences(&per_prompt_divergences)
        .expect("tokens were measured");

    assert_eq!(summary.mean_kl_divergence, 0.5);
    assert_eq!(summary.prompt_count, 2);
    assert_eq!(summary.token_count, 4);
}

#[test]
fn should_take_the_median_and_tail_over_every_token_with_linear_interpolation() {
    // numpy.median([0, 1, 2, 3]) is 1.5; numpy.percentile(..., 95) is 2.85.
    let per_prompt_divergences = vec![vec![3.0, 0.0], vec![2.0, 1.0]];

    let summary = KlEvaluationSummary::from_per_prompt_divergences(&per_prompt_divergences)
        .expect("tokens were measured");

    assert_eq!(summary.median_kl_divergence, 1.5);
    assert!((summary.ninety_fifth_percentile_kl_divergence - 2.85).abs() < 1e-12);
}

#[test]
fn should_summarize_a_single_token_as_itself() {
    let summary = KlEvaluationSummary::from_per_prompt_divergences(&[vec![0.25]])
        .expect("one token was measured");

    assert_eq!(summary.mean_kl_divergence, 0.25);
    assert_eq!(summary.median_kl_divergence, 0.25);
    assert_eq!(summary.ninety_fifth_percentile_kl_divergence, 0.25);
}

#[test]
fn should_skip_a_prompt_with_no_tokens_and_report_nothing_when_all_are_empty() {
    let with_one_empty = KlEvaluationSummary::from_per_prompt_divergences(&[vec![], vec![2.0]])
        .expect("one prompt was measured");

    assert_eq!(with_one_empty.prompt_count, 1);
    assert_eq!(with_one_empty.mean_kl_divergence, 2.0);
    assert!(KlEvaluationSummary::from_per_prompt_divergences(&[vec![], vec![]]).is_none());
    assert!(KlEvaluationSummary::from_per_prompt_divergences(&[]).is_none());
}
