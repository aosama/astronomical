use super::{engine_misses_by_layer, furthest_layers, report_engine_agreement};
use crate::replay_expert_misses::replay_expert_misses_tests::saved_case;

#[test]
fn should_add_each_layer_s_misses_over_every_token() {
    let engine_miss_counts_by_token = vec![vec![1, 0, 2], vec![0, 3], vec![1, 1, 1]];

    assert_eq!(
        engine_misses_by_layer(&engine_miss_counts_by_token),
        vec![2, 4, 3]
    );
}

#[test]
fn should_count_no_layers_for_no_tokens() {
    assert!(engine_misses_by_layer(&[]).is_empty());
}

#[test]
fn should_name_the_layers_that_differ_most_first_and_the_lowest_among_equals() {
    let furthest = furthest_layers(&[10, 4, 0, 7], &[2, 4, 3, 10], 3);

    assert_eq!(furthest, vec![(0, 10, 2), (2, 0, 3), (3, 7, 10)]);
}

#[test]
fn should_count_a_layer_only_one_side_has_as_missing_nothing_on_the_other() {
    let furthest = furthest_layers(&[5], &[5, 0, 4], 1);

    assert_eq!(furthest, vec![(2, 0, 4)]);
}

#[test]
fn should_report_nothing_for_a_recording_without_the_engine_s_misses() {
    assert!(report_engine_agreement(&saved_case(Vec::new())).is_empty());
}

#[test]
fn should_say_the_lab_stands_for_an_engine_that_missed_what_the_lab_misses() {
    let mut saved_replay_case = saved_case(vec![vec![0, 1, 2, 3], vec![0, 1], Vec::new()]);
    let engine_division =
        crate::replay_expert_misses::capacity_divisions::divided_like_todays_engine(
            &saved_replay_case,
        );
    let mut keeping_rule = crate::replay_expert_misses::todays_engine_rule(&saved_replay_case);
    let lab_outcome =
        memory_policy::astro_expert_cache_model::replay(&engine_division, keeping_rule.as_mut());
    // An engine whose misses fall in the same layers as the lab's.
    let mut engine_miss_counts_by_token = vec![Vec::new(); lab_outcome.reply_token_count];
    engine_miss_counts_by_token[0] = lab_outcome
        .miss_count_by_layer
        .iter()
        .map(|&layer_misses| u32::try_from(layer_misses).expect("fits u32"))
        .collect();
    saved_replay_case.engine_miss_counts_by_token = engine_miss_counts_by_token;

    let report_lines = report_engine_agreement(&saved_replay_case);

    assert!(
        report_lines[0].contains("lab_stands_for_engine=true"),
        "{}",
        report_lines[0]
    );
}

#[test]
fn should_say_the_lab_does_not_stand_for_an_engine_that_missed_far_more() {
    let mut saved_replay_case = saved_case(vec![vec![0, 1, 2, 3], vec![0, 1], Vec::new()]);
    saved_replay_case.engine_miss_counts_by_token = vec![vec![0, 40, 40]; 3];

    let report_lines = report_engine_agreement(&saved_replay_case);

    assert!(
        report_lines[0].contains("lab_stands_for_engine=false"),
        "{}",
        report_lines[0]
    );
    assert!(report_lines[1].contains("layer=1") || report_lines[1].contains("layer=2"));
}
