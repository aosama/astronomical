//! Unit tests for temperature lookup and the calibrated softmax.

use super::*;
use crate::test_support::assert_close;

fn published_table() -> TemperatureTable {
    TemperatureTable::from_raw_temperatures(
        KindTemperatures {
            choice: 1.6,
            score: 1.25,
            noul: 2.0,
        },
        &[
            BucketTemperature {
                bucket_key: "choice:3-5".to_owned(),
                temperature: 1.75,
            },
            BucketTemperature {
                bucket_key: "choice:11+".to_owned(),
                temperature: 0.1,
            },
        ],
    )
}

#[test]
fn should_leave_a_temperature_inside_the_applied_range_unchanged() {
    assert_close(clamp_temperature(1.75), 1.75, 0.0);
    assert_close(clamp_temperature(MINIMUM_APPLIED_TEMPERATURE), 0.5, 0.0);
    assert_close(clamp_temperature(MAXIMUM_APPLIED_TEMPERATURE), 5.0, 0.0);
}

#[test]
fn should_confine_a_temperature_outside_the_applied_range() {
    assert_close(clamp_temperature(0.1), MINIMUM_APPLIED_TEMPERATURE, 0.0);
    assert_close(clamp_temperature(-3.0), MINIMUM_APPLIED_TEMPERATURE, 0.0);
    assert_close(clamp_temperature(99.0), MAXIMUM_APPLIED_TEMPERATURE, 0.0);
}

#[test]
fn should_fall_back_to_neutral_for_a_temperature_that_is_not_a_finite_number() {
    assert_close(clamp_temperature(f64::NAN), NEUTRAL_TEMPERATURE, 0.0);
    assert_close(clamp_temperature(f64::INFINITY), NEUTRAL_TEMPERATURE, 0.0);
    assert_close(
        clamp_temperature(f64::NEG_INFINITY),
        NEUTRAL_TEMPERATURE,
        0.0,
    );
}

#[test]
fn should_name_option_count_ranges_at_every_boundary() {
    assert_eq!(option_count_range_name(1), "2");
    assert_eq!(option_count_range_name(2), "2");
    assert_eq!(option_count_range_name(3), "3-5");
    assert_eq!(option_count_range_name(5), "3-5");
    assert_eq!(option_count_range_name(6), "6-10");
    assert_eq!(option_count_range_name(10), "6-10");
    assert_eq!(option_count_range_name(11), "11+");
    assert_eq!(option_count_range_name(255), "11+");
}

#[test]
fn should_prefer_the_option_count_temperature_and_fall_back_to_the_kind() {
    let table = published_table();
    assert_close(
        table.temperature_for(DecisionQuestionKind::Choice, 4),
        1.75,
        0.0,
    );
    assert_close(
        table.temperature_for(DecisionQuestionKind::Choice, 2),
        1.6,
        0.0,
    );
    assert_close(
        table.temperature_for(DecisionQuestionKind::Score, 4),
        1.25,
        0.0,
    );
    assert_close(
        table.temperature_for(DecisionQuestionKind::Noul, 2),
        2.0,
        0.0,
    );
}

#[test]
fn should_clamp_a_sharpening_option_count_temperature() {
    let table = published_table();
    assert_close(
        table.temperature_for(DecisionQuestionKind::Choice, 12),
        MINIMUM_APPLIED_TEMPERATURE,
        0.0,
    );
}

#[test]
fn should_apply_one_temperature_to_every_kind_in_a_uniform_table() {
    let table = TemperatureTable::uniform(2.41);
    assert_close(
        table.temperature_for(DecisionQuestionKind::Choice, 30),
        2.41,
        0.0,
    );
    assert_close(
        table.temperature_for(DecisionQuestionKind::Noul, 2),
        2.41,
        0.0,
    );
}

#[test]
fn should_produce_probabilities_that_sum_to_one() {
    let probabilities = calibrated_probabilities(&[0.5, 3.75, -2.0, 1.125], 1.75);
    let mut probability_total = 0.0;
    for probability in &probabilities {
        probability_total += probability;
    }
    assert_close(probability_total, 1.0, 1e-12);
    assert!(probabilities[1] > probabilities[3]);
}

#[test]
fn should_soften_the_distribution_as_the_temperature_rises() {
    let sharp_probabilities = calibrated_probabilities(&[2.0, 0.0], 1.0);
    let soft_probabilities = calibrated_probabilities(&[2.0, 0.0], 4.0);
    assert!(soft_probabilities[0] < sharp_probabilities[0]);
}

#[test]
fn should_stay_finite_for_logits_that_would_overflow_a_plain_exponential() {
    let probabilities = calibrated_probabilities(&[10_000.0, -10_000.0], 1.0);
    assert_close(probabilities[0], 1.0, 1e-12);
    assert_close(probabilities[1], 0.0, 1e-12);
}

#[test]
fn should_give_a_single_option_all_the_probability() {
    assert_eq!(calibrated_probabilities(&[4.0], 2.0), vec![1.0]);
}

#[test]
fn should_return_no_probabilities_for_no_logits() {
    assert!(calibrated_probabilities(&[], 1.0).is_empty());
}

#[test]
fn should_name_a_bucket_by_kind_and_option_count_range() {
    assert_eq!(
        temperature_bucket_key(DecisionQuestionKind::Choice, 4),
        "choice:3-5"
    );
    assert_eq!(
        temperature_bucket_key(DecisionQuestionKind::Score, 11),
        "score:11+"
    );
    assert_eq!(
        temperature_bucket_key(DecisionQuestionKind::Noul, 2),
        "noul:2"
    );
}

#[test]
fn should_replace_non_finite_raw_temperatures_when_a_table_is_built() {
    let table = TemperatureTable::from_raw_temperatures(
        KindTemperatures {
            choice: f64::NAN,
            score: f64::INFINITY,
            noul: 2.0,
        },
        &[BucketTemperature {
            bucket_key: "noul:2".to_owned(),
            temperature: f64::NAN,
        }],
    );
    assert_close(
        table.temperature_for(DecisionQuestionKind::Choice, 2),
        NEUTRAL_TEMPERATURE,
        0.0,
    );
    assert_close(
        table.temperature_for(DecisionQuestionKind::Score, 2),
        NEUTRAL_TEMPERATURE,
        0.0,
    );
    assert_close(
        table.temperature_for(DecisionQuestionKind::Noul, 2),
        NEUTRAL_TEMPERATURE,
        0.0,
    );
}

#[test]
fn should_return_finite_probabilities_for_a_zero_or_non_finite_temperature() {
    for unusable_temperature in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let probabilities = calibrated_probabilities(&[1.0, 0.0], unusable_temperature);
        assert!(probabilities[0].is_finite() && probabilities[1].is_finite());
        assert_close(probabilities[0] + probabilities[1], 1.0, 1e-12);
    }
}
