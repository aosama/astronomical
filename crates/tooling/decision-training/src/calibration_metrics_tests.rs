//! Unit tests for the quality numbers.

use decision_contract::DecisionQuestionKind;
use decision_contract::calibration_file::KindCalibration;

use super::*;
use crate::test_support::{
    BEST_TEMPERATURE_FOR_GAP_THREE, CALIBRATED_SEVENTY_PERCENT_GAP, assert_close, borrowed,
    noul_questions, scored_question,
};

fn single_temperature(temperature: f64) -> KindCalibration {
    KindCalibration {
        temperature,
        range_temperatures: Vec::new(),
        fitted_question_count: 0,
    }
}

#[test]
fn should_score_a_calibrated_model_with_near_zero_calibration_error() {
    let calibrated_questions = noul_questions(100, CALIBRATED_SEVENTY_PERCENT_GAP, 7, 0);
    let metrics = calibration_metrics(
        &borrowed(&calibrated_questions),
        ScoringTemperature::Shipped,
    );
    assert_eq!(metrics.question_count, 100);
    assert_close(metrics.accuracy(), 0.7, 1e-12);
    assert_close(metrics.expected_calibration_error, 0.0, 1e-6);
    // -(0.7 ln 0.7 + 0.3 ln 0.3), the entropy of a 70/30 coin.
    assert_close(metrics.mean_negative_log_likelihood, 0.610_864_2, 1e-6);
}

#[test]
fn should_show_an_overconfident_model_improving_under_its_best_temperature() {
    let overconfident_questions = noul_questions(100, 3.0, 7, 0);
    let questions = borrowed(&overconfident_questions);
    let best_temperature = single_temperature(BEST_TEMPERATURE_FOR_GAP_THREE);
    let shipped_metrics = calibration_metrics(&questions, ScoringTemperature::Shipped);
    let calibrated_metrics = calibration_metrics(
        &questions,
        ScoringTemperature::Calibrated(&best_temperature),
    );
    assert!(
        calibrated_metrics.mean_negative_log_likelihood
            < shipped_metrics.mean_negative_log_likelihood
    );
    assert!(calibrated_metrics.mean_brier_score < shipped_metrics.mean_brier_score);
    assert!(
        calibrated_metrics.expected_calibration_error < shipped_metrics.expected_calibration_error
    );
    // A temperature never changes which answer is most probable.
    assert_eq!(
        calibrated_metrics.correct_count,
        shipped_metrics.correct_count
    );
}

#[test]
fn should_score_a_certain_right_answer_as_perfect() {
    let certain_question = scored_question(DecisionQuestionKind::Choice, &[50.0, 0.0, 0.0], 0, 1);
    let metrics = calibration_metrics(&[&certain_question], ScoringTemperature::Shipped);
    assert_eq!(metrics.correct_count, 1);
    assert_close(metrics.mean_negative_log_likelihood, 0.0, 1e-12);
    assert_close(metrics.mean_brier_score, 0.0, 1e-12);
    assert_close(metrics.expected_calibration_error, 0.0, 1e-12);
}

#[test]
fn should_stay_finite_for_a_certain_wrong_answer() {
    let certainly_wrong = scored_question(DecisionQuestionKind::Noul, &[0.0, 2_000.0], 0, 1);
    let metrics = calibration_metrics(&[&certainly_wrong], ScoringTemperature::Shipped);
    assert!(metrics.mean_negative_log_likelihood.is_finite());
    assert_close(metrics.mean_negative_log_likelihood, 2_000.0, 1e-9);
    assert_close(metrics.mean_brier_score, 2.0, 1e-12);
    assert_close(metrics.expected_calibration_error, 1.0, 1e-12);
}

#[test]
fn should_score_at_each_question_s_own_shipped_temperature() {
    let mut warm_question = scored_question(DecisionQuestionKind::Noul, &[0.0, 2.0], 1, 1);
    warm_question.shipped_temperature = 2.0;
    let at_shipped = calibration_metrics(&[&warm_question], ScoringTemperature::Shipped);
    let at_two = calibration_metrics(
        &[&warm_question],
        ScoringTemperature::Calibrated(&single_temperature(2.0)),
    );
    assert_close(
        at_shipped.mean_negative_log_likelihood,
        at_two.mean_negative_log_likelihood,
        1e-15,
    );
}

#[test]
fn should_clamp_a_calibrated_temperature_as_engines_do() {
    let noul_question = scored_question(DecisionQuestionKind::Noul, &[0.0, 2.0], 1, 1);
    let below_range = calibration_metrics(
        &[&noul_question],
        ScoringTemperature::Calibrated(&single_temperature(0.01)),
    );
    let at_floor = calibration_metrics(
        &[&noul_question],
        ScoringTemperature::Calibrated(&single_temperature(0.5)),
    );
    assert_close(
        below_range.mean_negative_log_likelihood,
        at_floor.mean_negative_log_likelihood,
        1e-15,
    );
}

#[test]
fn should_report_zeros_for_no_questions() {
    let metrics = calibration_metrics(&[], ScoringTemperature::Shipped);
    assert_eq!(metrics, CalibrationMetrics::default());
    assert_close(metrics.accuracy(), 0.0, 0.0);
}

#[test]
fn should_put_a_top_probability_of_exactly_one_in_the_last_bin() {
    assert_eq!(confidence_bin_index(1.0), CALIBRATION_ERROR_BIN_COUNT - 1);
    assert_eq!(confidence_bin_index(0.0), 0);
}
