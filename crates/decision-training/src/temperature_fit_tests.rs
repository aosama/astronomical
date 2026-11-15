//! Unit tests for fitting one temperature.

use super::*;
use crate::test_support::{
    BEST_TEMPERATURE_FOR_GAP_THREE, CALIBRATED_SEVENTY_PERCENT_GAP, assert_close, borrowed,
    noul_questions,
};

#[test]
fn should_soften_an_overconfident_model_to_the_temperature_that_matches_its_accuracy() {
    let overconfident_questions = noul_questions(100, 3.0, 7, 0);
    let fitted_temperature = fit_temperature(&borrowed(&overconfident_questions)).unwrap();
    assert_close(fitted_temperature, BEST_TEMPERATURE_FOR_GAP_THREE, 1e-4);
}

#[test]
fn should_leave_an_already_calibrated_model_at_temperature_one() {
    let calibrated_questions = noul_questions(100, CALIBRATED_SEVENTY_PERCENT_GAP, 7, 0);
    let fitted_temperature = fit_temperature(&borrowed(&calibrated_questions)).unwrap();
    assert_close(fitted_temperature, 1.0, 1e-4);
}

#[test]
fn should_stop_at_the_highest_applied_temperature_for_a_wildly_overconfident_model() {
    // Right half the time while scoring a gap of 30: the best temperature is
    // infinite, and the most the engines apply is 5.
    let coin_flip_questions = noul_questions(100, 30.0, 5, 0);
    let fitted_temperature = fit_temperature(&borrowed(&coin_flip_questions)).unwrap();
    assert_close(fitted_temperature, MAXIMUM_APPLIED_TEMPERATURE, 1e-6);
}

#[test]
fn should_stop_at_the_lowest_applied_temperature_for_a_timid_model() {
    // Always right but barely leaning: the best temperature is near zero,
    // and the least the engines apply is 0.5.
    let timid_questions = noul_questions(100, 0.01, 10, 0);
    let fitted_temperature = fit_temperature(&borrowed(&timid_questions)).unwrap();
    assert_close(fitted_temperature, MINIMUM_APPLIED_TEMPERATURE, 1e-6);
}

#[test]
fn should_fit_nothing_to_no_questions() {
    assert_eq!(fit_temperature(&[]), None);
}

#[test]
fn should_lower_the_loss_relative_to_temperature_one() {
    let overconfident_questions = noul_questions(100, 3.0, 7, 0);
    let borrowed_questions = borrowed(&overconfident_questions);
    let fitted_temperature = fit_temperature(&borrowed_questions).unwrap();
    let fitted_loss = mean_negative_log_likelihood(&borrowed_questions, fitted_temperature);
    let shipped_loss = mean_negative_log_likelihood(&borrowed_questions, 1.0);
    assert!(
        fitted_loss < shipped_loss,
        "{fitted_loss} is not below {shipped_loss}"
    );
}
