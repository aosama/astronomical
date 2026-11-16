//! Unit tests for the never-worse gate.

use super::*;
use crate::test_support::{
    BEST_TEMPERATURE_FOR_GAP_THREE, CALIBRATED_SEVENTY_PERCENT_GAP, assert_close, noul_questions,
    scored_question,
};

fn split_of(
    fitting_questions: Vec<ScoredQuestion>,
    holdout_questions: Vec<ScoredQuestion>,
) -> HoldoutSplit {
    HoldoutSplit {
        fitting_questions,
        holdout_questions,
    }
}

fn verdict_for(calibration_plan: &CalibrationPlan, kind: DecisionQuestionKind) -> &KindVerdict {
    for kind_verdict in &calibration_plan.kind_verdicts {
        if kind_verdict.kind == kind {
            return kind_verdict;
        }
    }
    panic!("no verdict for {kind:?}");
}

/// Choice questions in two option-count ranges that need very different
/// temperatures: two-option questions are overconfident, four-option ones
/// are always right but timid.
fn choice_questions_needing_two_temperatures(
    per_range_count: usize,
    first_state_key: u64,
) -> Vec<ScoredQuestion> {
    let mut choice_questions = Vec::new();
    for question_position in 0..per_range_count {
        let state_key = first_state_key + 2 * question_position as u64;
        let two_option_correct = usize::from(question_position % 10 < 7);
        choice_questions.push(scored_question(
            DecisionQuestionKind::Choice,
            &[0.0, 3.0],
            two_option_correct,
            state_key,
        ));
        choice_questions.push(scored_question(
            DecisionQuestionKind::Choice,
            &[1.0, 0.0, 0.0, 0.0],
            0,
            state_key + 1,
        ));
    }
    choice_questions
}

#[test]
fn should_calibrate_an_overconfident_kind_and_leave_the_others_to_the_family() {
    let holdout_split = split_of(
        noul_questions(100, 3.0, 7, 0),
        noul_questions(50, 3.0, 7, 1_000),
    );
    let calibration_plan = plan_calibration(&holdout_split).unwrap();
    let noul_verdict = verdict_for(&calibration_plan, DecisionQuestionKind::Noul);
    let KindOutcome::Calibrated {
        kind_calibration,
        calibrated_holdout_metrics,
    } = &noul_verdict.outcome
    else {
        panic!("noul was not calibrated: {:?}", noul_verdict.outcome);
    };
    assert_close(
        kind_calibration.temperature,
        BEST_TEMPERATURE_FOR_GAP_THREE,
        1e-4,
    );
    assert!(kind_calibration.range_temperatures.is_empty());
    assert_eq!(kind_calibration.fitted_question_count, 100);
    let shipped_metrics = noul_verdict.shipped_holdout_metrics.unwrap();
    assert!(
        calibrated_holdout_metrics.mean_negative_log_likelihood
            < shipped_metrics.mean_negative_log_likelihood
    );
    assert_eq!(
        calibrated_holdout_metrics.correct_count,
        shipped_metrics.correct_count
    );
    assert!(calibration_plan.calibration_override.noul.is_some());
    assert!(calibration_plan.calibration_override.choice.is_none());
    assert_eq!(
        verdict_for(&calibration_plan, DecisionQuestionKind::Choice).outcome,
        KindOutcome::KeptShipped(KeptShippedReason::NoQuestions)
    );
}

#[test]
fn should_keep_the_shipped_temperature_when_it_is_already_right() {
    let holdout_split = split_of(
        noul_questions(100, CALIBRATED_SEVENTY_PERCENT_GAP, 7, 0),
        noul_questions(50, CALIBRATED_SEVENTY_PERCENT_GAP, 7, 1_000),
    );
    let calibration_plan = plan_calibration(&holdout_split).unwrap();
    let noul_verdict = verdict_for(&calibration_plan, DecisionQuestionKind::Noul);
    assert!(matches!(
        noul_verdict.outcome,
        KindOutcome::KeptShipped(KeptShippedReason::NoImprovement { .. })
    ));
    assert!(calibration_plan.calibration_override.decides_no_kind());
}

#[test]
fn should_keep_the_shipped_temperature_when_the_fit_does_worse_on_held_out_questions() {
    // The fitting half is overconfident but the held-out half is calibrated
    // as shipped: the fit would make held-out answers worse, so it is refused.
    let holdout_split = split_of(
        noul_questions(100, 3.0, 7, 0),
        noul_questions(50, 3.0, 10, 1_000),
    );
    let calibration_plan = plan_calibration(&holdout_split).unwrap();
    assert!(calibration_plan.calibration_override.decides_no_kind());
}

#[test]
fn should_refuse_to_fit_on_too_few_questions_on_either_side() {
    let few_fitting = split_of(
        noul_questions(19, 3.0, 7, 0),
        noul_questions(50, 3.0, 7, 1_000),
    );
    let few_holdout = split_of(
        noul_questions(100, 3.0, 7, 0),
        noul_questions(9, 3.0, 7, 1_000),
    );
    let few_fitting_plan = plan_calibration(&few_fitting).unwrap();
    let few_holdout_plan = plan_calibration(&few_holdout).unwrap();
    assert_eq!(
        verdict_for(&few_fitting_plan, DecisionQuestionKind::Noul).outcome,
        KindOutcome::KeptShipped(KeptShippedReason::TooFewFittingQuestions { have: 19, need: 20 })
    );
    assert_eq!(
        verdict_for(&few_holdout_plan, DecisionQuestionKind::Noul).outcome,
        KindOutcome::KeptShipped(KeptShippedReason::TooFewHoldoutQuestions { have: 9, need: 10 })
    );
}

#[test]
fn should_fit_a_temperature_at_exactly_the_minimum_question_counts() {
    let holdout_split = split_of(
        noul_questions(20, 3.0, 7, 0),
        noul_questions(10, 3.0, 7, 1_000),
    );
    let calibration_plan = plan_calibration(&holdout_split).unwrap();
    assert!(calibration_plan.calibration_override.noul.is_some());
}

#[test]
fn should_keep_per_range_temperatures_only_when_they_beat_one_temperature() {
    let holdout_split = split_of(
        choice_questions_needing_two_temperatures(40, 0),
        choice_questions_needing_two_temperatures(20, 10_000),
    );
    let calibration_plan = plan_calibration(&holdout_split).unwrap();
    let choice_calibration = calibration_plan.calibration_override.choice.unwrap();
    let range_names: Vec<&str> = choice_calibration
        .range_temperatures
        .iter()
        .map(|(range_name, _)| range_name.as_str())
        .collect();
    assert_eq!(range_names, vec!["2", "3-5"]);
    assert_close(
        choice_calibration.range_temperatures[0].1,
        BEST_TEMPERATURE_FOR_GAP_THREE,
        1e-4,
    );
    assert_close(
        choice_calibration.range_temperatures[1].1,
        MINIMUM_APPLIED_TEMPERATURE_FOR_TESTS,
        1e-6,
    );
}

const MINIMUM_APPLIED_TEMPERATURE_FOR_TESTS: f64 =
    astronomical_decision_contract::calibration::MINIMUM_APPLIED_TEMPERATURE;

#[test]
fn should_stop_when_a_temperature_would_change_the_most_probable_answer() {
    // A negative shipped temperature reverses the option order, which no
    // real engine does; it stands in for scoring code that has gone wrong.
    let mut broken_question = scored_question(DecisionQuestionKind::Noul, &[0.0, 2.0], 1, 1);
    broken_question.shipped_temperature = -1.0;
    let single_temperature = KindCalibration {
        temperature: 1.0,
        range_temperatures: Vec::new(),
        fitted_question_count: 1,
    };
    assert_eq!(
        check_answers_unchanged(&[&broken_question], &single_temperature),
        Err(CalibrationPlanError::TemperatureChangedTheAnswer {
            question: "question-1".to_owned()
        })
    );
}

#[test]
fn should_not_count_a_tie_as_a_changed_answer() {
    let tied_question = scored_question(DecisionQuestionKind::Choice, &[1.0, 1.0, 0.0], 1, 1);
    let single_temperature = KindCalibration {
        temperature: 2.0,
        range_temperatures: Vec::new(),
        fitted_question_count: 1,
    };
    assert_eq!(
        check_answers_unchanged(&[&tied_question], &single_temperature),
        Ok(())
    );
}
