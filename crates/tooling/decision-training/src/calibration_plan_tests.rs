//! Unit tests for fitting and judging temperatures per kind.

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
    let KindOutcome::Calibrated(kind_fit) = &noul_verdict.outcome else {
        panic!("noul was not calibrated: {:?}", noul_verdict.outcome);
    };
    let kind_calibration = &kind_fit.kind_calibration;
    let calibrated_holdout_metrics = &kind_fit.calibrated_metrics;
    assert_eq!(kind_fit.judged_on, JudgedOn::HeldBackQuestions);
    assert!(kind_fit.improved());
    assert_close(
        kind_calibration.temperature,
        BEST_TEMPERATURE_FOR_GAP_THREE,
        1e-4,
    );
    assert!(kind_calibration.range_temperatures.is_empty());
    assert_eq!(kind_calibration.fitted_question_count, 100);
    let shipped_metrics = &kind_fit.shipped_metrics;
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
        KindOutcome::NoQuestions
    );
}

fn noul_fit(calibration_plan: &CalibrationPlan) -> &KindFit {
    match &verdict_for(calibration_plan, DecisionQuestionKind::Noul).outcome {
        KindOutcome::Calibrated(kind_fit) => kind_fit,
        KindOutcome::NoQuestions => panic!("noul was not calibrated"),
    }
}

#[test]
fn should_write_a_fit_that_did_not_improve_and_say_so() {
    let holdout_split = split_of(
        noul_questions(100, CALIBRATED_SEVENTY_PERCENT_GAP, 7, 0),
        noul_questions(50, CALIBRATED_SEVENTY_PERCENT_GAP, 7, 1_000),
    );
    let calibration_plan = plan_calibration(&holdout_split).unwrap();
    assert!(!noul_fit(&calibration_plan).improved());
    assert!(calibration_plan.calibration_override.noul.is_some());
}

#[test]
fn should_write_a_fit_that_does_worse_on_held_out_questions_and_say_so() {
    // The fitting half is overconfident but the held-out half is calibrated
    // as shipped: the fit makes held-out answers worse, which is reported.
    let holdout_split = split_of(
        noul_questions(100, 3.0, 7, 0),
        noul_questions(50, 3.0, 10, 1_000),
    );
    let calibration_plan = plan_calibration(&holdout_split).unwrap();
    let kind_fit = noul_fit(&calibration_plan);
    assert_eq!(kind_fit.judged_on, JudgedOn::HeldBackQuestions);
    assert!(!kind_fit.improved());
    assert!(calibration_plan.calibration_override.noul.is_some());
}

#[test]
fn should_fit_a_single_question_and_judge_it_on_itself() {
    let holdout_split = split_of(noul_questions(1, 3.0, 7, 0), Vec::new());
    let calibration_plan = plan_calibration(&holdout_split).unwrap();
    let kind_fit = noul_fit(&calibration_plan);
    assert_eq!(kind_fit.judged_on, JudgedOn::LearnedFromExamples);
    assert_eq!(kind_fit.kind_calibration.fitted_question_count, 1);
}

#[test]
fn should_fit_on_held_out_questions_when_no_question_of_the_kind_is_left_to_fit() {
    let holdout_split = split_of(Vec::new(), noul_questions(5, 3.0, 7, 0));
    let calibration_plan = plan_calibration(&holdout_split).unwrap();
    let kind_fit = noul_fit(&calibration_plan);
    assert_eq!(kind_fit.judged_on, JudgedOn::LearnedFromExamples);
    assert_eq!(kind_fit.kind_calibration.fitted_question_count, 5);
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
    decision_contract::calibration::MINIMUM_APPLIED_TEMPERATURE;

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

#[test]
fn should_record_the_plans_verdict_from_its_kinds() {
    let improved = plan_calibration(&split_of(
        noul_questions(100, 3.0, 7, 0),
        noul_questions(50, 3.0, 7, 1_000),
    ))
    .unwrap();
    let not_improved = plan_calibration(&split_of(
        noul_questions(100, 3.0, 7, 0),
        noul_questions(50, 3.0, 10, 1_000),
    ))
    .unwrap();
    let unjudged = plan_calibration(&split_of(noul_questions(3, 3.0, 7, 0), Vec::new())).unwrap();

    assert_eq!(improved.recorded_verdict(), RecordedVerdict::JudgedBetter);
    assert_eq!(
        not_improved.recorded_verdict(),
        RecordedVerdict::JudgedNotBetter
    );
    assert_eq!(unjudged.recorded_verdict(), RecordedVerdict::NotJudged);
}

#[test]
fn should_record_not_judged_for_a_plan_that_calibrated_nothing() {
    let empty_plan = plan_calibration(&split_of(Vec::new(), Vec::new())).unwrap();
    assert_eq!(empty_plan.recorded_verdict(), RecordedVerdict::NotJudged);
    assert!(empty_plan.calibration_override.decides_no_kind());
}

#[test]
fn should_record_not_judged_when_one_kind_is_judged_and_another_is_not() {
    let mut fitting_questions = noul_questions(100, 3.0, 7, 0);
    fitting_questions.extend(choice_questions_needing_two_temperatures(5, 50_000));
    let mixed_plan = plan_calibration(&split_of(
        fitting_questions,
        noul_questions(50, 3.0, 7, 1_000),
    ))
    .unwrap();
    assert_eq!(noul_fit(&mixed_plan).judged_on, JudgedOn::HeldBackQuestions);
    assert_eq!(mixed_plan.recorded_verdict(), RecordedVerdict::NotJudged);
}
