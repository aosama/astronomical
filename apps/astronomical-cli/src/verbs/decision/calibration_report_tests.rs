//! Unit tests for the calibration report.

use astronomical_decision_contract::calibration_file::CalibrationOverride;

use astronomical_decision_training::calibration_metrics::CalibrationMetrics;

use super::*;

fn holdout_metrics(negative_log_likelihood: f64, calibration_error: f64) -> CalibrationMetrics {
    CalibrationMetrics {
        question_count: 50,
        correct_count: 35,
        mean_negative_log_likelihood: negative_log_likelihood,
        mean_brier_score: 0.4,
        expected_calibration_error: calibration_error,
    }
}

fn noul_fit(judged_on: JudgedOn, calibrated_surprise: f64) -> KindFit {
    KindFit {
        kind_calibration: KindCalibration {
            temperature: 3.5407,
            range_temperatures: vec![("2".to_owned(), 3.25)],
            fitted_question_count: 100,
        },
        judged_on,
        shipped_metrics: holdout_metrics(1.3, 0.25),
        calibrated_metrics: holdout_metrics(calibrated_surprise, 0.004),
    }
}

fn plan_of(noul_outcome: KindOutcome) -> CalibrationPlan {
    CalibrationPlan {
        kind_verdicts: vec![
            KindVerdict {
                kind: DecisionQuestionKind::Choice,
                fitting_question_count: 0,
                holdout_question_count: 0,
                outcome: KindOutcome::NoQuestions,
            },
            KindVerdict {
                kind: DecisionQuestionKind::Noul,
                fitting_question_count: 100,
                holdout_question_count: 50,
                outcome: noul_outcome,
            },
        ],
        calibration_override: CalibrationOverride::default(),
    }
}

fn counts() -> CalibrationCounts {
    CalibrationCounts {
        example_count: 150,
        question_count: 150,
        distinct_state_count: 150,
        holdout_question_count: 50,
    }
}

#[test]
fn should_say_in_plain_words_what_changed_for_each_kind() {
    let plan = plan_of(KindOutcome::Calibrated(noul_fit(
        JudgedOn::HeldBackQuestions,
        0.611,
    )));
    let report_text = calibration_report_text("laya-MLX-8bit", counts(), &plan);
    assert!(report_text.starts_with(
        "Calibration of laya-MLX-8bit on 150 examples: 150 questions about 150 different texts, 50 of them held out to judge the fit.\n"
    ));
    assert!(report_text.contains("Choice questions: as shipped, the examples ask none\n"));
    assert!(
        report_text
            .contains("Yes-or-no questions: calibrated, temperature 3.54, 3.25 with 2 options\n")
    );
    assert!(report_text.contains(
        "On 50 held-out questions: surprise at the right answer 1.300 -> 0.611 (lower is better)"
    ));
    assert!(report_text.contains("confidence off by 25.0% -> 0.4%"));
    assert!(report_text.contains("right 70.0% of the time (unchanged)"));
    assert!(report_text.contains("Judged better on held-out questions."));
    assert!(calibration_warnings(&plan).is_empty());
}

#[test]
fn should_write_a_fit_that_did_not_help_and_say_so() {
    let plan = plan_of(KindOutcome::Calibrated(noul_fit(
        JudgedOn::HeldBackQuestions,
        1.4,
    )));
    let report_text = calibration_report_text("laya-MLX-8bit", counts(), &plan);
    assert!(
        report_text.contains(
            "Note: it did no better than the shipped temperatures on held-out questions."
        )
    );
    assert!(report_text.contains("Judged not better"));
    assert!(report_text.contains("whether to use it is your call"));
    assert_eq!(calibration_warnings(&plan).len(), 1);
}

#[test]
fn should_say_a_fit_on_one_text_was_not_judged() {
    let plan = plan_of(KindOutcome::Calibrated(noul_fit(
        JudgedOn::LearnedFromExamples,
        0.611,
    )));
    let report_text = calibration_report_text("laya-MLX-8bit", counts(), &plan);
    assert!(report_text.contains("On the 50 questions it was fitted to"));
    assert!(report_text.contains("Not judged"));
    assert!(calibration_warnings(&plan)[0].starts_with("Yes-or-no questions: no question"));
}
