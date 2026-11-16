//! Unit tests for the calibration report.

use astronomical_decision_contract::calibration_file::CalibrationOverride;

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

fn mixed_plan() -> CalibrationPlan {
    CalibrationPlan {
        kind_verdicts: vec![
            KindVerdict {
                kind: DecisionQuestionKind::Choice,
                fitting_question_count: 0,
                holdout_question_count: 0,
                shipped_holdout_metrics: None,
                outcome: KindOutcome::KeptShipped(KeptShippedReason::NoQuestions),
            },
            KindVerdict {
                kind: DecisionQuestionKind::Score,
                fitting_question_count: 40,
                holdout_question_count: 6,
                shipped_holdout_metrics: None,
                outcome: KindOutcome::KeptShipped(KeptShippedReason::TooFewHoldoutQuestions {
                    have: 6,
                    need: 10,
                }),
            },
            KindVerdict {
                kind: DecisionQuestionKind::Noul,
                fitting_question_count: 100,
                holdout_question_count: 50,
                shipped_holdout_metrics: Some(holdout_metrics(1.3, 0.25)),
                outcome: KindOutcome::Calibrated {
                    kind_calibration: KindCalibration {
                        temperature: 3.5407,
                        range_temperatures: vec![("2".to_owned(), 3.25)],
                        fitted_question_count: 100,
                    },
                    calibrated_holdout_metrics: holdout_metrics(0.611, 0.004),
                },
            },
        ],
        calibration_override: CalibrationOverride::default(),
    }
}

#[test]
fn should_say_in_plain_words_what_changed_for_each_kind() {
    let counts = CalibrationCounts {
        example_count: 150,
        question_count: 150,
        holdout_question_count: 50,
    };
    let report_text = calibration_report_text("laya-MLX-8bit", counts, &mixed_plan());
    assert!(report_text.starts_with(
        "Calibration of laya-MLX-8bit on 150 examples: 150 questions, 50 of them held out to judge the fit.\n"
    ));
    assert!(report_text.contains("Choice questions: kept as shipped, the examples ask none\n"));
    assert!(report_text.contains(
        "Score questions: kept as shipped, 6 held-out questions, 10 needed; add examples about more different states\n"
    ));
    assert!(
        report_text
            .contains("Yes-or-no questions: calibrated, temperature 3.54, 3.25 with 2 options\n")
    );
    assert!(report_text.contains("surprise at the right answer 1.300 -> 0.611 (lower is better)"));
    assert!(report_text.contains("confidence off by 25.0% -> 0.4%"));
    assert!(report_text.contains("right 70.0% of the time (unchanged)"));
}

#[test]
fn should_name_the_reason_a_fit_was_refused() {
    let refused_verdict = KindVerdict {
        kind: DecisionQuestionKind::Noul,
        fitting_question_count: 100,
        holdout_question_count: 50,
        shipped_holdout_metrics: Some(holdout_metrics(0.6, 0.01)),
        outcome: KindOutcome::KeptShipped(KeptShippedReason::NoImprovement {
            best_holdout_metrics: holdout_metrics(0.6, 0.01),
        }),
    };
    assert_eq!(
        kind_verdict_text(&refused_verdict),
        "Yes-or-no questions: kept as shipped, the fit did not beat the shipped temperatures on held-out questions\n"
    );
}
