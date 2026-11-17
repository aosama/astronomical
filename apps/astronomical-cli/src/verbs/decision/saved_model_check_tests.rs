//! Unit tests for comparing the written model's answers with the trained ones.

use super::*;

#[test]
fn should_measure_gaps_and_never_pass_a_length_mismatch_or_a_non_finite_logit() {
    let close = largest_logit_gap(&[1.0, 100.0], &[1.001, 100.05]);
    assert!(close.within_tolerance);
    assert!((close.gap - 0.05).abs() < 1e-4);
    let apart = largest_logit_gap(&[1.0, 2.0], &[1.0, 2.1]);
    assert!(!apart.within_tolerance);
    assert!(!largest_logit_gap(&[1.0], &[1.0, 2.0]).within_tolerance);
    assert!(!largest_logit_gap(&[f32::NAN], &[1.0]).within_tolerance);
    assert!(largest_logit_gap(&[], &[]).within_tolerance);
}

mod written_folder_answers {
    use astronomical_decision_contract::calibration_file::{CalibrationOverride, KindCalibration};
    use astronomical_decision_contract::{
        DecisionAnswer, DecisionQuestionKind, NoulAnswer, QuestionInputCoverage,
    };
    use astronomical_decision_training::calibration_metrics::CalibrationMetrics;
    use astronomical_decision_training::calibration_plan::{KindFit, KindVerdict};
    use astronomical_decision_training::judged_on::JudgedOn;

    use super::*;

    const ROMEO_STATE_KEY: u64 = 7;
    const TRAINED_LOGITS: [f32; 2] = [0.5, -0.5];

    fn served_question(shipped_temperature: f64) -> ScoredQuestion {
        ScoredQuestion {
            question_name: "in_love".to_owned(),
            kind: DecisionQuestionKind::Noul,
            option_logits: vec![0.0, 0.0],
            shipped_temperature,
            correct_option_index: 0,
            state_group_key: ROMEO_STATE_KEY,
        }
    }

    fn metrics() -> CalibrationMetrics {
        CalibrationMetrics {
            question_count: 1,
            correct_count: 1,
            mean_negative_log_likelihood: 0.5,
            mean_brier_score: 0.2,
            expected_calibration_error: 0.1,
        }
    }

    fn plan_with_noul_temperature(temperature: Option<f64>) -> CalibrationPlan {
        let outcome = match temperature {
            Some(temperature) => KindOutcome::Calibrated(KindFit {
                kind_calibration: KindCalibration {
                    temperature,
                    range_temperatures: Vec::new(),
                    fitted_question_count: 1,
                },
                judged_on: JudgedOn::LearnedFromExamples,
                shipped_metrics: metrics(),
                calibrated_metrics: metrics(),
            }),
            None => KindOutcome::NoQuestions,
        };
        CalibrationPlan {
            kind_verdicts: vec![KindVerdict {
                kind: DecisionQuestionKind::Noul,
                fitting_question_count: 1,
                holdout_question_count: 0,
                outcome,
            }],
            calibration_override: CalibrationOverride::default(),
        }
    }

    fn written_answer(option_logits: &[f32], applied_temperature: f64) -> QuestionOutcome {
        QuestionOutcome {
            question_name: "in_love".to_owned(),
            answer: DecisionAnswer::Noul(NoulAnswer {
                probability_true: 0.7,
                answer_confidence: 0.7,
            }),
            option_logits: option_logits.to_vec(),
            applied_temperature,
            input_coverage: QuestionInputCoverage {
                state_token_count: 9,
                state_tokens_read: 9,
                option_token_cap: None,
                distinguishable_option_count: 2,
            },
            head_inputs: None,
            training_record: None,
        }
    }

    fn pairs() -> Vec<ScoredPair> {
        vec![ScoredPair {
            served: served_question(1.0),
            trained_logits: TRAINED_LOGITS.to_vec(),
        }]
    }

    #[test]
    fn should_use_the_fitted_temperature_and_fall_back_to_the_served_one() {
        assert_eq!(
            temperature_after_training(
                &served_question(1.0),
                &plan_with_noul_temperature(Some(2.5))
            ),
            2.5
        );
        assert_eq!(
            temperature_after_training(&served_question(1.3), &plan_with_noul_temperature(None)),
            1.3
        );
    }

    #[test]
    fn should_accept_a_written_answer_that_matches_the_trained_one() {
        let plan = plan_with_noul_temperature(Some(2.5));
        let judging_pairs = pairs();
        let expectation = TrainedExpectation {
            calibration_plan: &plan,
            judging_pairs: &judging_pairs,
        };

        let gap = checked_answer(
            &expectation,
            ROMEO_STATE_KEY,
            &written_answer(&TRAINED_LOGITS, 2.5),
        )
        .unwrap();

        assert_eq!(gap, 0.0);
    }

    #[test]
    fn should_report_a_written_answer_with_other_scores_or_another_temperature() {
        let plan = plan_with_noul_temperature(Some(2.5));
        let judging_pairs = pairs();
        let expectation = TrainedExpectation {
            calibration_plan: &plan,
            judging_pairs: &judging_pairs,
        };

        assert!(matches!(
            checked_answer(
                &expectation,
                ROMEO_STATE_KEY,
                &written_answer(&[2.0, -0.5], 2.5)
            ),
            Err(DecisionToolError::SavedModelDisagrees { .. })
        ));
        assert!(matches!(
            checked_answer(
                &expectation,
                ROMEO_STATE_KEY,
                &written_answer(&TRAINED_LOGITS, 1.0)
            ),
            Err(DecisionToolError::SavedModelDisagrees { .. })
        ));
    }

    #[test]
    fn should_pass_over_a_question_that_was_not_judged() {
        let plan = plan_with_noul_temperature(None);
        let judging_pairs = pairs();
        let expectation = TrainedExpectation {
            calibration_plan: &plan,
            judging_pairs: &judging_pairs,
        };

        assert_eq!(
            checked_answer(
                &expectation,
                ROMEO_STATE_KEY + 1,
                &written_answer(&[9.0, 9.0], 9.0)
            )
            .unwrap(),
            0.0
        );
    }
}
