//! Unit tests for the training report.

use astronomical_decision_contract::calibration_file::CalibrationOverride;
use astronomical_decision_training::calibration_plan::CalibrationPlan;
use astronomical_decision_training::training_verdict::{CheckAgreement, KindComparison};

use super::*;

fn metrics(correct_count: usize, surprise: f64, confidence_error: f64) -> CalibrationMetrics {
    CalibrationMetrics {
        question_count: 40,
        correct_count,
        mean_negative_log_likelihood: surprise,
        mean_brier_score: 0.3,
        expected_calibration_error: confidence_error,
    }
}

fn verdict_with(refusals: Vec<TrainingRefusal>) -> TrainingVerdict {
    TrainingVerdict {
        served_overall: metrics(26, 0.812, 0.18),
        shipped_overall: metrics(26, 0.774, 0.091),
        trained_overall: metrics(29, 0.701, 0.06),
        kind_comparisons: vec![KindComparison {
            kind: DecisionQuestionKind::Choice,
            shipped: metrics(26, 0.774, 0.091),
            trained: metrics(29, 0.701, 0.06),
        }],
        check_agreement: CheckAgreement {
            agreeing_count: 117,
            question_count: 120,
        },
        calibration_plan: CalibrationPlan {
            kind_verdicts: Vec::new(),
            calibration_override: CalibrationOverride::default(),
        },
        refusals,
    }
}

fn seed_run(seed: u64, best_epoch: usize, epochs_run: usize, ending: RunEnding) -> RunSummary {
    RunSummary {
        seed,
        best_choosing_loss: 0.5,
        best_epoch,
        epochs_run,
        ending,
    }
}

fn report_for(verdict: &TrainingVerdict) -> String {
    report_for_scope(verdict, TrainingScope::FinalLayers)
}

fn report_for_scope(verdict: &TrainingVerdict, scope: TrainingScope) -> String {
    let seed_runs = [
        seed_run(1, 8, 14, RunEnding::StoppedImproving),
        seed_run(2, 0, 6, RunEnding::StoppedImproving),
        seed_run(3, 0, 1, RunEnding::NonFiniteStep { epoch: 1 }),
    ];
    training_report_text(&TrainingSummary {
        model_id: "laya-MLX-8bit",
        scope,
        example_count: 210,
        part_counts: PartCounts {
            training: 140,
            choosing: 30,
            judging: 40,
        },
        indistinguishable_question_count: 3,
        largest_parity_gap: 0.000_4,
        runs: &seed_runs,
        best_seed: 1,
        verdict,
    })
}

#[test]
fn should_tell_the_whole_story_of_a_kept_training_in_plain_words() {
    let report_text = report_for(&verdict_with(Vec::new()));
    assert!(report_text.contains(
        "Training the final layers of laya-MLX-8bit on 210 examples: 210 questions, 140 to train on, 30 to choose with and 40 held back to judge."
    ));
    assert!(report_text.contains("3 questions were left out"));
    assert!(report_text.contains("to within 0.0004"));
    assert!(report_text.contains(
        "Run 1: best after pass 8 of 14, the best on the questions held back for choosing."
    ));
    assert!(report_text.contains("Run 2: no pass beat the starting weights in 6 passes."));
    assert!(report_text.contains("Run 3: a number stopped being finite in pass 1"));
    assert!(report_text.contains("as served today"));
    assert!(report_text.contains("shipped, recalibrated"));
    assert!(report_text.contains("Choice questions (40): right 65.0% -> 72.5%"));
    assert!(report_text.contains("117 of 120 answered as before"));
    assert!(
        report_text.ends_with(
            "Kept: the trained model beats the shipped one on questions it never saw.\n"
        )
    );
}

#[test]
fn should_name_every_reason_a_training_was_not_kept() {
    let report_text = report_for(&verdict_with(vec![
        TrainingRefusal::LessAccurate {
            shipped: 0.65,
            trained: 0.6,
        },
        TrainingRefusal::CheckAnswersChanged(CheckAgreement {
            agreeing_count: 100,
            question_count: 120,
        }),
        TrainingRefusal::Collapsed {
            kind: DecisionQuestionKind::Noul,
        },
    ]));
    assert!(report_text.contains("Not kept:\n"));
    assert!(report_text.contains("it was right less often (65.0% -> 60.0%)"));
    assert!(report_text.contains("changed the answers to 20 of 120 questions nobody labelled"));
    assert!(report_text.contains("nearly every yes-or-no question the same answer"));
}

#[test]
fn should_say_when_the_whole_model_trained_and_that_its_weights_were_kept() {
    let report = report_for_scope(&verdict_with(Vec::new()), TrainingScope::WholeModel);

    assert!(
        report.starts_with("Training the whole of laya-MLX-8bit"),
        "{report}"
    );
    assert!(report.contains("add-on weights"), "{report}");
    assert!(
        report.contains("original weights are never changed"),
        "{report}"
    );
    assert!(
        report.contains("the trainable copy of the whole model reproduced"),
        "{report}"
    );
    assert!(!report.contains("final layers of"), "{report}");
}
