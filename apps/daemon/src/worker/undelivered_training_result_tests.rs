use astronomical_decision_contract::decision_training_job::DecisionTrainingOutcome;
use astronomical_decision_contract::trained_tensors_file::handover_file_name;

use super::*;

fn trained_output(trained_tensors_file: Option<String>) -> DecisionsOutput {
    DecisionsOutput::ModelTrained {
        outcome: Box::new(DecisionTrainingOutcome {
            examples: Vec::new(),
            trained_tensors: Vec::new(),
            adapter_tensors: Vec::new(),
            runs: Vec::new(),
            best_seed: 1,
            largest_parity_gap: 0.0,
            parity_gap_beyond_rounding: None,
            trained_tensors_file,
            recomputed_text_count: 0,
            indistinguishable_question_count: 0,
        }),
        elapsed_millis: 1,
    }
}

#[test]
fn should_delete_the_file_an_undelivered_training_result_names() {
    let handover_directory = tempfile::tempdir().unwrap();
    let file_name = handover_file_name(9, 42);
    std::fs::write(handover_directory.path().join(&file_name), b"weights").unwrap();

    discard_undelivered_handover_file(
        &trained_output(Some(file_name.clone())),
        handover_directory.path(),
    );

    assert!(!handover_directory.path().join(&file_name).exists());
}

#[test]
fn should_leave_files_alone_for_results_without_one_or_with_a_foreign_name() {
    let handover_directory = tempfile::tempdir().unwrap();
    std::fs::write(handover_directory.path().join("romeo.txt"), b"keep").unwrap();

    discard_undelivered_handover_file(&trained_output(None), handover_directory.path());
    discard_undelivered_handover_file(
        &trained_output(Some("romeo.txt".to_owned())),
        handover_directory.path(),
    );
    discard_undelivered_handover_file(
        &DecisionsOutput::TrainingProgress {
            stage: astronomical_decision_contract::decision_training_job::DecisionTrainingStage::Training,
            completed: 1,
            total: 2,
        },
        handover_directory.path(),
    );

    assert!(handover_directory.path().join("romeo.txt").exists());
}
