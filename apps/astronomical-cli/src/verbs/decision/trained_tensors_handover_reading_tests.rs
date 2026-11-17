use astronomical_decision_contract::FloatVector;
use astronomical_decision_contract::decision_training_job::TrainedTensor;
use astronomical_decision_contract::trained_tensors_file::{
    handover_file_name, trained_tensors_file_bytes,
};

use super::*;

fn bare_outcome() -> DecisionTrainingOutcome {
    DecisionTrainingOutcome {
        examples: Vec::new(),
        trained_tensors: Vec::new(),
        adapter_tensors: Vec::new(),
        runs: Vec::new(),
        best_seed: 1,
        largest_parity_gap: 0.0,
        parity_gap_beyond_rounding: None,
        trained_tensors_file: None,
        recomputed_text_count: 0,
        indistinguishable_question_count: 0,
    }
}

fn scorer_bias() -> TrainedTensor {
    TrainedTensor {
        name: "scorer.bias".to_owned(),
        shape: vec![1],
        values: FloatVector(vec![0.75]),
    }
}

/// An outcome naming a hand-over file holding `file_bytes` in `folder`.
fn outcome_naming_file(
    folder: &Path,
    file_bytes: &[u8],
) -> (DecisionTrainingOutcome, std::path::PathBuf) {
    let file_name = handover_file_name(11, 22);
    let handover_path = folder.join(&file_name);
    std::fs::write(&handover_path, file_bytes).unwrap();
    let mut outcome = bare_outcome();
    outcome.trained_tensors_file = Some(file_name);
    (outcome, handover_path)
}

#[test]
fn should_put_the_handed_over_tensors_back_and_delete_the_file() {
    let handover_directory = tempfile::tempdir().unwrap();
    let file_bytes = trained_tensors_file_bytes(HandedOverTensors {
        trained_tensors: vec![scorer_bias()],
        adapter_tensors: Vec::new(),
    })
    .unwrap();
    let (outcome, handover_path) = outcome_naming_file(handover_directory.path(), &file_bytes);

    let restored = take_handed_over_tensors(outcome, handover_directory.path()).unwrap();

    assert_eq!(restored.trained_tensors, vec![scorer_bias()]);
    assert_eq!(restored.trained_tensors_file, None);
    assert!(!handover_path.exists());
}

#[test]
fn should_keep_tensors_that_came_in_place() {
    let handover_directory = tempfile::tempdir().unwrap();
    let mut outcome = bare_outcome();
    outcome.trained_tensors = vec![scorer_bias()];

    let restored = take_handed_over_tensors(outcome.clone(), handover_directory.path()).unwrap();

    assert_eq!(restored, outcome);
}

#[test]
fn should_refuse_a_name_that_points_anywhere_else_and_touch_nothing() {
    let handover_directory = tempfile::tempdir().unwrap();
    let romeo_path = handover_directory.path().join("romeo.txt");
    std::fs::write(&romeo_path, b"keep").unwrap();
    let mut outcome = bare_outcome();
    outcome.trained_tensors_file = Some("../romeo.txt".to_owned());

    let failure = take_handed_over_tensors(outcome, handover_directory.path()).unwrap_err();

    assert!(
        failure
            .to_string()
            .contains("not a name the worker gives its files")
    );
    assert!(romeo_path.exists());
}

#[test]
fn should_fail_the_training_when_the_handed_over_file_is_missing() {
    let handover_directory = tempfile::tempdir().unwrap();
    let mut outcome = bare_outcome();
    outcome.trained_tensors_file = Some(handover_file_name(1, 2));

    let failure = take_handed_over_tensors(outcome, handover_directory.path()).unwrap_err();

    assert!(failure.to_string().contains("could not be read"));
    assert!(failure.to_string().contains("nothing was written"));
}

#[test]
fn should_keep_a_file_that_is_not_trained_tensors_and_fail_the_training() {
    let handover_directory = tempfile::tempdir().unwrap();
    let (outcome, handover_path) =
        outcome_naming_file(handover_directory.path(), b"Parting is such sweet sorrow");

    assert!(take_handed_over_tensors(outcome, handover_directory.path()).is_err());
    assert!(handover_path.exists());
}
