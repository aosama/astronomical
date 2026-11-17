use std::os::unix::fs::PermissionsExt;

use astronomical_decision_contract::FloatVector;
use astronomical_decision_contract::decision_training_job::TrainedTensor;
use astronomical_decision_contract::trained_tensors_file::read_trained_tensors_file;

use super::*;

fn finished_outcome() -> Box<DecisionTrainingOutcome> {
    Box::new(DecisionTrainingOutcome {
        examples: Vec::new(),
        trained_tensors: vec![TrainedTensor {
            name: "scorer.bias".to_owned(),
            shape: vec![1],
            values: FloatVector(vec![0.25]),
        }],
        adapter_tensors: vec![
            TrainedTensor {
                name: "layers.0.mlp.up_proj.adapter_down".to_owned(),
                shape: vec![1, 2],
                values: FloatVector(vec![1.0, 2.0]),
            },
            TrainedTensor {
                name: "layers.0.mlp.up_proj.adapter_up".to_owned(),
                shape: vec![2, 1],
                values: FloatVector(vec![3.0, 4.0]),
            },
        ],
        runs: Vec::new(),
        best_seed: 1,
        largest_parity_gap: 0.0,
        parity_gap_beyond_rounding: None,
        trained_tensors_file: None,
        recomputed_text_count: 0,
        indistinguishable_question_count: 0,
    })
}

#[test]
fn should_move_the_tensors_to_a_private_file_and_leave_its_name() {
    let handover_directory = tempfile::tempdir().unwrap();
    let original = finished_outcome();

    let handed_over =
        hand_over_trained_tensors(original.clone(), handover_directory.path()).unwrap();

    assert!(handed_over.trained_tensors.is_empty());
    assert!(handed_over.adapter_tensors.is_empty());
    let file_name = handed_over.trained_tensors_file.clone().unwrap();
    let handover_path = handover_file_path(handover_directory.path(), &file_name).unwrap();
    let mode = std::fs::metadata(&handover_path)
        .unwrap()
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o600);
    let read_back = read_trained_tensors_file(&std::fs::read(&handover_path).unwrap()).unwrap();
    assert_eq!(read_back.trained_tensors, original.trained_tensors);
    assert_eq!(read_back.adapter_tensors, original.adapter_tensors);
}

#[test]
fn should_send_an_outcome_without_tensors_as_it_is() {
    let handover_directory = tempfile::tempdir().unwrap();
    let mut empty_outcome = finished_outcome();
    empty_outcome.trained_tensors.clear();
    empty_outcome.adapter_tensors.clear();

    let handed_over =
        hand_over_trained_tensors(empty_outcome.clone(), handover_directory.path()).unwrap();

    assert_eq!(handed_over, empty_outcome);
    assert_eq!(
        std::fs::read_dir(handover_directory.path())
            .unwrap()
            .count(),
        0
    );
}

#[test]
fn should_fail_with_a_reason_when_the_folder_cannot_be_written() {
    let missing_directory = Path::new("/nonexistent-astronomical-handover");

    let failure = hand_over_trained_tensors(finished_outcome(), missing_directory).unwrap_err();

    assert!(failure.contains("could not be written"));
}

#[test]
fn should_fail_on_a_value_that_is_not_a_finite_number_and_write_nothing() {
    let handover_directory = tempfile::tempdir().unwrap();
    let mut outcome = finished_outcome();
    outcome.trained_tensors[0].values = FloatVector(vec![f32::INFINITY]);

    let failure = hand_over_trained_tensors(outcome, handover_directory.path()).unwrap_err();

    assert!(failure.contains("not a finite number"));
    assert_eq!(
        std::fs::read_dir(handover_directory.path())
            .unwrap()
            .count(),
        0
    );
}

#[test]
fn should_discard_the_named_file_when_its_message_could_not_be_sent() {
    let handover_directory = tempfile::tempdir().unwrap();
    let handed_over =
        hand_over_trained_tensors(finished_outcome(), handover_directory.path()).unwrap();

    discard_handover_file(
        handed_over.trained_tensors_file.as_deref(),
        handover_directory.path(),
    );

    assert_eq!(
        std::fs::read_dir(handover_directory.path())
            .unwrap()
            .count(),
        0
    );
}

#[test]
fn should_leave_recent_hand_over_files_and_other_files_alone_when_clearing() {
    let handover_directory = tempfile::tempdir().unwrap();
    let recent_name = handover_file_name(7, 1);
    std::fs::write(handover_directory.path().join(&recent_name), b"recent").unwrap();
    std::fs::write(handover_directory.path().join("romeo.txt"), b"keep").unwrap();

    clear_stale_handover_files(handover_directory.path());

    assert!(handover_directory.path().join(&recent_name).exists());
    assert!(handover_directory.path().join("romeo.txt").exists());
}

#[test]
fn should_discard_nothing_for_an_outcome_without_a_file_or_with_a_foreign_name() {
    let handover_directory = tempfile::tempdir().unwrap();
    std::fs::write(handover_directory.path().join("romeo.txt"), b"keep").unwrap();

    discard_handover_file(None, handover_directory.path());
    discard_handover_file(Some("romeo.txt"), handover_directory.path());

    assert!(handover_directory.path().join("romeo.txt").exists());
}
