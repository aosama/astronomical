use decision_contract::training_record::{RecordedVerdict, training_record_document};

use super::*;

fn not_judged_record() -> TrainingRecord {
    TrainingRecord {
        verdict: RecordedVerdict::NotJudged,
        labelled_question_count: 1,
        distinct_state_count: 1,
        warnings: Vec::new(),
    }
}

#[test]
fn should_read_the_record_a_folder_carries() {
    let model_directory = tempfile::tempdir().unwrap();
    std::fs::write(
        model_directory
            .path()
            .join(DECISION_TRAINING_RECORD_FILE_NAME),
        training_record_document(&not_judged_record()),
    )
    .unwrap();

    assert_eq!(
        read_training_record_file(model_directory.path()).unwrap(),
        Some(not_judged_record())
    );
}

#[test]
fn should_find_no_record_in_a_folder_nobody_trained() {
    let model_directory = tempfile::tempdir().unwrap();

    assert!(
        read_training_record_file(model_directory.path())
            .unwrap()
            .is_none()
    );
}

#[test]
fn should_refuse_a_record_over_the_size_limit() {
    let model_directory = tempfile::tempdir().unwrap();
    let oversized = vec![b' '; MAXIMUM_DECISION_TRAINING_RECORD_FILE_BYTES as usize + 1];
    std::fs::write(
        model_directory
            .path()
            .join(DECISION_TRAINING_RECORD_FILE_NAME),
        oversized,
    )
    .unwrap();

    assert!(matches!(
        read_training_record_file(model_directory.path()),
        Err(TrainingRecordFileError::TooLarge)
    ));
}

#[test]
fn should_refuse_a_record_of_another_version() {
    let model_directory = tempfile::tempdir().unwrap();
    std::fs::write(
        model_directory.path().join(DECISION_TRAINING_RECORD_FILE_NAME),
        br#"{"format":"astronomical-decision-training-record","version":2,"verdict":"not_judged","labelled_question_count":1,"distinct_state_count":1,"warnings":[]}"#,
    )
    .unwrap();

    assert!(matches!(
        read_training_record_file(model_directory.path()),
        Err(TrainingRecordFileError::Malformed(_))
    ));
}
