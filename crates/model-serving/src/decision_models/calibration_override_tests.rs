//! Unit tests for reading a folder's calibration file.

use astronomical_decision_contract::DecisionQuestionKind;

use super::*;

#[test]
fn should_read_no_calibration_from_a_folder_without_the_file() {
    let model_directory = tempfile::tempdir().unwrap();
    let calibration_override = read_calibration_override(model_directory.path()).unwrap();
    assert!(calibration_override.decides_no_kind());
}

#[test]
fn should_read_the_calibration_a_folder_carries() {
    let model_directory = tempfile::tempdir().unwrap();
    std::fs::write(
        model_directory.path().join(DECISION_CALIBRATION_FILE_NAME),
        br#"{"format":"astronomical-decision-calibration","version":1,
            "kinds":{"choice":{"temperature":1.8}}}"#,
    )
    .unwrap();
    let calibration_override = read_calibration_override(model_directory.path()).unwrap();
    assert_eq!(
        calibration_override.temperature_for(DecisionQuestionKind::Choice, 4),
        Some(1.8)
    );
    assert_eq!(
        calibration_override.temperature_for(DecisionQuestionKind::Noul, 2),
        None
    );
}

#[test]
fn should_refuse_to_load_with_a_calibration_it_cannot_use() {
    let model_directory = tempfile::tempdir().unwrap();
    std::fs::write(
        model_directory.path().join(DECISION_CALIBRATION_FILE_NAME),
        b"Parting is such sweet sorrow",
    )
    .unwrap();
    assert!(read_calibration_override(model_directory.path()).is_err());
}

#[test]
fn should_refuse_a_calibration_file_that_is_too_large() {
    let model_directory = tempfile::tempdir().unwrap();
    let oversized_bytes = vec![b' '; MAXIMUM_DECISION_CALIBRATION_FILE_BYTES as usize + 1];
    std::fs::write(
        model_directory.path().join(DECISION_CALIBRATION_FILE_NAME),
        oversized_bytes,
    )
    .unwrap();
    let read_outcome = read_calibration_override(model_directory.path());
    assert!(format!("{read_outcome:?}").contains("too large"));
}
