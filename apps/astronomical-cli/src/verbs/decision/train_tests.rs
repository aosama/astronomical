//! Unit tests for the training journey's own steps. Training itself is
//! tested in the training crate; the whole journey runs live on the
//! Development instance.

use astronomical_decision_contract::DecisionQuestionKind;
use astronomical_decision_contract::calibration_file::{
    KindCalibration, calibration_document_text,
};

use super::*;

fn calibration_of(kind: DecisionQuestionKind, temperature: f64) -> CalibrationOverride {
    let mut calibration = CalibrationOverride::default();
    calibration.set_kind_calibration(
        kind,
        KindCalibration {
            temperature,
            range_temperatures: Vec::new(),
            fitted_question_count: 40,
        },
    );
    calibration
}

#[test]
fn should_keep_an_earlier_calibration_for_kinds_training_did_not_refit() {
    let source_directory = tempfile::tempdir().unwrap();
    std::fs::write(
        source_directory.path().join(DECISION_CALIBRATION_FILE_NAME),
        calibration_document_text(&calibration_of(DecisionQuestionKind::Noul, 2.5)),
    )
    .unwrap();
    let merged = merged_calibration(
        source_directory.path(),
        &calibration_of(DecisionQuestionKind::Choice, 1.5),
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        merged.temperature_for(DecisionQuestionKind::Noul, 2),
        Some(2.5)
    );
    assert_eq!(
        merged.temperature_for(DecisionQuestionKind::Choice, 4),
        Some(1.5)
    );
}

#[test]
fn should_write_no_calibration_when_no_kind_has_one() {
    let source_directory = tempfile::tempdir().unwrap();
    let merged =
        merged_calibration(source_directory.path(), &CalibrationOverride::default()).unwrap();
    assert!(merged.is_none());
}

#[test]
fn should_refuse_an_earlier_calibration_it_cannot_read() {
    let source_directory = tempfile::tempdir().unwrap();
    std::fs::write(
        source_directory.path().join(DECISION_CALIBRATION_FILE_NAME),
        b"Parting is such sweet sorrow",
    )
    .unwrap();
    assert!(matches!(
        merged_calibration(source_directory.path(), &CalibrationOverride::default()),
        Err(DecisionToolError::EarlierCalibrationUnreadable(_))
    ));
}

#[test]
fn should_train_only_the_decision_families() {
    assert_eq!(
        tunable_family(ModelFamily::Clef, "clef").unwrap(),
        TunableHeadFamily::Clef
    );
    assert_eq!(
        tunable_family(ModelFamily::Kev, "kev").unwrap(),
        TunableHeadFamily::Kev
    );
    assert!(matches!(
        tunable_family(ModelFamily::Qwen3_5, "qwen"),
        Err(DecisionToolError::NotTrainable { .. })
    ));
}

#[test]
fn should_keep_a_written_folder_and_say_so_whatever_its_check_found() {
    let disagreement = kept_folder_failure(DecisionToolError::SavedModelDisagrees {
        question_name: "mood".to_owned(),
        largest_gap: 0.5,
    });
    assert!(matches!(
        disagreement,
        DecisionToolError::SavedModelDisagrees { .. }
    ));
    assert!(disagreement.to_string().contains("the new folder is kept"));
    let daemon_stopped = kept_folder_failure(DecisionToolError::TrainingFailed(
        "the daemon stopped".to_owned(),
    ));
    assert!(matches!(
        daemon_stopped,
        DecisionToolError::SavedModelUnchecked(_)
    ));
}

#[test]
fn should_tell_the_person_an_unchecked_folder_is_kept() {
    let unchecked = DecisionToolError::SavedModelUnchecked("Astronomical was busy".to_owned());

    assert!(unchecked.to_string().contains("the new folder is kept"));
}
