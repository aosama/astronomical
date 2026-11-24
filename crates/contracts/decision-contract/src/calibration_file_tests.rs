//! Unit tests for reading and writing `decision_calibration.json`.

use super::*;
use crate::calibration::{MAXIMUM_APPLIED_TEMPERATURE, MINIMUM_APPLIED_TEMPERATURE};
use crate::test_support::assert_close;

fn choice_and_noul_override() -> CalibrationOverride {
    CalibrationOverride {
        choice: Some(KindCalibration {
            temperature: 1.4,
            range_temperatures: vec![("3-5".to_owned(), 1.75)],
            fitted_question_count: 120,
        }),
        score: None,
        noul: Some(KindCalibration {
            temperature: 2.25,
            range_temperatures: Vec::new(),
            fitted_question_count: 37,
        }),
    }
}

fn document_with_kinds(kinds_json: &str) -> Vec<u8> {
    format!(r#"{{"format":"astronomical-decision-calibration","version":1,"kinds":{kinds_json}}}"#)
        .into_bytes()
}

#[test]
fn should_read_back_exactly_the_calibration_it_wrote() {
    let document_text = calibration_document_text(&choice_and_noul_override());
    let read_override = parse_calibration_document(document_text.as_bytes()).unwrap();
    assert_eq!(read_override, choice_and_noul_override());
    assert!(document_text.ends_with('\n'));
}

#[test]
fn should_leave_a_kind_it_does_not_name_to_the_family() {
    let read_override = choice_and_noul_override();
    assert_eq!(
        read_override.temperature_for(DecisionQuestionKind::Score, 5),
        None
    );
}

#[test]
fn should_prefer_the_range_temperature_and_fall_back_to_the_kind_temperature() {
    let read_override = choice_and_noul_override();
    let in_fitted_range = read_override.temperature_for(DecisionQuestionKind::Choice, 4);
    let outside_fitted_range = read_override.temperature_for(DecisionQuestionKind::Choice, 12);
    assert_close(in_fitted_range.unwrap(), 1.75, 0.0);
    assert_close(outside_fitted_range.unwrap(), 1.4, 0.0);
}

#[test]
fn should_confine_temperatures_at_the_edges_of_the_applied_range() {
    let document = document_with_kinds(
        r#"{"choice":{"temperature":0.01},"score":{"temperature":50.0},"noul":{"temperature":0.5}}"#,
    );
    let read_override = parse_calibration_document(&document).unwrap();
    let choice_temperature = read_override.temperature_for(DecisionQuestionKind::Choice, 2);
    let score_temperature = read_override.temperature_for(DecisionQuestionKind::Score, 5);
    let noul_temperature = read_override.temperature_for(DecisionQuestionKind::Noul, 2);
    assert_close(
        choice_temperature.unwrap(),
        MINIMUM_APPLIED_TEMPERATURE,
        0.0,
    );
    assert_close(score_temperature.unwrap(), MAXIMUM_APPLIED_TEMPERATURE, 0.0);
    assert_close(noul_temperature.unwrap(), MINIMUM_APPLIED_TEMPERATURE, 0.0);
}

#[test]
fn should_refuse_a_document_of_another_format_or_version() {
    let wrong_format = br#"{"format":"laya","version":1,"kinds":{"noul":{"temperature":1.0}}}"#;
    let wrong_version = br#"{"format":"astronomical-decision-calibration","version":2,
        "kinds":{"noul":{"temperature":1.0}}}"#;
    assert_eq!(
        parse_calibration_document(wrong_format),
        Err(CalibrationFileError::UnknownFormat)
    );
    assert_eq!(
        parse_calibration_document(wrong_version),
        Err(CalibrationFileError::UnknownFormat)
    );
}

#[test]
fn should_refuse_text_that_is_not_json() {
    let parse_outcome = parse_calibration_document(b"Romeo, Romeo, wherefore art thou");
    assert!(matches!(
        parse_outcome,
        Err(CalibrationFileError::NotJson { .. })
    ));
}

#[test]
fn should_refuse_a_document_that_calibrates_no_kind() {
    let no_kinds_member = br#"{"format":"astronomical-decision-calibration","version":1}"#;
    assert_eq!(
        parse_calibration_document(no_kinds_member),
        Err(CalibrationFileError::NoKinds)
    );
    assert_eq!(
        parse_calibration_document(&document_with_kinds("{}")),
        Err(CalibrationFileError::NoKinds)
    );
}

#[test]
fn should_refuse_an_unknown_kind_name() {
    let document = document_with_kinds(r#"{"pick":{"temperature":1.0}}"#);
    assert_eq!(
        parse_calibration_document(&document),
        Err(CalibrationFileError::UnknownKind {
            kind_name: "pick".to_owned()
        })
    );
}

#[test]
fn should_refuse_a_missing_zero_or_non_numeric_kind_temperature() {
    for kind_json in [
        r#"{"noul":{}}"#,
        r#"{"noul":{"temperature":0.0}}"#,
        r#"{"noul":{"temperature":"warm"}}"#,
    ] {
        assert_eq!(
            parse_calibration_document(&document_with_kinds(kind_json)),
            Err(CalibrationFileError::InvalidKindTemperature { kind: "noul" }),
            "{kind_json} was accepted"
        );
    }
}

#[test]
fn should_refuse_a_range_name_the_lookup_could_never_match() {
    // A misspelled range would be silently ignored at serving time, which
    // is exactly the "stale setting masks the new fit" failure to avoid.
    let document = document_with_kinds(
        r#"{"choice":{"temperature":1.0,"temperature_by_options":{"3-6":1.5}}}"#,
    );
    assert_eq!(
        parse_calibration_document(&document),
        Err(CalibrationFileError::InvalidRangeTemperature {
            kind: "choice",
            range_name: "3-6".to_owned()
        })
    );
}

#[test]
fn should_accept_every_range_name_the_lookup_produces() {
    for option_count in [2, 3, 5, 6, 10, 11, 255] {
        let range_name = option_count_range_name(option_count);
        let document = document_with_kinds(&format!(
            r#"{{"choice":{{"temperature":1.0,"temperature_by_options":{{"{range_name}":1.5}}}}}}"#
        ));
        assert!(
            parse_calibration_document(&document).is_ok(),
            "{range_name} was refused"
        );
    }
}
