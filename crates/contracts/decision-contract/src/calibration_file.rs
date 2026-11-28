//! `decision_calibration.json`: temperatures fitted on a person's own
//! labelled examples, which replace the ones a checkpoint shipped with.
//!
//! The file is family-neutral on purpose. Laya keeps its temperatures in
//! `config.json`, Kev keeps one in its settings file (`kev_decision.json`, or
//! `decision.json` in a published build), and Clef ships none,
//! so a calibration written into each family's own file would need three
//! writers and three readers, and would change files the checkpoint was
//! downloaded with. One file that every family reads keeps the checkpoint's
//! own files byte-identical and makes a calibrated copy easy to recognise.
//!
//! # Which temperature wins
//!
//! The file is organised by question kind, and a kind is all-or-nothing:
//!
//! * A kind the file names is decided by the file alone: its per-option-count
//!   temperature when it fitted one for that count, otherwise its kind
//!   temperature. The family's own per-option-count values are never
//!   consulted for that kind. Laya's training notes record the failure this
//!   prevents: an inherited `temperature_by_options` entry silently masked a
//!   newly fitted temperature.
//! * A kind the file leaves out keeps the family's own temperatures exactly.
//!   Calibration writes every kind its examples ask, whether or not the fit
//!   helped; what judging it showed is kept beside it in
//!   `decision_training_record.json`, so a kind is left out only when no
//!   example asked it.
//!
//! Parsing and writing live here, apart from any filesystem access, so the
//! command-line tool and every engine agree on the format.

use thiserror::Error;

use crate::calibration::{OPTION_COUNT_RANGE_NAMES, clamp_temperature, option_count_range_name};
use crate::question::DecisionQuestionKind;

/// The file a calibrated decision model folder carries beside its weights.
pub const DECISION_CALIBRATION_FILE_NAME: &str = "decision_calibration.json";
/// The `format` value that marks the document as this file.
pub const DECISION_CALIBRATION_FORMAT: &str = "astronomical-decision-calibration";
/// The only document version this build reads and writes.
pub const DECISION_CALIBRATION_VERSION: u64 = 1;
/// The largest calibration file read: a full document is well under 2 KiB,
/// so anything near this size is not one.
pub const MAXIMUM_DECISION_CALIBRATION_FILE_BYTES: u64 = 64 * 1024;

const ALL_QUESTION_KINDS: [DecisionQuestionKind; 3] = [
    DecisionQuestionKind::Choice,
    DecisionQuestionKind::Score,
    DecisionQuestionKind::Noul,
];

/// The fitted temperatures of one question kind.
#[derive(Clone, Debug, PartialEq)]
pub struct KindCalibration {
    /// Applied to every question of this kind whose option-count range has
    /// no temperature of its own.
    pub temperature: f64,
    /// `(option-count range name, temperature)`, such as `("3-5", 1.6)`.
    pub range_temperatures: Vec<(String, f64)>,
    /// How many labelled questions the fit used. Recorded so a person reading
    /// the file can see the evidence behind each number; engines ignore it.
    pub fitted_question_count: usize,
}

impl KindCalibration {
    /// The raw fitted temperature for a question with `option_count`
    /// options: its range's when one was fitted, otherwise the kind's.
    #[must_use]
    pub fn temperature_for_option_count(&self, option_count: usize) -> f64 {
        let wanted_range_name = option_count_range_name(option_count);
        for (range_name, range_temperature) in &self.range_temperatures {
            if range_name == wanted_range_name {
                return *range_temperature;
            }
        }
        self.temperature
    }
}

/// Every kind a calibration file decides; `None` leaves the family's own.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CalibrationOverride {
    pub choice: Option<KindCalibration>,
    pub score: Option<KindCalibration>,
    pub noul: Option<KindCalibration>,
}

impl CalibrationOverride {
    /// The calibrated temperature for one question, already confined to the
    /// applied range, or `None` when the file leaves this kind to the family.
    #[must_use]
    pub fn temperature_for(&self, kind: DecisionQuestionKind, option_count: usize) -> Option<f64> {
        let kind_calibration = self.kind_calibration(kind)?;
        Some(clamp_temperature(
            kind_calibration.temperature_for_option_count(option_count),
        ))
    }

    /// The calibration of one kind, when the file decides that kind.
    #[must_use]
    pub const fn kind_calibration(&self, kind: DecisionQuestionKind) -> Option<&KindCalibration> {
        match kind {
            DecisionQuestionKind::Choice => self.choice.as_ref(),
            DecisionQuestionKind::Score => self.score.as_ref(),
            DecisionQuestionKind::Noul => self.noul.as_ref(),
        }
    }

    /// Makes the file decide `kind` with `kind_calibration`.
    pub fn set_kind_calibration(
        &mut self,
        kind: DecisionQuestionKind,
        kind_calibration: KindCalibration,
    ) {
        *self.kind_calibration_slot(kind) = Some(kind_calibration);
    }

    fn kind_calibration_slot(
        &mut self,
        kind: DecisionQuestionKind,
    ) -> &mut Option<KindCalibration> {
        match kind {
            DecisionQuestionKind::Choice => &mut self.choice,
            DecisionQuestionKind::Score => &mut self.score,
            DecisionQuestionKind::Noul => &mut self.noul,
        }
    }

    /// Whether the file decides no kind at all, so writing it would change nothing.
    #[must_use]
    pub const fn decides_no_kind(&self) -> bool {
        self.choice.is_none() && self.score.is_none() && self.noul.is_none()
    }
}

/// Why a calibration document cannot be used.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum CalibrationFileError {
    #[error("{DECISION_CALIBRATION_FILE_NAME} is not JSON: {cause}")]
    NotJson { cause: String },
    #[error(
        "{DECISION_CALIBRATION_FILE_NAME} does not say it is a version {DECISION_CALIBRATION_VERSION} {DECISION_CALIBRATION_FORMAT} document"
    )]
    UnknownFormat,
    #[error(
        "{DECISION_CALIBRATION_FILE_NAME} must calibrate at least one question kind under `kinds`"
    )]
    NoKinds,
    #[error(
        "{DECISION_CALIBRATION_FILE_NAME} names {kind_name:?} under `kinds`, which is not choice, score or noul"
    )]
    UnknownKind { kind_name: String },
    #[error(
        "{DECISION_CALIBRATION_FILE_NAME} must give `kinds.{kind}.temperature` as a positive number"
    )]
    InvalidKindTemperature { kind: &'static str },
    #[error(
        "{DECISION_CALIBRATION_FILE_NAME} has a {kind} temperature for options {range_name:?}, which is not a positive number or not one of 2, 3-5, 6-10, 11+"
    )]
    InvalidRangeTemperature {
        kind: &'static str,
        range_name: String,
    },
}

/// The calibration a document holds.
pub fn parse_calibration_document(
    document_bytes: &[u8],
) -> Result<CalibrationOverride, CalibrationFileError> {
    let document: serde_json::Value = match serde_json::from_slice(document_bytes) {
        Ok(document) => document,
        Err(json_error) => {
            return Err(CalibrationFileError::NotJson {
                cause: json_error.to_string(),
            });
        }
    };
    let format_matches = document.get("format").and_then(serde_json::Value::as_str)
        == Some(DECISION_CALIBRATION_FORMAT);
    let version_matches = document.get("version").and_then(serde_json::Value::as_u64)
        == Some(DECISION_CALIBRATION_VERSION);
    if !format_matches || !version_matches {
        return Err(CalibrationFileError::UnknownFormat);
    }
    let Some(kind_members) = document.get("kinds").and_then(serde_json::Value::as_object) else {
        return Err(CalibrationFileError::NoKinds);
    };
    let mut calibration_override = CalibrationOverride::default();
    for (kind_name, kind_member) in kind_members {
        let Some(kind) = DecisionQuestionKind::from_wire_name(kind_name) else {
            return Err(CalibrationFileError::UnknownKind {
                kind_name: kind_name.clone(),
            });
        };
        *calibration_override.kind_calibration_slot(kind) =
            Some(parse_kind_calibration(kind, kind_member)?);
    }
    if calibration_override.decides_no_kind() {
        return Err(CalibrationFileError::NoKinds);
    }
    Ok(calibration_override)
}

/// The document for `calibration_override`, as pretty JSON text ending in a newline.
#[must_use]
pub fn calibration_document_text(calibration_override: &CalibrationOverride) -> String {
    let mut kind_members = serde_json::Map::new();
    for kind in ALL_QUESTION_KINDS {
        let Some(kind_calibration) = calibration_override.kind_calibration(kind) else {
            continue;
        };
        let mut range_members = serde_json::Map::new();
        for (range_name, range_temperature) in &kind_calibration.range_temperatures {
            range_members.insert(
                range_name.clone(),
                serde_json::Value::from(*range_temperature),
            );
        }
        kind_members.insert(
            kind.wire_name().to_owned(),
            serde_json::json!({
                "temperature": kind_calibration.temperature,
                "temperature_by_options": range_members,
                "fitted_question_count": kind_calibration.fitted_question_count,
            }),
        );
    }
    let document = serde_json::json!({
        "format": DECISION_CALIBRATION_FORMAT,
        "version": DECISION_CALIBRATION_VERSION,
        "kinds": kind_members,
    });
    // A value built only from strings, finite numbers and maps always
    // serializes; the empty fallback keeps this function infallible.
    let mut document_text = serde_json::to_string_pretty(&document).unwrap_or_default();
    document_text.push('\n');
    document_text
}

fn parse_kind_calibration(
    kind: DecisionQuestionKind,
    kind_member: &serde_json::Value,
) -> Result<KindCalibration, CalibrationFileError> {
    let raw_temperature = kind_member
        .get("temperature")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(f64::NAN);
    if !is_usable_temperature(raw_temperature) {
        return Err(CalibrationFileError::InvalidKindTemperature {
            kind: kind.wire_name(),
        });
    }
    let fitted_question_count = kind_member
        .get("fitted_question_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    Ok(KindCalibration {
        temperature: raw_temperature,
        range_temperatures: parse_range_temperatures(
            kind,
            kind_member.get("temperature_by_options"),
        )?,
        fitted_question_count: usize::try_from(fitted_question_count).unwrap_or(usize::MAX),
    })
}

fn parse_range_temperatures(
    kind: DecisionQuestionKind,
    range_member: Option<&serde_json::Value>,
) -> Result<Vec<(String, f64)>, CalibrationFileError> {
    let Some(range_member) = range_member else {
        return Ok(Vec::new());
    };
    let Some(range_fields) = range_member.as_object() else {
        return Err(CalibrationFileError::InvalidRangeTemperature {
            kind: kind.wire_name(),
            range_name: "temperature_by_options".to_owned(),
        });
    };
    let mut range_temperatures = Vec::with_capacity(range_fields.len());
    for (range_name, raw_temperature) in range_fields {
        let temperature = raw_temperature.as_f64().unwrap_or(f64::NAN);
        // A range name the lookup can never produce would be silently
        // ignored at serving time, so it is refused here instead.
        if !is_usable_temperature(temperature)
            || !OPTION_COUNT_RANGE_NAMES.contains(&range_name.as_str())
        {
            return Err(CalibrationFileError::InvalidRangeTemperature {
                kind: kind.wire_name(),
                range_name: range_name.clone(),
            });
        }
        range_temperatures.push((range_name.clone(), temperature));
    }
    Ok(range_temperatures)
}

fn is_usable_temperature(temperature: f64) -> bool {
    temperature.is_finite() && temperature > 0.0
}

#[cfg(test)]
#[path = "calibration_file_tests.rs"]
mod tests;
