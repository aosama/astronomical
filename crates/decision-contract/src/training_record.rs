//! `decision_training_record.json`: what the evidence showed about a
//! calibrated or trained decision model folder, kept beside its weights.
//!
//! Calibration and training never refuse to write the folder a person asked
//! for; they report what the evidence shows and let the person decide what
//! to do with it. The verdict therefore has to travel with the folder: an
//! answer from a model judged worse, or never judged at all, must not look
//! like one from a model that beat the original on questions it never saw.
//! Every engine reads this one family-neutral file, and every answer the
//! folder gives carries it under `input.training`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The file a calibrated or trained folder carries beside its weights.
pub const DECISION_TRAINING_RECORD_FILE_NAME: &str = "decision_training_record.json";
/// The `format` value that marks the document as this file.
pub const DECISION_TRAINING_RECORD_FORMAT: &str = "astronomical-decision-training-record";
/// The only document version this build reads and writes.
pub const DECISION_TRAINING_RECORD_VERSION: u64 = 1;
/// The largest record read: a full document is a few hundred bytes plus its
/// warnings, so anything near this size is not one.
pub const MAXIMUM_DECISION_TRAINING_RECORD_FILE_BYTES: u64 = 64 * 1024;

/// What comparing the folder with the original model showed.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordedVerdict {
    /// Better than the original on questions held back from the fit.
    JudgedBetter,
    /// Worse than the original on at least one measure, on held-back questions.
    JudgedWorse,
    /// Too few different texts to hold any back, so it was only compared on
    /// the examples it learned from, which shows it learned them, not that
    /// it generalises.
    NotJudged,
}

impl RecordedVerdict {
    /// The verdict as the answer JSON and the report name it.
    #[must_use]
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::JudgedBetter => "judged_better",
            Self::JudgedWorse => "judged_worse",
            Self::NotJudged => "not_judged",
        }
    }
}

/// How a folder came to be, and what the evidence showed.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingRecord {
    pub verdict: RecordedVerdict,
    /// Labelled questions in the examples file.
    pub labelled_question_count: usize,
    /// Different texts (states) those questions were about.
    pub distinct_state_count: usize,
    /// Everything the report warned about, in the report's own words.
    pub warnings: Vec<String>,
}

impl TrainingRecord {
    /// This record for a folder made from one that already carried
    /// `earlier`, such as calibrating a trained folder. The less favourable
    /// verdict stands, because the earlier parts are still in the folder and
    /// what was found about them still holds; their warnings are kept,
    /// marked as earlier.
    #[must_use]
    pub fn after_earlier(self, earlier: Option<&Self>) -> Self {
        let Some(earlier) = earlier else {
            return self;
        };
        let mut warnings = Vec::with_capacity(earlier.warnings.len() + self.warnings.len());
        for earlier_warning in &earlier.warnings {
            warnings.push(format!("earlier: {earlier_warning}"));
        }
        warnings.extend(self.warnings);
        Self {
            verdict: less_favourable(earlier.verdict, self.verdict),
            labelled_question_count: self.labelled_question_count,
            distinct_state_count: self.distinct_state_count,
            warnings,
        }
    }
}

/// Judged worse outranks not judged, which outranks judged better.
fn less_favourable(first: RecordedVerdict, second: RecordedVerdict) -> RecordedVerdict {
    let concern_rank = |verdict: RecordedVerdict| match verdict {
        RecordedVerdict::JudgedBetter => 0,
        RecordedVerdict::NotJudged => 1,
        RecordedVerdict::JudgedWorse => 2,
    };
    if concern_rank(first) >= concern_rank(second) {
        first
    } else {
        second
    }
}

/// The document as written to disk: the record behind a format marker.
/// The fields are spelled out rather than flattened, because serde cannot
/// refuse unknown fields through a flattened struct.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TrainingRecordDocument {
    format: String,
    version: u64,
    verdict: RecordedVerdict,
    labelled_question_count: usize,
    distinct_state_count: usize,
    warnings: Vec<String>,
}

/// Why a record file cannot be used.
#[derive(Debug, Error)]
pub enum TrainingRecordError {
    #[error("{DECISION_TRAINING_RECORD_FILE_NAME} is not valid JSON of the expected shape: {0}")]
    Malformed(String),
    #[error(
        "{DECISION_TRAINING_RECORD_FILE_NAME} is not a training record (format {found:?}, version {version})"
    )]
    UnknownFormat { found: String, version: u64 },
}

/// Reads a record document.
pub fn parse_training_record(record_bytes: &[u8]) -> Result<TrainingRecord, TrainingRecordError> {
    let document: TrainingRecordDocument = match serde_json::from_slice(record_bytes) {
        Ok(document) => document,
        Err(parse_error) => return Err(TrainingRecordError::Malformed(parse_error.to_string())),
    };
    if document.format != DECISION_TRAINING_RECORD_FORMAT
        || document.version != DECISION_TRAINING_RECORD_VERSION
    {
        return Err(TrainingRecordError::UnknownFormat {
            found: document.format,
            version: document.version,
        });
    }
    Ok(TrainingRecord {
        verdict: document.verdict,
        labelled_question_count: document.labelled_question_count,
        distinct_state_count: document.distinct_state_count,
        warnings: document.warnings,
    })
}

/// The record as pretty JSON, ending in a newline, for a person to read.
#[must_use]
pub fn training_record_document(training_record: &TrainingRecord) -> Vec<u8> {
    let document = TrainingRecordDocument {
        format: DECISION_TRAINING_RECORD_FORMAT.to_owned(),
        version: DECISION_TRAINING_RECORD_VERSION,
        verdict: training_record.verdict,
        labelled_question_count: training_record.labelled_question_count,
        distinct_state_count: training_record.distinct_state_count,
        warnings: training_record.warnings.clone(),
    };
    // A struct of strings, counts and an enum always serializes.
    let mut document_bytes = serde_json::to_vec_pretty(&document).unwrap_or_default();
    document_bytes.push(b'\n');
    document_bytes
}

#[cfg(test)]
#[path = "training_record_tests.rs"]
mod tests;
