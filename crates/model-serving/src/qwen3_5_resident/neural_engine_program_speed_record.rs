//! Remembers how much a kept set of programs sped a model up on this Mac.
//!
//! Comparing the engines costs some of a model's first prompts a few percent, so the
//! answer is kept. A model the programs slow down then never builds or wires them at all.
//! What the answer depends on is named in the record's `measured_for` line, so a macOS
//! update, which changes how the Neural Engine runs them, asks the question again.

use std::fs;
use std::path::{Path, PathBuf};

use super::neural_engine_chunk_comparison::ChunkComparisonVerdict;

const PROGRAM_SPEED_RECORD_FILE_NAME: &str = "program-speed.txt";
const INCOMPLETE_RECORD_FILE_NAME: &str = ".program-speed.txt.incomplete";

/// Where one set's answer lives, kept with the programs so it can be written long after
/// the plan that found them has gone.
#[derive(Clone, Debug)]
pub(crate) struct ProgramSpeedRecord {
    set_directory: PathBuf,
    measured_for: String,
}

impl ProgramSpeedRecord {
    pub(crate) fn new(set_directory: &Path, measured_for: String) -> Self {
        Self {
            set_directory: set_directory.to_path_buf(),
            measured_for,
        }
    }

    /// The answer on file, or `None` when this Mac must compare the engines again. A gain
    /// that is not a positive number cannot be true, so it counts as damage.
    pub(crate) fn recorded_verdict(&self) -> Option<ChunkComparisonVerdict> {
        let record =
            fs::read_to_string(self.set_directory.join(PROGRAM_SPEED_RECORD_FILE_NAME)).ok()?;
        let mut record_lines = record.lines();
        if record_lines.next()? != self.measured_for {
            return None;
        }
        let gain = record_lines.next()?.parse::<f64>().ok()?;
        if gain.is_finite() && gain > 0.0 {
            Some(ChunkComparisonVerdict::from_gain(gain))
        } else {
            None
        }
    }

    /// A record that cannot be written costs the next load another comparison and nothing
    /// else, so it is logged and never fails anything.
    pub(crate) fn record_gain(&self, gain: f64) {
        let incomplete_path = self.set_directory.join(INCOMPLETE_RECORD_FILE_NAME);
        let record = format!("{}\n{gain}\n", self.measured_for);
        if let Err(write_error) = fs::write(&incomplete_path, record) {
            tracing::warn!(error = %write_error, "could not record how fast Neural Engine programs ran");
            return;
        }
        let record_path = self.set_directory.join(PROGRAM_SPEED_RECORD_FILE_NAME);
        if let Err(rename_error) = fs::rename(&incomplete_path, record_path) {
            tracing::warn!(error = %rename_error, "could not record how fast Neural Engine programs ran");
        }
    }
}

#[cfg(test)]
#[path = "neural_engine_program_speed_record_tests.rs"]
mod tests;
