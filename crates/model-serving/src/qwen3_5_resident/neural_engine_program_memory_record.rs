//! Remembers what a kept set of programs measured, so later loads need not measure.
//!
//! Measuring takes half a minute of GPU idle and a few seconds of the Neural Engine
//! (`neural_engine_program_measurement`). What it reads depends on the programs, the
//! Mac and the macOS build, so a record names all of these it can in its `measured_for`
//! line, and the weights it covered, which change when the output gates did not build.
//!
//! The reading is of the whole system, so another app wiring memory during it can inflate
//! it, and a kept reading is trusted until `measured_for` changes. So one reading is
//! never trusted: a record counts only once two readings on separate loads agree.

use std::fs;
use std::path::{Path, PathBuf};

const PROGRAM_MEMORY_RECORD_FILE_NAME: &str = "program-memory.txt";
const INCOMPLETE_RECORD_FILE_NAME: &str = ".program-memory.txt.incomplete";
const AGREED_MARK: &str = "agreed";
const FIRST_READING_MARK: &str = "first-reading";
/// Two clean readings of one set agreed within about 1 percent on every model measured,
/// so a sixteenth apart means something else was wiring memory during one of them.
const LARGEST_DISAGREEMENT_DIVISOR: u64 = 16;

fn program_memory_record_path(set_directory: &Path) -> PathBuf {
    set_directory.join(PROGRAM_MEMORY_RECORD_FILE_NAME)
}

/// The reading on file for this set here, and whether two readings have agreed on it.
/// A reading below the weights cannot be true, so it counts as damage.
fn reading_on_file(
    set_directory: &Path,
    measured_for: &str,
    weight_bytes: u64,
) -> Option<(u64, bool)> {
    let record = fs::read_to_string(program_memory_record_path(set_directory)).ok()?;
    let mut record_lines = record.lines();
    if record_lines.next()? != measured_for
        || record_lines.next()?.parse::<u64>().ok()? != weight_bytes
    {
        return None;
    }
    let wired_bytes = record_lines.next()?.parse::<u64>().ok()?;
    let is_agreed = match record_lines.next()? {
        AGREED_MARK => true,
        FIRST_READING_MARK => false,
        _ => return None,
    };
    (wired_bytes >= weight_bytes).then_some((wired_bytes, is_agreed))
}

/// What this set wired here, or `None` when it must be measured.
pub(crate) fn recorded_program_memory_bytes(
    set_directory: &Path,
    measured_for: &str,
    weight_bytes: u64,
) -> Option<u64> {
    match reading_on_file(set_directory, measured_for, weight_bytes)? {
        (wired_bytes, true) => Some(wired_bytes),
        (_, false) => None,
    }
}

const fn readings_agree(first_bytes: u64, second_bytes: u64) -> bool {
    first_bytes.abs_diff(second_bytes) <= second_bytes / LARGEST_DISAGREEMENT_DIVISOR
}

/// Keeps a settled reading. When it agrees with the one on file, the lower of the two
/// becomes the record; otherwise it replaces it and waits for another to agree. A record
/// that cannot be written costs the next load another measurement and nothing else, so
/// it is logged and never fails the load.
pub(crate) fn record_program_memory_bytes(
    set_directory: &Path,
    measured_for: &str,
    weight_bytes: u64,
    wired_bytes: u64,
) {
    let (kept_bytes, mark) = match reading_on_file(set_directory, measured_for, weight_bytes) {
        Some((previous_reading_bytes, _))
            if readings_agree(previous_reading_bytes, wired_bytes) =>
        {
            (previous_reading_bytes.min(wired_bytes), AGREED_MARK)
        }
        _ => (wired_bytes, FIRST_READING_MARK),
    };
    let record = format!("{measured_for}\n{weight_bytes}\n{kept_bytes}\n{mark}\n");
    let incomplete_path = set_directory.join(INCOMPLETE_RECORD_FILE_NAME);
    if let Err(write_error) = fs::write(&incomplete_path, record) {
        tracing::warn!(error = %write_error, "could not record what Neural Engine programs measured");
        return;
    }
    if let Err(rename_error) =
        fs::rename(&incomplete_path, program_memory_record_path(set_directory))
    {
        tracing::warn!(error = %rename_error, "could not record what Neural Engine programs measured");
    }
}

#[cfg(test)]
#[path = "neural_engine_program_memory_record_tests.rs"]
mod tests;
