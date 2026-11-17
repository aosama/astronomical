//! Writing a finished training's tensors to a file for the command-line
//! tool, so the message that ends the training carries only the file's name
//! (`trained_tensors_file` in the decision contract says why).
//!
//! The file goes in the system's temporary folder, which the worker and the
//! command-line tool share as one user on one Mac; the tool reads it and
//! deletes it. A file whose result never reaches the tool (it gave up
//! waiting, or a message failed) is deleted where that is known, and each
//! new hand-over also clears ones older than a day, so a leak stays bounded
//! even in a sandboxed app's own temporary folder, which macOS never clears.
//! The App Store build ships no command-line tool, so it never trains.

use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use astronomical_decision_contract::decision_training_job::DecisionTrainingOutcome;
use astronomical_decision_contract::trained_tensors_file::{
    HANDOVER_FILE_PREFIX, HandedOverTensors, handover_file_name, handover_file_path,
    trained_tensors_file_bytes,
};

/// How old a hand-over file must be before a later training clears it: no
/// command-line tool waits a day for a result.
const STALE_HANDOVER_AGE: Duration = Duration::from_secs(24 * 60 * 60);
/// Readable and writable by the user alone: the weights are theirs.
const HANDOVER_FILE_MODE: u32 = 0o600;

/// The outcome with its tensors moved to a new file in `handover_directory`
/// and the file's name in their place; an outcome with no tensors goes as it
/// is, since there is nothing to write.
///
/// # Errors
/// Returns why the file could not be written; the training then fails with
/// that reason rather than sending tensors no message could carry.
pub(super) fn hand_over_trained_tensors(
    mut outcome: Box<DecisionTrainingOutcome>,
    handover_directory: &Path,
) -> Result<Box<DecisionTrainingOutcome>, String> {
    if outcome.trained_tensors.is_empty() && outcome.adapter_tensors.is_empty() {
        return Ok(outcome);
    }
    clear_stale_handover_files(handover_directory);
    let handed_over_tensors = HandedOverTensors {
        trained_tensors: std::mem::take(&mut outcome.trained_tensors),
        adapter_tensors: std::mem::take(&mut outcome.adapter_tensors),
    };
    let file_bytes = match trained_tensors_file_bytes(handed_over_tensors) {
        Ok(file_bytes) => file_bytes,
        Err(file_error) => return Err(file_error.to_string()),
    };
    let file_name = handover_file_name(std::process::id(), nanoseconds_since_epoch());
    let handover_path = handover_directory.join(&file_name);
    if let Err(write_error) = write_new_file(&handover_path, &file_bytes) {
        let _ = std::fs::remove_file(&handover_path);
        return Err(format!(
            "the trained weights could not be written to {}: {write_error}",
            handover_path.display()
        ));
    }
    outcome.trained_tensors_file = Some(file_name);
    Ok(outcome)
}

/// Writes `file_bytes` to a file that must not exist yet: `create_new`
/// refuses an existing name or a link planted under it.
fn write_new_file(file_path: &Path, file_bytes: &[u8]) -> std::io::Result<()> {
    let mut new_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(HANDOVER_FILE_MODE)
        .open(file_path)?;
    new_file.write_all(file_bytes)
}

/// Deletes the hand-over file named `handover_file_name`, when the message
/// carrying the name could not be sent, so no reader will ever come for it.
pub(super) fn discard_handover_file(handover_file_name: Option<&str>, handover_directory: &Path) {
    let Some(file_name) = handover_file_name else {
        return;
    };
    if let Some(handover_path) = handover_file_path(handover_directory, file_name) {
        let _ = std::fs::remove_file(handover_path);
    }
}

/// Removes hand-over files older than [`STALE_HANDOVER_AGE`]; a file that
/// cannot be read or removed is left, since clearing is housekeeping.
fn clear_stale_handover_files(handover_directory: &Path) {
    let Ok(folder_entries) = std::fs::read_dir(handover_directory) else {
        return;
    };
    for folder_entry in folder_entries.flatten() {
        let file_name = folder_entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            continue;
        };
        if !file_name.starts_with(HANDOVER_FILE_PREFIX) {
            continue;
        }
        if is_stale(&folder_entry)
            && let Some(stale_path) = handover_file_path(handover_directory, file_name)
        {
            let _ = std::fs::remove_file(stale_path);
        }
    }
}

/// Whether the entry was last written more than [`STALE_HANDOVER_AGE`] ago;
/// an entry whose age cannot be read is not.
fn is_stale(folder_entry: &std::fs::DirEntry) -> bool {
    let Ok(entry_metadata) = folder_entry.metadata() else {
        return false;
    };
    let Ok(modified) = entry_metadata.modified() else {
        return false;
    };
    match modified.elapsed() {
        Ok(age) => age > STALE_HANDOVER_AGE,
        Err(_) => false,
    }
}

/// Nanoseconds since the epoch, or 0 on a clock set before it; the process
/// id in the name still keeps it apart from another worker's.
fn nanoseconds_since_epoch() -> u128 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(since_epoch) => since_epoch.as_nanos(),
        Err(_) => 0,
    }
}

#[cfg(test)]
#[path = "trained_tensors_handover_writing_tests.rs"]
mod tests;
