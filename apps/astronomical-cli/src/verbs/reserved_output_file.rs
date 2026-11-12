//! A file a verb claims before it starts minutes of work, so a path that is
//! taken, sits in a missing folder, or cannot be written fails at once rather
//! than after the work has nowhere to go. Claiming uses `create_new`, so an
//! existing file is never overwritten, by this process or by a race with
//! another one. `image` and `speak` both save their results this way.

use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// A path this process created and now owns, still empty.
pub(crate) struct ReservedOutputFile {
    path: PathBuf,
    file: File,
}

/// Why a path could not be claimed.
#[derive(Debug)]
pub(crate) enum OutputReservationError {
    AlreadyExists,
    Unwritable { cause: String },
}

impl ReservedOutputFile {
    pub(crate) fn claim(output_path: &Path) -> Result<Self, OutputReservationError> {
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output_path)
        {
            Ok(file) => Ok(Self {
                path: output_path.to_path_buf(),
                file,
            }),
            Err(open_error) if open_error.kind() == ErrorKind::AlreadyExists => {
                Err(OutputReservationError::AlreadyExists)
            }
            Err(open_error) => Err(OutputReservationError::Unwritable {
                cause: open_error.to_string(),
            }),
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Writes the content and returns the path to print: absolute when the
    /// OS can say. A failed write removes the claim and returns the cause.
    pub(crate) fn fill(mut self, content_bytes: &[u8]) -> Result<PathBuf, String> {
        let write_result = match self.file.write_all(content_bytes) {
            Ok(()) => self.file.flush(),
            Err(write_error) => Err(write_error),
        };
        match write_result {
            Ok(()) => Ok(std::fs::canonicalize(&self.path).unwrap_or(self.path)),
            Err(write_error) => {
                self.discard();
                Err(write_error.to_string())
            }
        }
    }

    /// Removes a claimed file that will never be filled.
    pub(crate) fn discard(self) {
        let Self { path, file } = self;
        drop(file);
        // Best effort: the claim was ours and empty, so a failed removal
        // leaves at worst an empty file the user can see and delete.
        let _removal_result = std::fs::remove_file(path);
    }
}

/// `astronomical-<kind>-<unix seconds>.<extension>`: sorts by creation time
/// and does not collide across runs. Two runs in the same second collide on
/// purpose: the claim then refuses rather than overwrites.
pub(crate) fn timestamped_file_name(kind: &str, extension: &str) -> String {
    let unix_seconds = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration_since_epoch) => duration_since_epoch.as_secs(),
        Err(_clock_before_epoch) => 0,
    };
    format!("astronomical-{kind}-{unix_seconds}.{extension}")
}

#[cfg(test)]
#[path = "reserved_output_file_tests.rs"]
mod tests;
