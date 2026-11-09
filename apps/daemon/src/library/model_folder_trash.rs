//! Moves a downloaded model's folder to the Trash.
//!
//! Removal goes to the Trash rather than deleting outright, so a mistaken
//! click can be undone from Finder until the Trash is emptied. macOS 15 and
//! later ship `/usr/bin/trash`, which moves a path the way Finder does without
//! asking for Automation permission; on an older macOS the command is missing
//! and the caller reports that the model has to be removed in Finder instead.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use thiserror::Error;

const MACOS_TRASH_COMMAND_PATH: &str = "/usr/bin/trash";

/// Why a folder could not be moved to the Trash.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum ModelFolderTrashError {
    /// This macOS has no `trash` command (it arrived in macOS 15).
    #[error("this Mac cannot move folders to the Trash from Astronomical; use Finder instead")]
    Unavailable,
    /// The command ran and refused, or could not be started.
    #[error("the folder could not be moved to the Trash")]
    Failed,
}

/// Something that can move one folder to the Trash.
pub trait ModelFolderTrash: Send + Sync {
    /// Whether this Mac can move folders to the Trash from here. The catalog
    /// offers Remove from disk only when it can, so an older macOS shows the
    /// model's folder in Finder instead of a button that can only fail.
    fn is_available(&self) -> bool;

    fn move_to_trash(&self, model_folder: &Path) -> Result<(), ModelFolderTrashError>;
}

/// The macOS `trash` command.
pub struct MacTrashCommand {
    trash_command_path: PathBuf,
}

impl MacTrashCommand {
    #[must_use]
    pub fn new() -> Self {
        Self::at_path(PathBuf::from(MACOS_TRASH_COMMAND_PATH))
    }

    /// A command at another path, so tests can stand in for `trash` with a
    /// program that succeeds, fails, or leaves the folder behind.
    pub(crate) fn at_path(trash_command_path: PathBuf) -> Self {
        Self { trash_command_path }
    }
}

impl Default for MacTrashCommand {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelFolderTrash for MacTrashCommand {
    fn is_available(&self) -> bool {
        self.trash_command_path.is_file()
    }

    fn move_to_trash(&self, model_folder: &Path) -> Result<(), ModelFolderTrashError> {
        if !self.is_available() {
            return Err(ModelFolderTrashError::Unavailable);
        }
        let trash_output = Command::new(&self.trash_command_path)
            .arg(model_folder)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()
            .map_err(|spawn_error| {
                tracing::warn!(%spawn_error, "the trash command could not be started");
                ModelFolderTrashError::Failed
            })?;
        if !trash_output.status.success() {
            tracing::warn!(
                exit_status = %trash_output.status,
                stderr = %String::from_utf8_lossy(&trash_output.stderr).trim(),
                "the trash command refused to move the model folder"
            );
            return Err(ModelFolderTrashError::Failed);
        }
        // The command's exit status is not the promise this feature makes: the
        // page reports the gigabytes as freed, so the folder must be gone.
        if std::fs::symlink_metadata(model_folder).is_ok() {
            tracing::warn!("the trash command succeeded but the model folder is still there");
            return Err(ModelFolderTrashError::Failed);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "model_folder_trash_tests.rs"]
mod model_folder_trash_tests;
