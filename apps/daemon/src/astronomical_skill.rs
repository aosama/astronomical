//! Keeps the Astronomical skill on disk where agents can read it.
//!
//! The skill's source is `skill/astronomical` in this repository. It is
//! compiled into the engine and written to the instance's skill folder at
//! every start, so the copy on disk always matches the running release and
//! needs no installer, no harness configuration and no network (issue #2397).

use std::{
    fs, io,
    path::{Path, PathBuf},
};

/// The skill's folder name, which the Agent Skills format requires to equal
/// the skill's `name`.
pub const ASTRONOMICAL_SKILL_NAME: &str = "astronomical";
const SKILL_INDEX_FILE_NAME: &str = "SKILL.md";
const REFERENCES_FOLDER_NAME: &str = "references";

macro_rules! skill_file {
    ($relative_path:literal) => {
        (
            $relative_path,
            include_str!(concat!("../../../skill/astronomical/", $relative_path)),
        )
    };
}

/// Every file of the skill, as (path inside the skill folder, text). A test
/// fails when this list and the repository's skill folder differ.
pub(crate) const EMBEDDED_SKILL_FILES: [(&str, &str); 11] = [
    skill_file!("SKILL.md"),
    skill_file!("references/chat-and-pictures.md"),
    skill_file!("references/coding-tools.md"),
    skill_file!("references/decision-models.md"),
    skill_file!("references/getting-started.md"),
    skill_file!("references/images.md"),
    skill_file!("references/local-api.md"),
    skill_file!("references/models-and-memory.md"),
    skill_file!("references/settings.md"),
    skill_file!("references/speech.md"),
    skill_file!("references/troubleshooting.md"),
];

/// What bringing the skill folder up to date did.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SkillInstallOutcome {
    pub files_written: usize,
    pub stale_files_removed: usize,
}

/// The skill's index file inside an instance's skill folder: the path the
/// engine tells models to read.
#[must_use]
pub fn skill_index_path(skill_directory: &Path) -> PathBuf {
    skill_directory
        .join(ASTRONOMICAL_SKILL_NAME)
        .join(SKILL_INDEX_FILE_NAME)
}

/// Makes the skill folder match the compiled-in skill: writes each file whose
/// text differs or is missing, and removes reference files the skill no
/// longer has. A file someone altered is put back, so what a model reads is
/// always what this release shipped. Skill updates that arrive between
/// releases (issue #2397) will write a newer skill into this folder; when
/// they exist, this must keep a newer copy instead of replacing it.
pub fn install_astronomical_skill(skill_directory: &Path) -> io::Result<SkillInstallOutcome> {
    let skill_root = skill_directory.join(ASTRONOMICAL_SKILL_NAME);
    let references_directory = skill_root.join(REFERENCES_FOLDER_NAME);
    fs::create_dir_all(&references_directory)?;
    let mut files_written = 0;
    for (relative_path, embedded_text) in EMBEDDED_SKILL_FILES {
        let installed_path = skill_root.join(relative_path);
        if file_holds(&installed_path, embedded_text) {
            continue;
        }
        write_atomically(&installed_path, embedded_text)?;
        files_written += 1;
    }
    let stale_files_removed = remove_stale_reference_files(&references_directory)?;
    Ok(SkillInstallOutcome {
        files_written,
        stale_files_removed,
    })
}

fn file_holds(installed_path: &Path, expected_text: &str) -> bool {
    match fs::read(installed_path) {
        Ok(installed_bytes) => installed_bytes == expected_text.as_bytes(),
        Err(_) => false,
    }
}

/// Writes through a sibling file and a rename, so an agent reading the skill
/// while the engine starts never sees half a file.
fn write_atomically(installed_path: &Path, file_text: &str) -> io::Result<()> {
    let mut partial_file_name = installed_path
        .file_name()
        .map(std::ffi::OsStr::to_os_string)
        .unwrap_or_default();
    partial_file_name.push(format!(".partial-{}", std::process::id()));
    let partial_path = installed_path.with_file_name(partial_file_name);
    fs::write(&partial_path, file_text)?;
    let rename_result = fs::rename(&partial_path, installed_path);
    if rename_result.is_err() {
        // Best effort: the rename's own error is the one worth reporting.
        let _ = fs::remove_file(&partial_path);
    }
    rename_result
}

/// Removes files in the references folder that this release's skill does not
/// have, so a reference retired in an update does not linger.
fn remove_stale_reference_files(references_directory: &Path) -> io::Result<usize> {
    let mut stale_files_removed = 0;
    for directory_entry in fs::read_dir(references_directory)? {
        let directory_entry = directory_entry?;
        let file_name = directory_entry.file_name();
        let relative_path = format!("{REFERENCES_FOLDER_NAME}/{}", file_name.to_string_lossy());
        if is_embedded_skill_file(&relative_path) || !directory_entry.file_type()?.is_file() {
            continue;
        }
        fs::remove_file(directory_entry.path())?;
        stale_files_removed += 1;
    }
    Ok(stale_files_removed)
}

fn is_embedded_skill_file(relative_path: &str) -> bool {
    for (embedded_relative_path, _) in EMBEDDED_SKILL_FILES {
        if embedded_relative_path == relative_path {
            return true;
        }
    }
    false
}

#[cfg(test)]
#[path = "astronomical_skill_tests.rs"]
mod astronomical_skill_tests;
