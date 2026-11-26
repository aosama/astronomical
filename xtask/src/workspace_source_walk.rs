//! Which files of the workspace this repository authors, for the checks that
//! read every source: `cargo xtask check-duplication` and the function-length
//! exemption scan of `cargo xtask function-length-budget`. Both walk the same
//! roots and skip the same folders, so a change to that scope is made here
//! once and is visible to both.

use std::fs;
use std::path::Path;

/// The top-level folders whose files the workspace authors.
pub(crate) const AUTHORED_ROOT_DIRECTORIES: [&str; 4] = ["crates", "apps", "verification", "xtask"];

/// Folders never descended into: build output, and vendored or pinned code
/// this repository does not author.
const SKIPPED_DIRECTORY_NAMES: [&str; 2] = ["target", "third-party"];

/// Whether the walk should skip a directory with this name.
pub(crate) fn is_skipped_directory_name(directory_name: &str) -> bool {
    SKIPPED_DIRECTORY_NAMES.contains(&directory_name)
}

/// Whether a file name is a Rust source.
pub(crate) fn is_rust_source_file_name(file_name: &str) -> bool {
    file_name.ends_with(".rs")
}

/// Every file under the authored roots whose name `is_wanted_file_name`
/// accepts, as repository-relative paths with `/` separators, sorted so a run
/// reads files in the same order on every machine.
pub(crate) fn collect_authored_file_paths(
    repository_root: &Path,
    is_wanted_file_name: fn(&str) -> bool,
) -> Vec<String> {
    let mut relative_paths = Vec::new();
    for root_directory in AUTHORED_ROOT_DIRECTORIES {
        collect_wanted_files(
            &repository_root.join(root_directory),
            root_directory,
            is_wanted_file_name,
            &mut relative_paths,
        );
    }
    relative_paths.sort();
    relative_paths
}

/// A directory that cannot be read is skipped rather than failing the run:
/// the authored roots hold only files this repository tracks, and an
/// unreadable folder there would be noticed by every other check first.
fn collect_wanted_files(
    directory: &Path,
    relative_directory: &str,
    is_wanted_file_name: fn(&str) -> bool,
    relative_paths: &mut Vec<String>,
) {
    let Ok(directory_entries) = fs::read_dir(directory) else {
        return;
    };
    for directory_entry in directory_entries.flatten() {
        let entry_name = directory_entry.file_name().to_string_lossy().into_owned();
        let relative_entry_path = format!("{relative_directory}/{entry_name}");
        let Ok(entry_type) = directory_entry.file_type() else {
            continue;
        };
        if entry_type.is_dir() {
            if !is_skipped_directory_name(&entry_name) {
                collect_wanted_files(
                    &directory_entry.path(),
                    &relative_entry_path,
                    is_wanted_file_name,
                    relative_paths,
                );
            }
        } else if is_wanted_file_name(&entry_name) {
            relative_paths.push(relative_entry_path);
        }
    }
}
