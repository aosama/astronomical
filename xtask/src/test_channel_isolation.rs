//! `cargo xtask check-test-channel-isolation` — fails when any package test
//! tree reaches for Stable state: the home-directory config loader, the
//! typed Stable instance, or a `~/.astronomical` / `.astronomical` state
//! path. Tests must use Development configuration or isolated
//! Development-shaped state, so a test run can never read or damage the
//! state of the Stable app someone is using. Typed port of the retired
//! `check-test-channel-isolation.sh`.
//!
//! The shell scanned `tests/*/tests`, a directory this repository does not
//! have, so the whole-product trees under `verification/` were never
//! checked; they are scanned here.

use std::path::{Path, PathBuf};

use crate::step_log::{log_line, start_step};

/// Literal forms of the shell's forbidden pattern
/// `load_from_home_directory\(|AstronomicalRuntimeInstance::Stable|~/\.astronomical("|/)|"\.astronomical("|/)`.
/// `.astronomical-dev` never matches: it is neither followed by a quote nor
/// by a slash directly after `.astronomical`.
const FORBIDDEN_FRAGMENTS: [&str; 6] = [
    "load_from_home_directory(",
    "AstronomicalRuntimeInstance::Stable",
    "~/.astronomical\"",
    "~/.astronomical/",
    "\".astronomical\"",
    "\".astronomical/",
];

/// A reviewed exception: this marker on the offending line or the line just
/// above it allows a Stable *label* that maps to the test's own isolated
/// state, such as a stub socket. The marker sits in the diff with its reason,
/// so every exception is visible in review.
pub(crate) const ALLOWANCE_MARKER: &str = "channel-isolation: allow";

/// Package test trees scanned, as `parent/*/tests` glob parents.
const TEST_TREE_PARENTS: [&str; 3] = ["apps", "crates", "verification"];

/// The config crate's tests load configuration from temporary home
/// directories and name the Stable instance deliberately: they test the
/// loader itself, never live Stable state.
const EXEMPT_TEST_TREE: &str = "crates/config/tests";

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    if !arguments.is_empty() {
        return Err("check-test-channel-isolation does not accept arguments".to_owned());
    }
    let isolation_step = start_step("test-channel-isolation");
    let violations = channel_isolation_violations(repository_root)?;
    if violations.is_empty() {
        isolation_step.finish("success");
        return Ok(());
    }
    for violation in &violations {
        eprintln!("{violation}");
    }
    isolation_step.finish("failed");
    Err(
        "Error: tests must use Development configuration or isolated Development-shaped state"
            .to_owned(),
    )
}

/// `path:line:text` for every forbidden reference in a scanned test tree.
pub(crate) fn channel_isolation_violations(repository_root: &Path) -> Result<Vec<String>, String> {
    let mut violations = Vec::new();
    let test_trees = scanned_test_trees(repository_root);
    log_line(&format!("test_trees={}", test_trees.len()));
    for test_tree in test_trees {
        let mut source_files = Vec::new();
        collect_rust_sources(&test_tree, &mut source_files)?;
        for source_file in source_files {
            collect_file_violations(repository_root, &source_file, &mut violations);
        }
    }
    Ok(violations)
}

pub(crate) fn scanned_test_trees(repository_root: &Path) -> Vec<PathBuf> {
    let mut test_trees = Vec::new();
    for tree_parent in TEST_TREE_PARENTS {
        let Ok(package_directories) = std::fs::read_dir(repository_root.join(tree_parent)) else {
            continue;
        };
        for package_directory in package_directories.flatten() {
            let test_tree = package_directory.path().join("tests");
            if test_tree.is_dir() && test_tree != repository_root.join(EXEMPT_TEST_TREE) {
                test_trees.push(test_tree);
            }
        }
    }
    test_trees.sort();
    test_trees
}

pub(crate) fn line_reaches_for_stable_state(source_line: &str) -> bool {
    for forbidden_fragment in FORBIDDEN_FRAGMENTS {
        if source_line.contains(forbidden_fragment) {
            return true;
        }
    }
    false
}

fn collect_file_violations(
    repository_root: &Path,
    source_file: &Path,
    violations: &mut Vec<String>,
) {
    let Ok(source_text) = std::fs::read_to_string(source_file) else {
        return;
    };
    let relative_path = source_file
        .strip_prefix(repository_root)
        .unwrap_or(source_file);
    let mut previous_line = "";
    for (line_index, source_line) in source_text.lines().enumerate() {
        let is_allowed =
            source_line.contains(ALLOWANCE_MARKER) || previous_line.contains(ALLOWANCE_MARKER);
        if line_reaches_for_stable_state(source_line) && !is_allowed {
            violations.push(format!(
                "{}:{}:{}",
                relative_path.display(),
                line_index + 1,
                source_line.trim()
            ));
        }
        previous_line = source_line;
    }
}

fn collect_rust_sources(
    directory_path: &Path,
    source_files: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let directory_entries = match std::fs::read_dir(directory_path) {
        Ok(directory_entries) => directory_entries,
        Err(read_error) => {
            return Err(format!(
                "could not scan {}: {read_error}",
                directory_path.display()
            ));
        }
    };
    for directory_entry in directory_entries.flatten() {
        let entry_path = directory_entry.path();
        let Ok(entry_type) = directory_entry.file_type() else {
            continue;
        };
        if entry_type.is_dir() {
            collect_rust_sources(&entry_path, source_files)?;
        } else if entry_type.is_file() && entry_path.extension().is_some_and(is_rust_extension) {
            source_files.push(entry_path);
        }
    }
    Ok(())
}

fn is_rust_extension(file_extension: &std::ffi::OsStr) -> bool {
    file_extension == "rs"
}
