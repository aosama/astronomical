//! The Rust files the source rules read, found once for every rule.

use std::fs;
use std::path::{Path, PathBuf};

use super::kernel_inventory::shown_path;

/// Every `.rs` file under the `searched_roots` folders of the repository, as
/// its repository-relative path and its path on disk, sorted. Build output
/// (`target` folders) and paths starting with one of `skipped_path_prefixes`
/// are left out.
pub(crate) fn rust_files_under(
    repository_root: &Path,
    searched_roots: &[&str],
    skipped_path_prefixes: &[&str],
) -> Vec<(String, PathBuf)> {
    let mut rust_files = Vec::new();
    for searched_root in searched_roots {
        collect_rust_files(
            repository_root,
            &repository_root.join(searched_root),
            skipped_path_prefixes,
            &mut rust_files,
        );
    }
    rust_files.sort();
    rust_files
}

fn collect_rust_files(
    repository_root: &Path,
    directory: &Path,
    skipped_path_prefixes: &[&str],
    rust_files: &mut Vec<(String, PathBuf)>,
) {
    let Ok(directory_entries) = fs::read_dir(directory) else {
        return;
    };
    for directory_entry in directory_entries.flatten() {
        let entry_path = directory_entry.path();
        let relative_path = shown_path(&entry_path, repository_root);
        if relative_path.ends_with("/target") || is_skipped(&relative_path, skipped_path_prefixes) {
            continue;
        }
        if entry_path.is_dir() {
            collect_rust_files(
                repository_root,
                &entry_path,
                skipped_path_prefixes,
                rust_files,
            );
        } else if relative_path.ends_with(".rs") {
            rust_files.push((relative_path, entry_path));
        }
    }
}

fn is_skipped(relative_path: &str, skipped_path_prefixes: &[&str]) -> bool {
    for skipped_path_prefix in skipped_path_prefixes {
        if relative_path.starts_with(skipped_path_prefix) {
            return true;
        }
    }
    false
}
