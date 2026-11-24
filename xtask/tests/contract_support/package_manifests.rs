//! Every package manifest in the workspace, found on disk so the contracts
//! read the same declarations cargo does.

use std::path::{Path, PathBuf};

use super::repository_root;

/// The folders that hold workspace packages.
const PACKAGE_ROOTS: [&str; 3] = ["apps", "crates", "verification"];

/// Every package manifest under the package roots. Crates sit one folder per
/// layer, and family crates one folder deeper per kind, so the walk descends
/// until it reaches a folder holding a `Cargo.toml`.
pub(crate) fn package_manifest_paths() -> Vec<PathBuf> {
    let mut manifest_paths = Vec::new();
    for package_root in PACKAGE_ROOTS {
        collect_package_manifest_paths(&repository_root().join(package_root), &mut manifest_paths);
    }
    manifest_paths
}

fn collect_package_manifest_paths(folder: &Path, manifest_paths: &mut Vec<PathBuf>) {
    let manifest_path = folder.join("Cargo.toml");
    if manifest_path.is_file() {
        manifest_paths.push(manifest_path);
        return;
    }
    let Ok(directory_entries) = std::fs::read_dir(folder) else {
        return;
    };
    for directory_entry in directory_entries.flatten() {
        let entry_path = directory_entry.path();
        if entry_path.is_dir() && entry_path.file_name() != Some(std::ffi::OsStr::new("target")) {
            collect_package_manifest_paths(&entry_path, manifest_paths);
        }
    }
}
