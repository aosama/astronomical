//! Which Rust test packages a pull request needs, from the paths it changed.
//!
//! The rule is fail-safe: a path is either placed (it belongs to a package,
//! a package reads it, or it is known not to reach Rust) or it selects every
//! package. Placed packages are widened to everything that depends on them,
//! then narrowed to the packages the macOS lanes test.

use std::collections::{BTreeMap, BTreeSet};

use super::external_reads::is_same_or_beneath;
use super::workspace_graph::WorkspaceGraph;
use crate::classify_scope::is_static_only_path;

/// The Swift and web apps: no Rust package lives here, and a file here
/// reaches Rust only when a package reads it, which the readers map shows.
const NON_RUST_APPLICATION_FOLDERS: &[&str] = &["apps/thin-talk", "apps/astronomical-menu"];

/// What a pull request's Rust tests must cover.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RustTestSelection {
    /// Every package; `reason` names the path or condition that required it.
    All { reason: String },
    /// Only these packages, possibly none.
    Packages(BTreeSet<String>),
}

/// What one changed path asks for.
#[derive(Debug, PartialEq, Eq)]
enum PathEffect {
    Packages(BTreeSet<String>),
    Ignored,
    Everything,
}

/// Selects the packages of `lane_package_names` that `changed_paths` can
/// affect. `external_readers` maps a path outside the packages to the
/// packages that read it.
pub(crate) fn select_rust_tests(
    changed_paths: &[String],
    workspace_graph: &WorkspaceGraph,
    external_readers: &BTreeMap<String, BTreeSet<String>>,
    lane_package_names: &BTreeSet<String>,
    native_inputs_changed: bool,
) -> RustTestSelection {
    if native_inputs_changed {
        return RustTestSelection::All {
            reason: "native inputs changed".to_owned(),
        };
    }
    let mut seed_package_names = BTreeSet::new();
    for changed_path in changed_paths {
        match path_effect(changed_path, workspace_graph, external_readers) {
            PathEffect::Packages(package_names) => seed_package_names.extend(package_names),
            PathEffect::Ignored => {}
            PathEffect::Everything => {
                return RustTestSelection::All {
                    reason: format!("unplaced path {changed_path}"),
                };
            }
        }
    }
    let affected_package_names = workspace_graph.with_dependents(&seed_package_names);
    RustTestSelection::Packages(
        affected_package_names
            .intersection(lane_package_names)
            .cloned()
            .collect(),
    )
}

fn path_effect(
    changed_path: &str,
    workspace_graph: &WorkspaceGraph,
    external_readers: &BTreeMap<String, BTreeSet<String>>,
) -> PathEffect {
    if is_manifest_or_workspace_setting(changed_path) {
        return PathEffect::Everything;
    }
    let mut package_names = readers_of(changed_path, external_readers);
    if let Some(owning_package) = workspace_graph.owning_package(changed_path) {
        package_names.insert(owning_package.to_owned());
    }
    if !package_names.is_empty() {
        return PathEffect::Packages(package_names);
    }
    if is_static_only_path(changed_path) || is_in_non_rust_application(changed_path) {
        return PathEffect::Ignored;
    }
    PathEffect::Everything
}

/// A manifest change can move features that every package in the same cargo
/// command shares, and the workspace settings reach every package, so either
/// selects everything.
fn is_manifest_or_workspace_setting(changed_path: &str) -> bool {
    changed_path == "Cargo.toml"
        || changed_path.ends_with("/Cargo.toml")
        || changed_path == "Cargo.lock"
        || changed_path == "rust-toolchain.toml"
        || changed_path.starts_with(".cargo/")
}

/// Packages that read `changed_path`, or a folder holding it, or a file
/// beneath it when the changed path is itself a folder.
fn readers_of(
    changed_path: &str,
    external_readers: &BTreeMap<String, BTreeSet<String>>,
) -> BTreeSet<String> {
    let mut reader_names = BTreeSet::new();
    for (read_path, package_names) in external_readers {
        if is_same_or_beneath(changed_path, read_path)
            || is_same_or_beneath(read_path, changed_path)
        {
            reader_names.extend(package_names.iter().cloned());
        }
    }
    reader_names
}

fn is_in_non_rust_application(changed_path: &str) -> bool {
    for application_folder in NON_RUST_APPLICATION_FOLDERS {
        if is_same_or_beneath(changed_path, application_folder) {
            return true;
        }
    }
    false
}
