//! Test impact selection for the macOS Rust lanes: a pull request compiles
//! and runs the tests of the packages it can affect, not the whole workspace.
//!
//! Test binaries are executables, which sccache cannot store, so every run
//! compiled every package's tests from source: 149 of the macOS build's 280
//! compiler-seconds before this, for a change that touched one crate or only
//! the Swift apps. The default branch still runs everything; only pull
//! requests are narrowed, and anything the selection cannot place selects
//! everything (see `selection`).
//!
//! The classifier answers `all`, `none` or a space-separated package list
//! (`rust_test_selection`). The macOS build passes that answer to
//! `cargo xtask selected-cargo ALIAS`, which runs the alias with its packages
//! and test targets narrowed to the selection.

mod external_reads;
mod narrowed_alias;
mod selection;
mod workspace_graph;

#[cfg(test)]
mod external_reads_tests;
#[cfg(test)]
mod narrowed_alias_tests;
#[cfg(test)]
mod selection_tests;
#[cfg(test)]
pub(crate) mod workspace_graph_tests;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

pub(crate) use narrowed_alias::{alias_arguments, narrowed_alias_arguments};
pub(crate) use selection::{RustTestSelection, select_rust_tests};
pub(crate) use workspace_graph::WorkspaceGraph;

/// The package whose `rest_api_tests` target the REST lane runs.
const REST_CONTRACTS_PACKAGE: &str = "astronomical-daemon";
/// The package whose `e2e` target the end-to-end lane runs.
const E2E_JOURNEYS_PACKAGE: &str = "astronomical-e2e";

/// Which macOS Rust lanes a run needs, from the classifier's verdict and the
/// selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RustLaneRequirements {
    /// The macOS build and the unit and integration lanes that use its archive.
    pub rust_core: bool,
    pub rest_contracts: bool,
    pub e2e_journeys: bool,
}

pub(crate) fn rust_lane_requirements(
    macos_verification_required: bool,
    rust_test_selection: &RustTestSelection,
) -> RustLaneRequirements {
    let (rust_core, rest_contracts, e2e_journeys) = match rust_test_selection {
        RustTestSelection::All { .. } => (true, true, true),
        RustTestSelection::Packages(package_names) => (
            !package_names.is_empty(),
            package_names.contains(REST_CONTRACTS_PACKAGE),
            package_names.contains(E2E_JOURNEYS_PACKAGE),
        ),
    };
    RustLaneRequirements {
        rust_core: macos_verification_required && rust_core,
        rest_contracts: macos_verification_required && rest_contracts,
        e2e_journeys: macos_verification_required && e2e_journeys,
    }
}

/// The alias whose package list is the set of packages the macOS lanes test.
pub(crate) const LANE_PACKAGES_ALIAS: &str = "test-integration";
const STATUS_PREFIX: &str = "[rust-test-selection]";

/// The classifier's answer for one pull request, or `all` when the
/// repository cannot be read, since a missing answer must never skip tests.
pub(crate) fn rust_test_selection_for(
    repository_root: &Path,
    changed_paths: &[String],
    native_inputs_changed: bool,
) -> RustTestSelection {
    match read_selection_inputs(repository_root) {
        Ok((workspace_graph, external_readers, lane_package_names)) => select_rust_tests(
            changed_paths,
            &workspace_graph,
            &external_readers,
            &lane_package_names,
            native_inputs_changed,
        ),
        Err(read_error) => RustTestSelection::All {
            reason: format!("selection inputs unreadable: {read_error}"),
        },
    }
}

type SelectionInputs = (
    WorkspaceGraph,
    BTreeMap<String, BTreeSet<String>>,
    BTreeSet<String>,
);

fn read_selection_inputs(repository_root: &Path) -> Result<SelectionInputs, String> {
    let workspace_graph =
        WorkspaceGraph::from_metadata(&workspace_metadata(repository_root)?, repository_root)?;
    let known_repository_files = tracked_repository_files(repository_root)?;
    let external_readers =
        external_readers(repository_root, &workspace_graph, &known_repository_files);
    let lane_package_names = lane_package_names(repository_root)?;
    Ok((workspace_graph, external_readers, lane_package_names))
}

/// The answer as the classifier publishes it.
pub(crate) fn selection_output_value(rust_test_selection: &RustTestSelection) -> String {
    match rust_test_selection {
        RustTestSelection::All { .. } => "all".to_owned(),
        RustTestSelection::Packages(package_names) if package_names.is_empty() => "none".to_owned(),
        RustTestSelection::Packages(package_names) => {
            package_names.iter().cloned().collect::<Vec<_>>().join(" ")
        }
    }
}

/// Reads an answer back: `all`, `none` or a package list.
pub(crate) fn selection_from_output_value(output_value: &str) -> RustTestSelection {
    match output_value.trim() {
        "all" | "" => RustTestSelection::All {
            reason: "classifier selected all".to_owned(),
        },
        "none" => RustTestSelection::Packages(BTreeSet::new()),
        package_list => RustTestSelection::Packages(
            package_list.split_whitespace().map(str::to_owned).collect(),
        ),
    }
}

/// `cargo xtask selected-cargo --selection VALUE ALIAS [ARGUMENT ...]`:
/// runs the alias narrowed to the selection, or says why it runs nothing.
pub fn run_selected_cargo(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    let [
        selection_flag,
        selection_value,
        alias_name,
        extra_arguments @ ..,
    ] = arguments
    else {
        return Err(format!(
            "{STATUS_PREFIX} usage: cargo xtask selected-cargo --selection all|none|PACKAGES ALIAS [ARGUMENT ...]"
        ));
    };
    if selection_flag != "--selection" {
        return Err(format!(
            "{STATUS_PREFIX} expected --selection, got {selection_flag}"
        ));
    }
    let rust_test_selection = selection_from_output_value(selection_value);
    let cargo_arguments = match &rust_test_selection {
        RustTestSelection::All { .. } => {
            let mut cargo_arguments = vec![alias_name.clone()];
            cargo_arguments.extend(extra_arguments.iter().cloned());
            Some(cargo_arguments)
        }
        RustTestSelection::Packages(package_names) => {
            let workspace_graph = WorkspaceGraph::from_metadata(
                &workspace_metadata(repository_root)?,
                repository_root,
            )?;
            let alias_text = read_alias_text(repository_root, alias_name)?;
            narrowed_alias_arguments(
                &alias_text,
                extra_arguments,
                package_names,
                &workspace_graph,
            )
        }
    };
    let Some(cargo_arguments) = cargo_arguments else {
        eprintln!(
            "{STATUS_PREFIX} alias={alias_name} status=skipped reason=no-selected-package-has-these-targets"
        );
        return Ok(());
    };
    eprintln!("{STATUS_PREFIX} alias={alias_name} status=start selection={selection_value}");
    run_cargo(repository_root, &cargo_arguments)
}

fn run_cargo(repository_root: &Path, cargo_arguments: &[String]) -> Result<(), String> {
    let cargo_status = Command::new("cargo")
        .current_dir(repository_root)
        .args(cargo_arguments)
        .status()
        .map_err(|error| format!("{STATUS_PREFIX} cargo did not start: {error}"))?;
    if cargo_status.success() {
        Ok(())
    } else {
        Err(format!(
            "{STATUS_PREFIX} cargo {} failed: {cargo_status}",
            cargo_arguments.join(" ")
        ))
    }
}

/// The workspace's packages from the runner's stable cargo. The package
/// graph does not depend on the compiler, and the classifier's job has no
/// pinned toolchain installed; asking stable keeps a toolchain download off
/// the job every lane waits on.
fn workspace_metadata(repository_root: &Path) -> Result<String, String> {
    let metadata_output = Command::new("cargo")
        .env("RUSTUP_TOOLCHAIN", "stable")
        .args([
            "metadata",
            "--no-deps",
            "--format-version",
            "1",
            "--offline",
            "--manifest-path",
        ])
        .arg(repository_root.join("Cargo.toml"))
        .output()
        .map_err(|error| format!("cargo metadata did not start: {error}"))?;
    if !metadata_output.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&metadata_output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&metadata_output.stdout).into_owned())
}

fn tracked_repository_files(repository_root: &Path) -> Result<BTreeSet<String>, String> {
    let git_output = Command::new("git")
        .arg("-C")
        .arg(repository_root)
        .args(["ls-files", "-z"])
        .output()
        .map_err(|error| format!("git ls-files did not start: {error}"))?;
    if !git_output.status.success() {
        return Err("git ls-files failed".to_owned());
    }
    Ok(String::from_utf8_lossy(&git_output.stdout)
        .split('\0')
        .filter(|tracked_path| !tracked_path.is_empty())
        .map(str::to_owned)
        .collect())
}

/// Maps each path a package reads from outside its folder to its readers,
/// scanning every tracked Rust source of every package.
fn external_readers(
    repository_root: &Path,
    workspace_graph: &WorkspaceGraph,
    known_repository_files: &BTreeSet<String>,
) -> BTreeMap<String, BTreeSet<String>> {
    let mut readers_by_path: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for tracked_path in known_repository_files {
        if !tracked_path.ends_with(".rs") {
            continue;
        }
        let Some(package_name) = workspace_graph.owning_package(tracked_path) else {
            continue;
        };
        let Ok(source_text) = std::fs::read_to_string(repository_root.join(tracked_path)) else {
            continue;
        };
        let package_directory = &workspace_graph.packages[package_name].relative_directory;
        for read_path in external_reads::external_paths_read_by(
            package_directory,
            tracked_path,
            &source_text,
            known_repository_files,
        ) {
            readers_by_path
                .entry(read_path)
                .or_default()
                .insert(package_name.to_owned());
        }
    }
    readers_by_path
}

fn lane_package_names(repository_root: &Path) -> Result<BTreeSet<String>, String> {
    let alias_text = read_alias_text(repository_root, LANE_PACKAGES_ALIAS)?;
    let mut package_names = BTreeSet::new();
    let alias_tokens = alias_arguments(&alias_text);
    for token_pair in alias_tokens.windows(2) {
        if token_pair[0] == "-p" {
            package_names.insert(token_pair[1].clone());
        }
    }
    if package_names.is_empty() {
        return Err(format!("alias {LANE_PACKAGES_ALIAS} lists no packages"));
    }
    Ok(package_names)
}

/// The text of `alias_name` in `.cargo/config.toml`'s `[alias]` table, or,
/// for a lane alias, the command its lane builds.
fn read_alias_text(repository_root: &Path, alias_name: &str) -> Result<String, String> {
    let config_text = std::fs::read_to_string(repository_root.join(".cargo/config.toml"))
        .map_err(|error| format!("cannot read .cargo/config.toml: {error}"))?;
    crate::test_lanes::resolved_alias_text(
        repository_root,
        alias_name,
        narrowed_alias::alias_text_from_config(&config_text, alias_name),
    )
}
