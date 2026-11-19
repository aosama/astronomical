//! `cargo xtask model-loading-changed PREVIOUS_TAG` — decides whether a
//! release changes how models load, for the Sparkle release skill's round
//! trip gate.
//!
//! The gate used to diff a hand-written folder list, which went stale when
//! the serving crate was split (issue #2340): it kept naming folders after
//! they were renamed, which `git diff` ignores silently, and missed every new
//! crate until a review caught it. The list is
//! now derived from `cargo metadata`: every local crate the inference worker
//! reaches through normal or build dependencies, plus the few paths outside
//! that closure which still change what a model loads. Watching the whole
//! closure, contract crates included, is deliberate: a hand-picked subset is
//! what drifted, and the cost of watching too much is only an extra round
//! trip.
//!
//! The output keeps the line the skill already reads,
//! `MODEL_LOADING_UNCHANGED=0` when nothing watched changed and `=1` when
//! something did, so the skill's decision rule is unchanged.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;
use std::process::Command;

const STATUS_PREFIX: &str = "[model-loading-changed]";
const WORKER_PACKAGE_NAME: &str = "astronomical-inference-worker";
/// Paths outside the worker's crate closure whose change still changes how a
/// model loads: the root manifest pins dependency versions and the release
/// profile, the quantizer and vision tower write the artifacts the worker
/// reads, the native build tool and the third-party pins and patches build
/// the MLX the worker links, and the registry lists the supported models.
pub(crate) const ADDITIONAL_MODEL_LOADING_PATHS: &[&str] = &[
    "Cargo.toml",
    "crates/quantization",
    "crates/vision-tower",
    "crates/native-build-tool",
    "third-party/pins",
    "third-party/patches",
    "third-party/native-dependency-manifest.cmake",
    "registry",
];

/// One workspace package as the closure walk needs it: its folder relative
/// to the repository root and the local packages it depends on outside
/// tests.
#[derive(Debug)]
pub(crate) struct LocalPackage {
    pub(crate) relative_directory: String,
    pub(crate) local_dependency_names: BTreeSet<String>,
}

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    let previous_tag = parse_previous_tag(arguments)?;
    let metadata_json_text = workspace_metadata(repository_root)?;
    let watched_paths = model_loading_paths(&metadata_json_text, repository_root)?;
    println!(
        "{STATUS_PREFIX} status=start previous_tag={previous_tag} watched_paths={}",
        watched_paths.len()
    );
    for watched_path in &watched_paths {
        println!("{STATUS_PREFIX} watched_path={watched_path}");
    }
    let git_exit_code = Command::new("git")
        .arg("-C")
        .arg(repository_root)
        .args(["diff", "--quiet"])
        .arg(format!("{previous_tag}..HEAD"))
        .arg("--")
        .args(&watched_paths)
        .status()
        .map_err(|error| {
            format!("{STATUS_PREFIX} status=failed reason=git-did-not-start error={error}")
        })?
        .code();
    let unchanged_flag = unchanged_flag_from_git_exit_code(git_exit_code)?;
    println!("MODEL_LOADING_UNCHANGED={unchanged_flag}");
    Ok(())
}

/// Exactly one argument, the tag of the previous release.
pub(crate) fn parse_previous_tag(arguments: &[String]) -> Result<&str, String> {
    match arguments {
        [previous_tag] if !previous_tag.starts_with('-') && !previous_tag.is_empty() => {
            Ok(previous_tag.as_str())
        }
        _ => Err(format!(
            "{STATUS_PREFIX} status=failed reason=usage usage=\"cargo xtask model-loading-changed PREVIOUS_TAG\""
        )),
    }
}

/// `git diff --quiet` exits 0 when nothing changed and 1 when something did;
/// any other outcome (a bad tag, a signal) is a failure, never a verdict.
pub(crate) fn unchanged_flag_from_git_exit_code(git_exit_code: Option<i32>) -> Result<u8, String> {
    match git_exit_code {
        Some(0) => Ok(0),
        Some(1) => Ok(1),
        other_outcome => Err(format!(
            "{STATUS_PREFIX} status=failed reason=git-diff-failed exit={other_outcome:?}"
        )),
    }
}

fn workspace_metadata(repository_root: &Path) -> Result<String, String> {
    let metadata_output = Command::new("cargo")
        .args([
            "metadata",
            "--no-deps",
            "--format-version",
            "1",
            "--manifest-path",
        ])
        .arg(repository_root.join("Cargo.toml"))
        .output()
        .map_err(|error| {
            format!(
                "{STATUS_PREFIX} status=failed reason=cargo-metadata-did-not-start error={error}"
            )
        })?;
    if !metadata_output.status.success() {
        return Err(format!(
            "{STATUS_PREFIX} status=failed reason=cargo-metadata exit={} stderr={}",
            metadata_output.status,
            String::from_utf8_lossy(&metadata_output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&metadata_output.stdout).into_owned())
}

/// The sorted folders whose change means model loading changed: the worker's
/// local closure plus [`ADDITIONAL_MODEL_LOADING_PATHS`].
pub(crate) fn model_loading_paths(
    metadata_json_text: &str,
    repository_root: &Path,
) -> Result<Vec<String>, String> {
    let local_packages = local_packages_from_metadata(metadata_json_text, repository_root)?;
    if !local_packages.contains_key(WORKER_PACKAGE_NAME) {
        return Err(format!(
            "{STATUS_PREFIX} status=failed reason=worker-package-missing package={WORKER_PACKAGE_NAME}"
        ));
    }
    let mut watched_paths: BTreeSet<String> = BTreeSet::new();
    for additional_path in ADDITIONAL_MODEL_LOADING_PATHS {
        watched_paths.insert((*additional_path).to_owned());
    }
    let mut visit_queue: VecDeque<&str> = VecDeque::from([WORKER_PACKAGE_NAME]);
    let mut visited_package_names: BTreeSet<&str> = BTreeSet::new();
    while let Some(package_name) = visit_queue.pop_front() {
        if !visited_package_names.insert(package_name) {
            continue;
        }
        // A path dependency outside the workspace is missing from
        // `cargo metadata --no-deps`; skipping it would let a change there
        // pass unwatched, so the gate refuses instead.
        let Some(local_package) = local_packages.get(package_name) else {
            return Err(format!(
                "{STATUS_PREFIX} status=failed reason=local-dependency-not-in-workspace package={package_name}"
            ));
        };
        watched_paths.insert(local_package.relative_directory.clone());
        for dependency_name in &local_package.local_dependency_names {
            visit_queue.push_back(dependency_name.as_str());
        }
    }
    Ok(watched_paths.into_iter().collect())
}

/// Reads every workspace package from a `cargo metadata --no-deps` document.
/// Only path dependencies count as local, and only normal and build edges
/// count: a dev-dependency never reaches the shipped worker. An unknown
/// dependency kind fails closed, because it means cargo's format changed.
pub(crate) fn local_packages_from_metadata(
    metadata_json_text: &str,
    repository_root: &Path,
) -> Result<BTreeMap<String, LocalPackage>, String> {
    let metadata_document: serde_json::Value =
        serde_json::from_str(metadata_json_text).map_err(|error| {
            format!("{STATUS_PREFIX} status=failed reason=metadata-not-json error={error}")
        })?;
    let packages_array = metadata_document
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{STATUS_PREFIX} status=failed reason=metadata-without-packages"))?;
    let mut local_packages = BTreeMap::new();
    for package_value in packages_array {
        let (package_name, local_package) =
            local_package_from_value(package_value, repository_root)?;
        local_packages.insert(package_name, local_package);
    }
    Ok(local_packages)
}

fn local_package_from_value(
    package_value: &serde_json::Value,
    repository_root: &Path,
) -> Result<(String, LocalPackage), String> {
    let package_name = string_field(package_value, "name")?;
    let manifest_path = string_field(package_value, "manifest_path")?;
    let relative_directory = relative_package_directory(manifest_path, repository_root)?;
    let dependencies_array = package_value
        .get("dependencies")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{STATUS_PREFIX} status=failed reason=package-without-dependencies package={package_name}"))?;
    let mut local_dependency_names = BTreeSet::new();
    for dependency_value in dependencies_array {
        if let Some(dependency_name) =
            local_shipped_dependency_name(dependency_value, package_name)?
        {
            local_dependency_names.insert(dependency_name);
        }
    }
    Ok((
        package_name.to_owned(),
        LocalPackage {
            relative_directory,
            local_dependency_names,
        },
    ))
}

/// The dependency's package name when it is a path dependency reached by the
/// shipped build (normal or build kind); `None` for registry and dev edges.
fn local_shipped_dependency_name(
    dependency_value: &serde_json::Value,
    package_name: &str,
) -> Result<Option<String>, String> {
    let dependency_name = string_field(dependency_value, "name")?;
    // An absent or null kind is cargo's spelling of a normal dependency; any
    // other value it does not document means its format changed.
    let dependency_kind = match dependency_value.get("kind") {
        None | Some(serde_json::Value::Null) => "normal",
        Some(serde_json::Value::String(kind_text)) => kind_text.as_str(),
        Some(_) => "unreadable",
    };
    let is_shipped_edge = match dependency_kind {
        "normal" | "build" => true,
        "dev" => false,
        unexpected_kind => {
            return Err(format!(
                "{STATUS_PREFIX} status=failed reason=unknown-dependency-kind kind={unexpected_kind} edge={package_name}->{dependency_name}"
            ));
        }
    };
    let is_local = dependency_value.get("path").is_some();
    if is_shipped_edge && is_local {
        Ok(Some(dependency_name.to_owned()))
    } else {
        Ok(None)
    }
}

fn string_field<'document>(
    json_value: &'document serde_json::Value,
    field_name: &str,
) -> Result<&'document str, String> {
    json_value
        .get(field_name)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            format!(
                "{STATUS_PREFIX} status=failed reason=metadata-missing-field field={field_name}"
            )
        })
}

/// The package's folder relative to the repository root, with `/`
/// separators, the form `git diff` pathspecs take.
fn relative_package_directory(
    manifest_path: &str,
    repository_root: &Path,
) -> Result<String, String> {
    let package_directory = Path::new(manifest_path).parent().ok_or_else(|| {
        format!(
            "{STATUS_PREFIX} status=failed reason=manifest-without-folder manifest={manifest_path}"
        )
    })?;
    let relative_directory = package_directory.strip_prefix(repository_root).map_err(|_| {
        format!("{STATUS_PREFIX} status=failed reason=package-outside-repository manifest={manifest_path}")
    })?;
    Ok(relative_directory.to_string_lossy().replace('\\', "/"))
}
