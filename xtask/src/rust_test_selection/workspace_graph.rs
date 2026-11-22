//! The workspace as the test selection sees it: each package's folder, its
//! test targets, and the workspace packages it depends on through any kind of
//! dependency. Dev-dependencies count here, unlike the release gate's reader in
//! `model_loading_changed`: a crate's tests compile against them, so a change
//! to one can break another crate's tests.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;

/// One workspace package, with paths relative to the repository root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WorkspacePackage {
    pub relative_directory: String,
    pub test_target_names: BTreeSet<String>,
    pub local_dependency_names: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WorkspaceGraph {
    pub packages: BTreeMap<String, WorkspacePackage>,
}

impl WorkspaceGraph {
    /// Reads a `cargo metadata --no-deps --format-version 1` document.
    pub(crate) fn from_metadata(
        metadata_json_text: &str,
        repository_root: &Path,
    ) -> Result<Self, String> {
        let metadata_document: serde_json::Value = serde_json::from_str(metadata_json_text)
            .map_err(|error| format!("cargo metadata is not JSON: {error}"))?;
        let package_values = metadata_document
            .get("packages")
            .and_then(serde_json::Value::as_array)
            .ok_or("cargo metadata has no packages")?;
        let mut packages = BTreeMap::new();
        for package_value in package_values {
            let package_name = text_field(package_value, "name")?;
            let workspace_package = workspace_package_from_value(package_value, repository_root)?;
            packages.insert(package_name.to_owned(), workspace_package);
        }
        Ok(Self { packages })
    }

    /// The package whose folder holds `repository_path`, the deepest one when
    /// folders nest, or None for a path outside every package.
    pub(crate) fn owning_package(&self, repository_path: &str) -> Option<&str> {
        let mut owner: Option<(&str, usize)> = None;
        for (package_name, workspace_package) in &self.packages {
            let package_directory = workspace_package.relative_directory.as_str();
            let is_inside = package_directory.is_empty()
                || repository_path == package_directory
                || repository_path.starts_with(&format!("{package_directory}/"));
            let is_deeper = match owner {
                None => true,
                Some((_, owner_depth)) => package_directory.len() > owner_depth,
            };
            if is_inside && is_deeper {
                owner = Some((package_name.as_str(), package_directory.len()));
            }
        }
        match owner {
            Some((package_name, _)) => Some(package_name),
            None => None,
        }
    }

    /// `seed_package_names` and every workspace package that depends on any
    /// of them, directly or through others.
    pub(crate) fn with_dependents(
        &self,
        seed_package_names: &BTreeSet<String>,
    ) -> BTreeSet<String> {
        let mut dependents_by_package: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (package_name, workspace_package) in &self.packages {
            for dependency_name in &workspace_package.local_dependency_names {
                dependents_by_package
                    .entry(dependency_name.as_str())
                    .or_default()
                    .push(package_name.as_str());
            }
        }
        let mut affected_packages: BTreeSet<String> = BTreeSet::new();
        let mut visit_queue: VecDeque<&str> = VecDeque::new();
        for seed_package_name in seed_package_names {
            visit_queue.push_back(seed_package_name);
        }
        while let Some(package_name) = visit_queue.pop_front() {
            if !affected_packages.insert(package_name.to_owned()) {
                continue;
            }
            for dependent_name in dependents_by_package
                .get(package_name)
                .into_iter()
                .flatten()
            {
                visit_queue.push_back(dependent_name);
            }
        }
        affected_packages
    }
}

fn workspace_package_from_value(
    package_value: &serde_json::Value,
    repository_root: &Path,
) -> Result<WorkspacePackage, String> {
    let manifest_path = Path::new(text_field(package_value, "manifest_path")?);
    let package_directory = manifest_path.parent().ok_or("manifest without a folder")?;
    let relative_directory = package_directory
        .strip_prefix(repository_root)
        .map_err(|_| {
            format!(
                "package outside the repository: {}",
                package_directory.display()
            )
        })?
        .to_string_lossy()
        .into_owned();
    Ok(WorkspacePackage {
        relative_directory,
        test_target_names: test_target_names(package_value),
        local_dependency_names: local_dependency_names(package_value),
    })
}

/// Every target cargo can run as a test binary: integration tests, and the
/// library itself, whose name the caller never passes with `--test`.
fn test_target_names(package_value: &serde_json::Value) -> BTreeSet<String> {
    let mut test_target_names = BTreeSet::new();
    let targets = package_value
        .get("targets")
        .and_then(serde_json::Value::as_array);
    for target_value in targets.into_iter().flatten() {
        let kinds = target_value
            .get("kind")
            .and_then(serde_json::Value::as_array);
        let is_test_target = kinds.into_iter().flatten().any(is_test_kind);
        if let (true, Some(target_name)) = (
            is_test_target,
            target_value.get("name").and_then(serde_json::Value::as_str),
        ) {
            test_target_names.insert(target_name.to_owned());
        }
    }
    test_target_names
}

fn is_test_kind(kind_value: &serde_json::Value) -> bool {
    kind_value.as_str() == Some("test")
}

/// Path dependencies of every kind: workspace packages are the only local
/// dependencies, and dev-dependencies reach the package's tests.
fn local_dependency_names(package_value: &serde_json::Value) -> BTreeSet<String> {
    let mut dependency_names = BTreeSet::new();
    let dependencies = package_value
        .get("dependencies")
        .and_then(serde_json::Value::as_array);
    for dependency_value in dependencies.into_iter().flatten() {
        let is_local = dependency_value
            .get("path")
            .is_some_and(serde_json::Value::is_string);
        if let (true, Some(dependency_name)) = (
            is_local,
            dependency_value
                .get("name")
                .and_then(serde_json::Value::as_str),
        ) {
            dependency_names.insert(dependency_name.to_owned());
        }
    }
    dependency_names
}

fn text_field<'a>(json_value: &'a serde_json::Value, field_name: &str) -> Result<&'a str, String> {
    json_value
        .get(field_name)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("cargo metadata entry without {field_name}"))
}
