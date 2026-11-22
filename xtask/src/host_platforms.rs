//! Which workspace crates the host operating system can build, for the
//! repository checks that lint across crates.
//!
//! Some crates build only on macOS: their native code needs Metal, Core ML
//! or another Apple framework, or they depend on one that does. Each such
//! crate says so in its manifest:
//!
//! ```toml
//! [package.metadata.astronomical]
//! host-platforms = ["macos"]
//! ```
//!
//! A host the declaration does not list cannot build that crate, nor any
//! workspace crate depending on it, since Cargo would build the dependency
//! anyway. A crate without the declaration builds everywhere, and a crate
//! drops its declaration once it builds on the other hosts too.
//!
//! Every crate builds on Linux today, so no manifest declares it. The
//! mechanism stays because the hooks and lints already honour it: the next
//! crate that needs an Apple-only framework declares itself here instead of
//! breaking the Linux pre-commit hook.

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

use serde_json::Value;

/// The workspace packages the running host cannot build, sorted.
pub(crate) fn packages_unbuildable_on_this_host(
    repository_root: &Path,
) -> Result<Vec<String>, String> {
    let workspace_metadata = read_workspace_metadata(repository_root)?;
    packages_unbuildable_on_host(&workspace_metadata, std::env::consts::OS)
}

fn read_workspace_metadata(repository_root: &Path) -> Result<Value, String> {
    let metadata_output = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(repository_root)
        .output()
        .map_err(|spawn_error| format!("cargo metadata could not start: {spawn_error}"))?;
    if !metadata_output.status.success() {
        return Err(format!(
            "cargo metadata failed ({})",
            metadata_output.status
        ));
    }
    serde_json::from_slice(&metadata_output.stdout)
        .map_err(|parse_error| format!("cargo metadata printed unreadable JSON: {parse_error}"))
}

/// The workspace packages this host cannot build, sorted: those whose
/// manifest names other host platforms, and every package that depends on
/// one of them, directly or through another workspace package.
pub(crate) fn packages_unbuildable_on_host(
    workspace_metadata: &Value,
    host_operating_system: &str,
) -> Result<Vec<String>, String> {
    let Some(workspace_packages) = workspace_metadata["packages"].as_array() else {
        return Err("cargo metadata listed no packages".to_owned());
    };
    let mut unbuildable_package_names = BTreeSet::new();
    for workspace_package in workspace_packages {
        if !is_buildable_on_host(workspace_package, host_operating_system)? {
            unbuildable_package_names.insert(package_name(workspace_package)?.to_owned());
        }
    }
    let mut has_found_another_dependent = !unbuildable_package_names.is_empty();
    while has_found_another_dependent {
        has_found_another_dependent = false;
        for workspace_package in workspace_packages {
            let dependent_name = package_name(workspace_package)?;
            if !unbuildable_package_names.contains(dependent_name)
                && depends_on_any(workspace_package, &unbuildable_package_names)
            {
                unbuildable_package_names.insert(dependent_name.to_owned());
                has_found_another_dependent = true;
            }
        }
    }
    Ok(unbuildable_package_names.into_iter().collect())
}

/// A package without the declaration builds everywhere; one with it builds
/// only on the platforms it lists.
fn is_buildable_on_host(
    workspace_package: &Value,
    host_operating_system: &str,
) -> Result<bool, String> {
    let host_platforms = &workspace_package["metadata"]["astronomical"]["host-platforms"];
    if host_platforms.is_null() {
        return Ok(true);
    }
    let Some(listed_platforms) = host_platforms.as_array() else {
        return Err(format!(
            "{} declares host-platforms as something other than a list of platform names",
            package_name(workspace_package)?
        ));
    };
    for listed_platform in listed_platforms {
        if listed_platform.as_str() == Some(host_operating_system) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Workspace packages are path dependencies, so only those are followed.
fn depends_on_any(workspace_package: &Value, package_names: &BTreeSet<String>) -> bool {
    let Some(package_dependencies) = workspace_package["dependencies"].as_array() else {
        return false;
    };
    for package_dependency in package_dependencies {
        let is_path_dependency = package_dependency["path"].is_string();
        if let Some(dependency_name) = package_dependency["name"].as_str()
            && is_path_dependency
            && package_names.contains(dependency_name)
        {
            return true;
        }
    }
    false
}

fn package_name(workspace_package: &Value) -> Result<&str, String> {
    workspace_package["name"]
        .as_str()
        .ok_or_else(|| "cargo metadata listed a package without a name".to_owned())
}
