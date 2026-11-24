//! Test lanes read from the workspace (issue #2473, Step 6).
//!
//! Each package declares the lanes it runs in under
//! `[package.metadata.astronomical] test-lanes`, and a lane's cargo command
//! is built here from those declarations and a fixed command shape. The cargo
//! aliases are `xtask lane-cargo NAME`, so adding a crate to a lane edits only
//! that crate's manifest, and a package that declares nothing fails instead of
//! being skipped.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

/// The marker that turns a `.cargo/config.toml` alias into a lane command.
pub(crate) const LANE_ALIAS_PREFIX: &str = "xtask lane-cargo ";
const STATUS_PREFIX: &str = "[test-lanes]";

/// One lane command: its alias name, the lane whose packages it takes, and
/// the cargo words before and after the package list.
pub(crate) struct LaneCommand {
    pub alias_name: &'static str,
    pub lane_name: &'static str,
    pub leading_words: &'static str,
    pub target_words: &'static str,
}

/// Every alias the lanes generate. The integration build and the three CI
/// archives share one lane, because cargo unifies features per command: a
/// different package list would recompile what the build step compiled.
pub(crate) const LANE_COMMANDS: &[LaneCommand] = &[
    LaneCommand {
        alias_name: "test-integration",
        lane_name: "integration",
        leading_words: "test --no-fail-fast",
        target_words: "--lib --test integration_tests --test hardware_journeys",
    },
    LaneCommand {
        alias_name: "nextest-integration",
        lane_name: "nextest",
        leading_words: "nextest run --no-fail-fast",
        target_words: "--lib --test integration_tests",
    },
    LaneCommand {
        alias_name: "nextest-archive-integration",
        lane_name: "integration",
        leading_words: "nextest archive --profile ci --archive-file target/nextest-archives/integration.tar.zst",
        target_words: "--lib --test integration_tests --test hardware_journeys",
    },
    LaneCommand {
        alias_name: "nextest-archive-rest-api",
        lane_name: "integration",
        leading_words: "nextest archive --profile ci --archive-file target/nextest-archives/rest-api.tar.zst",
        target_words: "--test rest_api_tests",
    },
    LaneCommand {
        alias_name: "nextest-archive-e2e",
        lane_name: "integration",
        leading_words: "nextest archive --profile ci --archive-file target/nextest-archives/e2e.tar.zst",
        target_words: "--test e2e",
    },
    LaneCommand {
        alias_name: "test-integration-and-rest",
        lane_name: "integration-and-rest",
        leading_words: "test --no-fail-fast",
        target_words: "--lib --test integration_tests --test hardware_journeys --test rest_api_tests",
    },
    LaneCommand {
        alias_name: "test-direct-mlx",
        lane_name: "direct-mlx",
        leading_words: "test --no-fail-fast",
        target_words: "--lib --test direct_mlx --test mlx_c_coverage_contract_tests -- --test-threads=1",
    },
];

/// One package's declaration: its lanes, and the feature the `direct-mlx`
/// lane turns on for it, when it names one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct LaneDeclaration {
    pub test_lanes: Vec<String>,
    pub direct_mlx_lane_feature: Option<String>,
}

/// Every workspace package's declaration, read from `cargo metadata`.
/// Refuses a package that declares no `test-lanes`, naming each one.
pub(crate) fn lane_declarations_from_metadata(
    metadata_json_text: &str,
) -> Result<BTreeMap<String, LaneDeclaration>, String> {
    let metadata_document: serde_json::Value = serde_json::from_str(metadata_json_text)
        .map_err(|error| format!("cargo metadata is not JSON: {error}"))?;
    let package_values = metadata_document
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .ok_or("cargo metadata has no packages")?;
    let mut lane_declarations = BTreeMap::new();
    let mut undeclared_package_names = Vec::new();
    for package_value in package_values {
        let package_name = package_value
            .get("name")
            .and_then(serde_json::Value::as_str)
            .ok_or("cargo metadata has a package without a name")?;
        match lane_declaration_from_package(package_value)? {
            Some(lane_declaration) => {
                lane_declarations.insert(package_name.to_owned(), lane_declaration);
            }
            None => undeclared_package_names.push(package_name.to_owned()),
        }
    }
    if !undeclared_package_names.is_empty() {
        return Err(format!(
            "packages declare no [package.metadata.astronomical] test-lanes: {}",
            undeclared_package_names.join(", ")
        ));
    }
    Ok(lane_declarations)
}

fn lane_declaration_from_package(
    package_value: &serde_json::Value,
) -> Result<Option<LaneDeclaration>, String> {
    let Some(astronomical_metadata) = package_value
        .get("metadata")
        .and_then(|metadata| metadata.get("astronomical"))
    else {
        return Ok(None);
    };
    let Some(lane_values) = astronomical_metadata
        .get("test-lanes")
        .and_then(serde_json::Value::as_array)
    else {
        return Ok(None);
    };
    let mut test_lanes = Vec::with_capacity(lane_values.len());
    for lane_value in lane_values {
        let lane_name = lane_value
            .as_str()
            .ok_or("a test-lanes entry is not text")?;
        if !LANE_COMMANDS
            .iter()
            .any(|lane_command| lane_command.lane_name == lane_name)
        {
            return Err(format!("test-lanes names an unknown lane: {lane_name}"));
        }
        test_lanes.push(lane_name.to_owned());
    }
    let direct_mlx_lane_feature = astronomical_metadata
        .get("direct-mlx-lane-feature")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    Ok(Some(LaneDeclaration {
        test_lanes,
        direct_mlx_lane_feature,
    }))
}

/// The full cargo command text of a lane alias, or None for an alias the
/// lanes do not generate.
pub(crate) fn lane_alias_text(
    alias_name: &str,
    lane_declarations: &BTreeMap<String, LaneDeclaration>,
) -> Option<String> {
    let lane_command = LANE_COMMANDS
        .iter()
        .find(|lane_command| lane_command.alias_name == alias_name)?;
    let mut alias_words = vec![lane_command.leading_words.to_owned()];
    for (package_name, lane_declaration) in lane_declarations {
        if !lane_declaration
            .test_lanes
            .iter()
            .any(|lane_name| lane_name == lane_command.lane_name)
        {
            continue;
        }
        alias_words.push(format!("-p {package_name}"));
        if lane_command.lane_name == "direct-mlx"
            && let Some(lane_feature) = &lane_declaration.direct_mlx_lane_feature
        {
            alias_words.push(format!("--features {package_name}/{lane_feature}"));
        }
    }
    alias_words.push(lane_command.target_words.to_owned());
    Some(alias_words.join(" "))
}

/// An alias's command text: a lane alias names no packages itself, so its
/// command is built from the packages' declared lanes, and narrowing sees the
/// same list cargo runs; any other alias is its own text.
pub(crate) fn resolved_alias_text(
    repository_root: &Path,
    alias_name: &str,
    configured_alias_text: Option<String>,
) -> Result<String, String> {
    let alias_text = configured_alias_text
        .ok_or_else(|| format!("alias {alias_name} is not in .cargo/config.toml"))?;
    if !alias_text.starts_with(LANE_ALIAS_PREFIX) {
        return Ok(alias_text);
    }
    let lane_declarations = workspace_lane_declarations(repository_root)?;
    lane_alias_text(alias_name, &lane_declarations)
        .ok_or_else(|| format!("no lane generates the alias {alias_name}"))
}

/// The lane commands built from this workspace's declarations.
pub(crate) fn workspace_lane_declarations(
    repository_root: &Path,
) -> Result<BTreeMap<String, LaneDeclaration>, String> {
    let metadata_output = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(repository_root)
        .output()
        .map_err(|error| format!("cannot run cargo metadata: {error}"))?;
    if !metadata_output.status.success() {
        return Err("cargo metadata failed".to_owned());
    }
    lane_declarations_from_metadata(&String::from_utf8_lossy(&metadata_output.stdout))
}

/// `cargo xtask lane-cargo ALIAS [ARGUMENT ...]`: runs the lane's cargo
/// command with any extra arguments after its own.
pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    let Some((alias_name, extra_arguments)) = arguments.split_first() else {
        return Err(format!(
            "{STATUS_PREFIX} usage: cargo xtask lane-cargo ALIAS [ARGUMENT ...]"
        ));
    };
    let lane_declarations = workspace_lane_declarations(repository_root)?;
    let alias_text = lane_alias_text(alias_name, &lane_declarations)
        .ok_or_else(|| format!("{STATUS_PREFIX} no lane generates the alias {alias_name}"))?;
    let mut cargo_arguments: Vec<String> =
        alias_text.split_whitespace().map(str::to_owned).collect();
    cargo_arguments.extend(extra_arguments.iter().cloned());
    eprintln!("{STATUS_PREFIX} alias={alias_name} status=start");
    let cargo_status = Command::new("cargo")
        .args(&cargo_arguments)
        .current_dir(repository_root)
        .status()
        .map_err(|error| format!("cannot run cargo: {error}"))?;
    if cargo_status.success() {
        Ok(())
    } else {
        Err(format!("{STATUS_PREFIX} alias={alias_name} status=failed"))
    }
}

#[cfg(test)]
#[path = "test_lanes_tests.rs"]
mod test_lanes_tests;
