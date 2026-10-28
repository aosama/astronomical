use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::arguments::{TestRunRequest, TestTarget};

#[derive(Debug, Default)]
pub(super) struct CargoArtifacts {
    pub tests: Vec<PathBuf>,
    pub binaries: BTreeMap<String, PathBuf>,
}

pub(super) fn test_build_arguments(request: &TestRunRequest) -> Vec<String> {
    let mut cargo_arguments = vec![
        "test".to_owned(),
        "--package".to_owned(),
        request.package.clone(),
        "--no-run".to_owned(),
        "--message-format=json".to_owned(),
    ];
    match &request.target {
        TestTarget::Library => cargo_arguments.push("--lib".to_owned()),
        TestTarget::Integration(target_name) => {
            cargo_arguments.extend(["--test".to_owned(), target_name.clone()]);
        }
        TestTarget::AllTests => cargo_arguments.push("--tests".to_owned()),
    }
    request.append_profile_arguments(&mut cargo_arguments);
    if let Some(features) = &request.features {
        cargo_arguments.extend(["--features".to_owned(), features.clone()]);
    }
    cargo_arguments
}

pub(super) fn workspace_binary_build_arguments(request: &TestRunRequest) -> Option<Vec<String>> {
    if !matches!(
        request.package.as_str(),
        "astronomical-acceptance" | "astronomical-e2e"
    ) {
        return None;
    }
    let mut cargo_arguments = vec![
        "build".to_owned(),
        "--package".to_owned(),
        "astronomical-daemon".to_owned(),
        "--package".to_owned(),
        "astronomical-inference-worker".to_owned(),
        "--bins".to_owned(),
        "--message-format=json".to_owned(),
    ];
    request.append_profile_arguments(&mut cargo_arguments);
    Some(cargo_arguments)
}

pub(super) fn parse_artifacts(
    cargo_output: &str,
    repository_root: &Path,
) -> Result<CargoArtifacts, String> {
    let mut artifacts = CargoArtifacts::default();
    for output_line in cargo_output.lines().filter(|line| line.starts_with('{')) {
        let cargo_message: Value = serde_json::from_str(output_line)
            .map_err(|error| format!("invalid Cargo artifact message: {error}"))?;
        if cargo_message["reason"] != "compiler-artifact" {
            continue;
        }
        let Some(executable_path) = cargo_message["executable"].as_str() else {
            continue;
        };
        let executable_path = PathBuf::from(executable_path);
        let executable_path = if executable_path.is_absolute() {
            executable_path
        } else {
            repository_root.join(executable_path)
        };
        let target_kinds = cargo_message["target"]["kind"]
            .as_array()
            .ok_or_else(|| "Cargo executable artifact is missing target kinds".to_owned())?;
        if cargo_message["profile"]["test"] == true {
            artifacts.tests.push(executable_path);
        } else if target_kinds.iter().any(|kind| kind == "bin") {
            let binary_name = cargo_message["target"]["name"]
                .as_str()
                .ok_or_else(|| "Cargo binary artifact is missing its target name".to_owned())?;
            if let Some(previous_executable) = artifacts
                .binaries
                .insert(binary_name.to_owned(), executable_path.clone())
                && previous_executable != executable_path
            {
                return Err(format!(
                    "Cargo reported conflicting executables for {binary_name}"
                ));
            }
        }
    }
    artifacts.tests.sort();
    artifacts.tests.dedup();
    Ok(artifacts)
}

pub(super) fn listed_test_names(listing_output: &str) -> Vec<String> {
    listing_output
        .lines()
        .filter_map(|output_line| output_line.strip_suffix(": test"))
        .map(str::to_owned)
        .collect()
}

pub(super) fn binary_digests(
    executable_paths: &BTreeMap<String, PathBuf>,
) -> Result<BTreeMap<String, String>, String> {
    let mut executable_digests = BTreeMap::new();
    for (binary_name, executable_path) in executable_paths {
        let mut executable_file = File::open(executable_path).map_err(|error| {
            format!("cannot inspect selected executable {binary_name}: {error}")
        })?;
        let executable_byte_count = executable_file
            .metadata()
            .map_err(|error| error.to_string())?
            .len();
        let mut digest = Sha256::new();
        let mut executable_bytes = [0_u8; 65_536];
        let mut digested_byte_count = 0_u64;
        let started_at = Instant::now();
        let mut last_report_at = started_at;
        loop {
            let read_byte_count = executable_file
                .read(&mut executable_bytes)
                .map_err(|error| error.to_string())?;
            if read_byte_count == 0 {
                break;
            }
            digest.update(&executable_bytes[..read_byte_count]);
            digested_byte_count += read_byte_count as u64;
            if last_report_at.elapsed() >= Duration::from_secs(5) {
                let remaining_byte_count =
                    executable_byte_count.saturating_sub(digested_byte_count);
                let estimated_remaining_seconds = started_at.elapsed().as_secs_f64()
                    * remaining_byte_count as f64
                    / digested_byte_count as f64;
                eprintln!(
                    "[test-run] stage=executable-identity binary={binary_name} completed_bytes={digested_byte_count} total_bytes={executable_byte_count} progress_percent={:.1} eta_seconds={estimated_remaining_seconds:.1}",
                    100.0 * digested_byte_count as f64 / executable_byte_count.max(1) as f64,
                );
                last_report_at = Instant::now();
            }
        }
        let executable_sha256 = digest
            .finalize()
            .iter()
            .map(|digest_byte| format!("{digest_byte:02x}"))
            .collect();
        executable_digests.insert(binary_name.clone(), executable_sha256);
    }
    Ok(executable_digests)
}

#[cfg(test)]
#[path = "cargo_artifacts_tests.rs"]
mod cargo_artifacts_tests;
