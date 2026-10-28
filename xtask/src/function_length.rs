//! Function-length budget report across every feature world.
//!
//! `clippy::too_many_lines` is pedantic and allow-by-default, so no routine
//! build ever reports it. This subcommand arms it for the duration of a report
//! run, deliberately keeps the repository's warnings-as-errors policy from
//! turning each finding into an aborted compile (which would truncate the
//! report at the first violating crate and hide every dependent crate behind
//! it), and aggregates all feature worlds into one deduplicated list.
//!
//! The budget itself (100 code lines) lives in clippy.toml. The lint is not
//! armed in `[workspace.lints]` yet: this lane exists to prove the report
//! stable across real work before the budget becomes a build gate.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

/// The exact clippy message this report aggregates; counting semantics belong
/// to clippy (code lines only — comments and blank lines are excluded).
const OVER_BUDGET_MESSAGE_MARKER: &str = "this function has too many lines (";

/// One cargo clippy invocation covering a feature world.
pub(crate) struct FunctionLengthLane {
    pub(crate) name: &'static str,
    pub(crate) cargo_arguments: Vec<&'static str>,
}

/// One over-budget function as clippy reported it.
///
/// Field order is the report's sort order: ascending by code lines (so the
/// caller can reverse into worst-first), then path and line for deterministic
/// output across runs.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct OverBudgetFunction {
    pub(crate) code_lines: u64,
    pub(crate) path: String,
    pub(crate) line: usize,
}

/// The feature worlds a report must cover, mirroring the test lanes that
/// compile them: the default world, the MLX direct-mlx world, and the
/// acceptance feature superset whose cfg-gated test sources no other world
/// compiles.
pub(crate) fn function_length_lanes() -> Vec<FunctionLengthLane> {
    vec![
        FunctionLengthLane {
            name: "default-world",
            cargo_arguments: vec!["--workspace", "--all-targets"],
        },
        FunctionLengthLane {
            name: "mlx-world",
            cargo_arguments: vec![
                "--workspace",
                "--all-targets",
                "--features",
                "astronomical-model-serving/direct-mlx,astronomical-runtime-integration/mlx,astronomical-jangtq2-qwen35/direct-mlx,astronomical-quantization/direct-mlx,astronomical-acceptance/direct-mlx",
            ],
        },
        FunctionLengthLane {
            name: "acceptance-feature-superset",
            cargo_arguments: vec![
                "-p",
                "astronomical-acceptance",
                "--features",
                "direct-mlx,serving-acceptance,expert-paging,performance-measurement,performance_throughput,mlx,mlx-memory-contract-probe",
                "--all-targets",
            ],
        },
    ]
}

/// Extracts over-budget findings from clippy's short-format output and returns
/// them worst-first, deduplicated by location.
///
/// The short format puts everything on one line:
/// `path:line:column: warning: this function has too many lines (142/100)`.
/// When the budget lint is denied, the same message arrives with an `error:`
/// prefix, so both prefixes resolve to the same location. Only that message
/// is aggregated; every other diagnostic passes through untouched for the
/// caller to treat as a genuine compile failure.
pub(crate) fn parse_over_budget_diagnostics(diagnostic_text: &str) -> Vec<OverBudgetFunction> {
    let mut findings_by_location: BTreeMap<(String, usize), OverBudgetFunction> = BTreeMap::new();
    for line in diagnostic_text.lines() {
        let Some(message_start) = line.find(OVER_BUDGET_MESSAGE_MARKER) else {
            continue;
        };
        let location_text = line[..message_start]
            .strip_suffix(": warning: ")
            .or_else(|| line[..message_start].strip_suffix(": error: "))
            .unwrap_or(&line[..message_start]);
        let mut location_segments = location_text.rsplitn(3, ':');
        let Some(column_text) = location_segments.next() else {
            continue;
        };
        let Some(line_text) = location_segments.next() else {
            continue;
        };
        let Some(path) = location_segments.next() else {
            continue;
        };
        if column_text.trim().parse::<usize>().is_err() {
            continue;
        }
        let Ok(line_number) = line_text.trim().parse::<usize>() else {
            continue;
        };
        let count_text = &line[message_start + OVER_BUDGET_MESSAGE_MARKER.len()..];
        let Some((code_lines_text, _)) = count_text.split_once('/') else {
            continue;
        };
        let Ok(code_lines) = code_lines_text.trim().parse::<u64>() else {
            continue;
        };
        let finding = OverBudgetFunction {
            code_lines,
            path: path.to_owned(),
            line: line_number,
        };
        findings_by_location
            .entry((finding.path.clone(), finding.line))
            .or_insert(finding);
    }
    let mut findings: Vec<OverBudgetFunction> = findings_by_location.into_values().collect();
    findings.sort();
    findings.reverse();
    findings
}

/// Test-side paths, by the same convention the repository's testing rules use:
/// acceptance and end-to-end verification trees, integration-test directories,
/// and beside-source test modules.
pub(crate) fn is_test_scope(path: &str) -> bool {
    path.starts_with("verification/")
        || path.contains("/tests/")
        || path.ends_with("_tests.rs")
        || path.ends_with("/tests.rs")
}

/// Reports the function-length budget across every feature world.
///
/// This is a report lane, not a gate: findings are printed and summarized, and
/// the command succeeds while debt exists. Only a failing cargo invocation —
/// a compile error, a missing tool — fails the lane.
pub(crate) fn run(repository_root: &Path) -> Result<(), String> {
    let started_at = Instant::now();
    let lanes = function_length_lanes();
    println!(
        "[function-length] status=start lanes={} root={}",
        lanes.len(),
        repository_root.display()
    );

    let mut findings_by_location: BTreeMap<(String, usize), OverBudgetFunction> = BTreeMap::new();
    for (lane_index, lane) in lanes.iter().enumerate() {
        let mut cargo_arguments: Vec<&str> = vec!["clippy", "--message-format=short"];
        cargo_arguments.extend(lane.cargo_arguments.iter().copied());
        cargo_arguments.extend(["--", "-W", "clippy::too_many_lines"]);
        let output = Command::new("cargo")
            .args(&cargo_arguments)
            .current_dir(repository_root)
            // The repository denies warnings via [build] warnings; this lane
            // downgrades that policy for the duration of the run so a finding
            // in one crate cannot abort the whole graph and hide the rest.
            .env("CARGO_BUILD_WARNINGS", "warn")
            .output()
            .map_err(|error| format!("[function-length] could not run cargo clippy: {error}"))?;
        if !output.status.success() {
            let standard_error = String::from_utf8_lossy(&output.stderr);
            return Err(format!(
                "[function-length] status=failed lane={} exit={} stderr_tail={}",
                lane.name,
                output.status,
                standard_error.lines().last().unwrap_or("<no stderr>")
            ));
        }
        let standard_error = String::from_utf8_lossy(&output.stderr);
        for finding in parse_over_budget_diagnostics(&standard_error) {
            findings_by_location
                .entry((finding.path.clone(), finding.line))
                .or_insert(finding);
        }
        println!(
            "[function-length] status=progress lane={}/{} name={} elapsed_seconds={}",
            lane_index + 1,
            lanes.len(),
            lane.name,
            started_at.elapsed().as_secs()
        );
    }

    let mut findings: Vec<OverBudgetFunction> = findings_by_location.into_values().collect();
    findings.sort();
    findings.reverse();
    // Count while printing, so the summary cannot drift from the listing.
    let mut production_count = 0usize;
    for finding in &findings {
        let scope = if is_test_scope(&finding.path) {
            "test"
        } else {
            production_count += 1;
            "production"
        };
        println!(
            "[function-length] over_budget path={} line={} code_lines={} scope={}",
            finding.path, finding.line, finding.code_lines, scope
        );
    }
    println!(
        "[function-length] status=success over_budget={} production={} tests={} elapsed_seconds={}",
        findings.len(),
        production_count,
        findings.len() - production_count,
        started_at.elapsed().as_secs()
    );
    Ok(())
}
