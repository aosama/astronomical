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
#[derive(Clone, Debug)]
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

/// How many lines below the reported span start the function-declaration scan
/// reaches. The lint anchors its span on the attribute line when the function
/// carries attributes (the ratchet guarantees at least one) and on the
/// declaration line otherwise; observed anchors sit one to four lines above
/// the `fn` keyword, so nine lines of reach leaves generous headroom without
/// crossing into the function body.
const FUNCTION_DECLARATION_SCAN_DEPTH_LINES: usize = 9;

/// Resolves the function declaration nearest the reported span start. The
/// scan walks strictly downward from the reported line: comment lines cannot
/// carry a declaration and are skipped so a doc example such as
/// `// fn example` cannot be mistaken for the site's name.
pub(crate) fn function_name_near_span(
    source_lines: &[&str],
    reported_line: usize,
) -> Option<String> {
    for scan_offset in 0..=FUNCTION_DECLARATION_SCAN_DEPTH_LINES {
        let Some(declaration_line) = source_lines.get(reported_line - 1 + scan_offset) else {
            break;
        };
        let trimmed_line = declaration_line.trim_start();
        if trimmed_line.starts_with("//") {
            continue;
        }
        if let Some(function_name) = parse_function_declaration_name(declaration_line) {
            return Some(function_name);
        }
    }
    None
}

/// Extracts the identifier from a `fn name` declaration fragment. The text
/// before the keyword may be a signature prefix such as `pub`, `pub(crate)`,
/// or `unsafe`; the guard rejects identifiers that merely end in "fn".
pub(crate) fn parse_function_declaration_name(declaration_line: &str) -> Option<String> {
    let keyword_position = declaration_line.find("fn ")?;
    if keyword_position > 0 {
        let preceding_character = declaration_line[..keyword_position].chars().last()?;
        if preceding_character.is_alphanumeric() || preceding_character == '_' {
            return None;
        }
    }
    let name_text = &declaration_line[keyword_position + "fn ".len()..];
    let mut name_end_bytes = 0usize;
    for name_character in name_text.chars() {
        if name_character.is_alphanumeric() || name_character == '_' {
            name_end_bytes += name_character.len_utf8();
        } else {
            break;
        }
    }
    if name_end_bytes == 0 {
        return None;
    }
    Some(name_text[..name_end_bytes].to_owned())
}

/// Reads the reported source file and returns the nearest function name.
/// An unreadable file or an unresolved span resolves to "unknown" so one
/// missing file cannot fail the whole report.
pub(crate) fn resolve_function_name(
    repository_root: &Path,
    finding: &OverBudgetFunction,
) -> String {
    let source_text = std::fs::read_to_string(repository_root.join(&finding.path));
    let Ok(source_text) = source_text else {
        return "unknown".to_owned();
    };
    let source_lines: Vec<&str> = source_text.lines().collect();
    match function_name_near_span(&source_lines, finding.line) {
        Some(function_name) => function_name,
        None => "unknown".to_owned(),
    }
}

/// Merges per-lane findings, keeping the largest reported count per site.
/// Feature worlds cfg-gate different lines inside the same function, so one
/// location can report slightly different code-line counts across worlds; the
/// maximum is the honest worst case.
pub(crate) fn merge_world_findings_maximum(
    findings_per_world: Vec<Vec<OverBudgetFunction>>,
) -> Vec<OverBudgetFunction> {
    let mut largest_count_by_location: BTreeMap<(String, usize), OverBudgetFunction> =
        BTreeMap::new();
    for world_findings in findings_per_world {
        for finding in world_findings {
            let location_key = (finding.path.clone(), finding.line);
            match largest_count_by_location.get(&location_key) {
                Some(largest) if largest.code_lines >= finding.code_lines => {}
                _ => {
                    largest_count_by_location.insert(location_key, finding);
                }
            }
        }
    }
    // Field order on OverBudgetFunction makes sort+reverse worst-first with a
    // deterministic path-and-line tie-break, matching the function-length lane.
    let mut merged_findings: Vec<OverBudgetFunction> =
        largest_count_by_location.into_values().collect();
    merged_findings.sort();
    merged_findings.reverse();
    merged_findings
}

/// Runs one feature world with the budget lint force-armed and returns the
/// world's findings worst-first. `--force-warn` overrides every suppression
/// attribute — `#[expect]` and `#[allow]` included — so the caller observes
/// the true over-budget set, not merely the untracked one.
pub(crate) fn run_lane_with_forced_budget_lint(
    repository_root: &Path,
    lane_status_prefix: &str,
    world_cargo_arguments: &[&str],
) -> Result<Vec<OverBudgetFunction>, String> {
    let mut cargo_arguments: Vec<&str> = vec!["clippy", "--message-format=short"];
    cargo_arguments.extend(world_cargo_arguments.iter().copied());
    cargo_arguments.extend(["--", "--force-warn=clippy::too_many_lines"]);
    let clippy_run = Command::new("cargo")
        .args(&cargo_arguments)
        .current_dir(repository_root)
        // Same warnings-policy trap the function-length lane documents: the
        // repository denies warnings, and a denied finding would abort the
        // graph mid-report and hide every dependent crate behind it.
        .env("CARGO_BUILD_WARNINGS", "warn")
        .output()
        .map_err(|error| format!("{lane_status_prefix} could not run cargo clippy: {error}"))?;
    if !clippy_run.status.success() {
        let standard_error = String::from_utf8_lossy(&clippy_run.stderr);
        return Err(format!(
            "{lane_status_prefix} status=failed exit={} stderr_tail={}",
            clippy_run.status,
            standard_error.lines().last().unwrap_or("<no stderr>")
        ));
    }
    let standard_error = String::from_utf8_lossy(&clippy_run.stderr);
    Ok(parse_over_budget_diagnostics(&standard_error))
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

/// Reports the function-length budget across every feature world, or — when
/// invoked as `function-length --fail-on-production` — delegates to the hard
/// production-scope gate that no suppression attribute can bypass.
///
/// The report is deliberately not a gate: findings are printed and summarized,
/// and the command succeeds while debt exists. Only a failing cargo invocation —
/// a compile error, a missing tool — fails the lane. The gate owns the failing
/// behavior so the two cannot drift into each other.
pub(crate) fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    if arguments.is_empty() {
        return run_report(repository_root);
    }
    if arguments[0] == "--fail-on-production" {
        return crate::function_length_gate::run(repository_root, &arguments[1..]);
    }
    Err(format!(
        "[function-length] unknown argument {}; supported: --fail-on-production [--lanes NAME[,NAME...]]",
        arguments[0]
    ))
}

/// Reports the untracked over-budget set across every feature world.
fn run_report(repository_root: &Path) -> Result<(), String> {
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
