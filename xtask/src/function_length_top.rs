//! Ranked function-length report: names the worst over-budget functions.
//!
//! The `function-length` lane reports only untracked debt: every pre-existing
//! over-budget function carries the expect ratchet attribute, and a
//! warning-level lint arm never fires through a fulfilled expectation. This
//! sibling lane answers "what are the worst offenders" in one command: it
//! re-runs the same three feature worlds with `--force-warn`, which overrides
//! expectations, resolves each reported site's function name from the source,
//! and prints a worst-first ranked table.
//!
//! Like `function-length`, this is a report lane, not a gate: it succeeds
//! while tracked debt exists and fails only when cargo itself fails.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use crate::function_length::{
    OverBudgetFunction, function_length_lanes, is_test_scope, parse_over_budget_diagnostics,
};

/// The attribute this campaign counts as tracked debt. Every pre-existing
/// over-budget function carries one; the ratchet deletes markers
/// automatically when a function shrinks below budget, so the count is a
/// monotonic paydown gauge.
pub(crate) const TRACKED_EXPECT_MARKER: &str = "expect(clippy::too_many_lines)";

/// How many lines below the reported span start the function-declaration scan
/// reaches. The lint anchors its span on the attribute line when the function
/// carries attributes (the ratchet guarantees at least one) and on the
/// declaration line otherwise; observed anchors sit one to four lines above
/// the `fn` keyword, so nine lines of reach leaves generous headroom without
/// crossing into the function body.
const FUNCTION_DECLARATION_SCAN_DEPTH_LINES: usize = 9;

/// How many ranked entries the report shows when the caller does not choose.
const DEFAULT_RANKED_COUNT: usize = 10;

/// Which part of the tree a ranked entry must come from.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum ScopeFilter {
    All,
    Production,
    Tests,
}

impl ScopeFilter {
    pub(crate) fn parse(scope_text: &str) -> Option<ScopeFilter> {
        match scope_text {
            "all" => Some(ScopeFilter::All),
            "production" => Some(ScopeFilter::Production),
            "tests" => Some(ScopeFilter::Tests),
            _ => None,
        }
    }

    pub(crate) fn label(&self) -> &'static str {
        match self {
            ScopeFilter::All => "all",
            ScopeFilter::Production => "production",
            ScopeFilter::Tests => "tests",
        }
    }

    pub(crate) fn allows(&self, source_path: &str) -> bool {
        match self {
            ScopeFilter::All => true,
            ScopeFilter::Production => !is_test_scope(source_path),
            ScopeFilter::Tests => is_test_scope(source_path),
        }
    }
}

/// The report's caller-supplied knobs.
pub(crate) struct RankedReportArguments {
    pub(crate) ranked_count: usize,
    pub(crate) scope_filter: ScopeFilter,
}

/// Parses `--top N` (default 10, 0 shows every finding) and
/// `--scope production|tests|all` (default all). Unknown input fails the lane
/// rather than being silently ignored.
pub(crate) fn parse_ranked_report_arguments(
    arguments: &[String],
) -> Result<RankedReportArguments, String> {
    let mut parsed = RankedReportArguments {
        ranked_count: DEFAULT_RANKED_COUNT,
        scope_filter: ScopeFilter::All,
    };
    let mut argument_index = 0usize;
    while argument_index < arguments.len() {
        match arguments[argument_index].as_str() {
            "--top" => {
                let Some(count_text) = arguments.get(argument_index + 1) else {
                    return Err("[function-length-top] --top requires a count".to_owned());
                };
                let Ok(ranked_count) = count_text.parse::<usize>() else {
                    return Err(format!(
                        "[function-length-top] --top requires a number, found {count_text}"
                    ));
                };
                parsed.ranked_count = ranked_count;
                argument_index += 2;
            }
            "--scope" => {
                let Some(scope_text) = arguments.get(argument_index + 1) else {
                    return Err(
                        "[function-length-top] --scope requires production|tests|all".to_owned(),
                    );
                };
                let Some(scope_filter) = ScopeFilter::parse(scope_text) else {
                    return Err(format!(
                        "[function-length-top] unknown scope {scope_text}; expected production|tests|all"
                    ));
                };
                parsed.scope_filter = scope_filter;
                argument_index += 2;
            }
            unknown_argument => {
                return Err(format!(
                    "[function-length-top] unknown argument {unknown_argument}; supported: --top N, --scope production|tests|all"
                ));
            }
        }
    }
    Ok(parsed)
}

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
fn resolve_function_name(repository_root: &Path, finding: &OverBudgetFunction) -> String {
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

/// Merges per-world findings, keeping the largest reported count per site.
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
/// world's findings worst-first.
fn run_world_with_forced_budget_lint(
    repository_root: &Path,
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
        .map_err(|error| format!("[function-length-top] could not run cargo clippy: {error}"))?;
    if !clippy_run.status.success() {
        let standard_error = String::from_utf8_lossy(&clippy_run.stderr);
        return Err(format!(
            "[function-length-top] status=failed exit={} stderr_tail={}",
            clippy_run.status,
            standard_error.lines().last().unwrap_or("<no stderr>")
        ));
    }
    let standard_error = String::from_utf8_lossy(&clippy_run.stderr);
    Ok(parse_over_budget_diagnostics(&standard_error))
}

/// Gathers every repository Rust source below the root, skipping build output
/// and hidden trees: their content is not repository source and would distort
/// the tracked-debt gauge.
fn collect_rust_source_paths(directory: &Path, source_paths: &mut Vec<PathBuf>) {
    let Ok(directory_entries) = std::fs::read_dir(directory) else {
        return;
    };
    for directory_entry in directory_entries.flatten() {
        let Ok(entry_kind) = directory_entry.file_type() else {
            continue;
        };
        let entry_file_name = directory_entry.file_name();
        let entry_name = entry_file_name.to_string_lossy().into_owned();
        if entry_kind.is_dir() {
            let is_build_output = entry_name == "target";
            let is_hidden_tree = entry_name.starts_with('.');
            if is_build_output || is_hidden_tree {
                continue;
            }
            collect_rust_source_paths(&directory_entry.path(), source_paths);
        } else if entry_kind.is_file() && entry_name.ends_with(".rs") {
            source_paths.push(directory_entry.path());
        }
    }
}

/// Counts the tracked-debt markers across every repository Rust source. This
/// is the campaign gauge quoted beside every ranked report: the count only
/// ever shrinks.
pub(crate) fn count_tracked_expect_markers(repository_root: &Path) -> u64 {
    let mut source_paths: Vec<PathBuf> = Vec::new();
    collect_rust_source_paths(repository_root, &mut source_paths);
    let mut marker_count = 0u64;
    for source_path in &source_paths {
        let Ok(source_text) = std::fs::read_to_string(source_path) else {
            continue;
        };
        for source_line in source_text.lines() {
            if source_line.contains(TRACKED_EXPECT_MARKER) {
                marker_count += 1;
            }
        }
    }
    marker_count
}

/// Reports the worst over-budget functions, named, across every feature world.
///
/// `--top N` caps the printed table (0 prints everything, default 10);
/// `--scope production|tests|all` filters which tree the table ranks. The
/// summary always describes the full merged set so a filtered table cannot
/// hide the overall shape of the debt.
pub(crate) fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    let started_at = Instant::now();
    let parsed_arguments = parse_ranked_report_arguments(arguments)?;
    let worlds = function_length_lanes();
    println!(
        "[function-length-top] status=start worlds={} ranked_count={} scope={} root={}",
        worlds.len(),
        parsed_arguments.ranked_count,
        parsed_arguments.scope_filter.label(),
        repository_root.display()
    );

    let mut findings_per_world: Vec<Vec<OverBudgetFunction>> = Vec::new();
    for (world_index, world) in worlds.iter().enumerate() {
        let world_findings =
            run_world_with_forced_budget_lint(repository_root, &world.cargo_arguments)?;
        println!(
            "[function-length-top] status=progress world={}/{} name={} findings={} elapsed_seconds={}",
            world_index + 1,
            worlds.len(),
            world.name,
            world_findings.len(),
            started_at.elapsed().as_secs()
        );
        findings_per_world.push(world_findings);
    }

    let merged_findings = merge_world_findings_maximum(findings_per_world);
    let mut production_total = 0usize;
    for finding in &merged_findings {
        if !is_test_scope(&finding.path) {
            production_total += 1;
        }
    }

    let mut shown_count = 0usize;
    for finding in &merged_findings {
        if !parsed_arguments.scope_filter.allows(&finding.path) {
            continue;
        }
        shown_count += 1;
        if parsed_arguments.ranked_count > 0 && shown_count > parsed_arguments.ranked_count {
            break;
        }
        let function_name = resolve_function_name(repository_root, finding);
        let scope_label = if is_test_scope(&finding.path) {
            "test"
        } else {
            "production"
        };
        println!(
            "[function-length-top] rank={} code_lines={} scope={} name={} path={} line={}",
            shown_count, finding.code_lines, scope_label, function_name, finding.path, finding.line
        );
    }
    println!(
        "[function-length-top] status=success ranked={} production={} tests={} shown={} tracked_expects={} elapsed_seconds={}",
        merged_findings.len(),
        production_total,
        merged_findings.len() - production_total,
        shown_count,
        count_tracked_expect_markers(repository_root),
        started_at.elapsed().as_secs()
    );
    Ok(())
}
