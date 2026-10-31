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

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::function_length::{
    OverBudgetFunction, function_length_lanes, is_test_scope, merge_world_findings_maximum,
    resolve_function_name, run_lane_with_forced_budget_lint,
};

/// The attribute this campaign counts as tracked debt. Every pre-existing
/// over-budget function carries one; the ratchet deletes markers
/// automatically when a function shrinks below budget, so the count is a
/// monotonic paydown gauge.
pub(crate) const TRACKED_EXPECT_MARKER: &str = "expect(clippy::too_many_lines)";

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
        let world_findings = run_lane_with_forced_budget_lint(
            repository_root,
            "[function-length-top]",
            &world.cargo_arguments,
        )?;
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
