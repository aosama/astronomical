//! Unit coverage for the ranked function-length report: argument parsing,
//! function-name resolution, world-merge semantics, scope filtering, and the
//! tracked-debt gauge.

use std::fs;
use std::path::Path;

use super::function_length::OverBudgetFunction;
use super::function_length_top::{
    ScopeFilter, TRACKED_EXPECT_MARKER, count_tracked_expect_markers, function_name_near_span,
    merge_world_findings_maximum, parse_function_declaration_name, parse_ranked_report_arguments,
};

/// Builds an expect-ratchet attribute line for fixtures without embedding
/// the debt marker literally: the gauge counts every line containing the
/// marker across the repository, and this test file must contribute zero.
fn expect_attribute_fixture() -> String {
    format!("#[{TRACKED_EXPECT_MARKER}]")
}

fn finding_with(code_lines: u64, path: &str, line: usize) -> OverBudgetFunction {
    OverBudgetFunction {
        code_lines,
        path: path.to_owned(),
        line,
    }
}

#[test]
fn should_default_to_the_top_ten_across_all_scopes() {
    let parsed = parse_ranked_report_arguments(&[]).expect("empty arguments should parse");

    assert_eq!(parsed.ranked_count, 10);
    assert_eq!(parsed.scope_filter, ScopeFilter::All);
}

#[test]
fn should_parse_an_explicit_ranked_count_and_scope() {
    let arguments = vec![
        "--top".to_owned(),
        "3".to_owned(),
        "--scope".to_owned(),
        "production".to_owned(),
    ];

    let parsed =
        parse_ranked_report_arguments(&arguments).expect("explicit arguments should parse");

    assert_eq!(parsed.ranked_count, 3);
    assert_eq!(parsed.scope_filter, ScopeFilter::Production);
}

#[test]
fn should_accept_a_zero_ranked_count_as_show_everything() {
    let arguments = vec!["--top".to_owned(), "0".to_owned()];

    let parsed = parse_ranked_report_arguments(&arguments).expect("zero should parse");

    assert_eq!(parsed.ranked_count, 0);
}

#[test]
fn should_reject_an_unknown_argument() {
    let arguments = vec!["--watch".to_owned()];

    let outcome = parse_ranked_report_arguments(&arguments);

    assert!(
        outcome.is_err(),
        "unknown arguments must fail, not be ignored"
    );
}

#[test]
fn should_reject_a_non_numeric_ranked_count() {
    let arguments = vec!["--top".to_owned(), "ten".to_owned()];

    let outcome = parse_ranked_report_arguments(&arguments);

    assert!(outcome.is_err(), "a word is not a count");
}

#[test]
fn should_reject_a_dangling_top_flag() {
    let arguments = vec!["--top".to_owned()];

    let outcome = parse_ranked_report_arguments(&arguments);

    assert!(outcome.is_err(), "--top without a count must fail");
}

#[test]
fn should_reject_an_unknown_scope() {
    let arguments = vec!["--scope".to_owned(), "everything".to_owned()];

    let outcome = parse_ranked_report_arguments(&arguments);

    assert!(outcome.is_err(), "an unknown scope must fail, not default");
}

#[test]
fn should_extract_a_function_name_from_a_declaration_line() {
    assert_eq!(
        parse_function_declaration_name("    fn advance_active_request("),
        Some("advance_active_request".to_owned())
    );
    assert_eq!(
        parse_function_declaration_name("    pub(crate) fn run(repository_root: &Path)"),
        Some("run".to_owned())
    );
    assert_eq!(
        parse_function_declaration_name("    unsafe fn signal_edge("),
        Some("signal_edge".to_owned())
    );
}

#[test]
fn should_reject_a_line_without_a_function_declaration() {
    assert_eq!(
        parse_function_declaration_name("        let budget = 100;"),
        None
    );
    assert_eq!(parse_function_declaration_name(""), None);
}

#[test]
fn should_not_treat_an_identifier_ending_in_fn_as_a_declaration() {
    assert_eq!(
        parse_function_declaration_name("    profn corrupted("),
        None
    );
}

#[test]
fn should_resolve_a_name_below_the_reported_attribute_line() {
    // The ratchet guarantees attributes on tracked functions; clippy anchors
    // its span on the attribute line, so the declaration sits one line below.
    // Any attribute proves the attribute-line anchor; the debt marker is
    // deliberately not embedded here (the gauge counts every line with it).
    let source_lines = vec![
        "    /// Doc comment.",
        "    #[allow(dead_code)]",
        "    fn advance_active_request(",
        "        &mut self,",
    ];

    let function_name = function_name_near_span(&source_lines, 2);

    assert_eq!(function_name, Some("advance_active_request".to_owned()));
}

#[test]
fn should_resolve_a_name_on_the_reported_line_itself() {
    let source_lines = vec![
        "impl Engine {",
        "    fn plain_function_without_attributes() {",
        "    }",
    ];

    let function_name = function_name_near_span(&source_lines, 2);

    assert_eq!(
        function_name,
        Some("plain_function_without_attributes".to_owned())
    );
}

#[test]
fn should_skip_comment_lines_during_the_name_scan() {
    // A doc example mentioning `fn` above the declaration must not win.
    let source_lines = vec![
        "    /// Mirrors `fn forward_moe_paged` semantics.",
        "    #[allow(dead_code)]",
        "    fn real_declaration(",
    ];

    let function_name = function_name_near_span(&source_lines, 1);

    assert_eq!(function_name, Some("real_declaration".to_owned()));
}

#[test]
fn should_return_no_name_when_the_file_ends_before_a_declaration() {
    let source_lines = vec!["struct OnlyAStruct;"];

    let function_name = function_name_near_span(&source_lines, 1);

    assert_eq!(function_name, None);
}

#[test]
fn should_merge_worlds_keeping_the_largest_count_per_site() {
    // The same function cfg-gates different lines per world; the maximum is
    // the honest worst case.
    let default_world = vec![finding_with(299, "crates/x/src/forward.rs", 54)];
    let mlx_world = vec![finding_with(305, "crates/x/src/forward.rs", 54)];
    let acceptance_world = vec![finding_with(272, "apps/daemon/src/chat.rs", 28)];

    let merged = merge_world_findings_maximum(vec![default_world, mlx_world, acceptance_world]);

    assert_eq!(merged.len(), 2);
    assert_eq!(merged[0], finding_with(305, "crates/x/src/forward.rs", 54));
    assert_eq!(merged[1], finding_with(272, "apps/daemon/src/chat.rs", 28));
}

#[test]
fn should_merge_to_an_empty_list_without_findings() {
    let merged = merge_world_findings_maximum(vec![Vec::new(), Vec::new()]);

    assert!(merged.is_empty());
}

#[test]
fn should_rank_merged_findings_worst_first_deterministically() {
    let first_world = vec![
        finding_with(103, "crates/x/src/small.rs", 7),
        finding_with(349, "crates/x/src/large.rs", 92),
    ];

    let first_ranking = merge_world_findings_maximum(vec![first_world.clone()]);
    let second_ranking = merge_world_findings_maximum(vec![first_world]);

    assert_eq!(first_ranking[0].code_lines, 349, "worst first");
    assert_eq!(
        first_ranking, second_ranking,
        "ties resolve the same every run"
    );
}

#[test]
fn should_scope_filter_allows_both_trees_under_all() {
    assert!(ScopeFilter::All.allows("crates/x/src/production.rs"));
    assert!(ScopeFilter::All.allows("crates/x/tests/integration.rs"));
}

#[test]
fn should_scope_filter_split_production_from_tests() {
    assert!(ScopeFilter::Production.allows("crates/x/src/production.rs"));
    assert!(!ScopeFilter::Production.allows("crates/x/tests/integration.rs"));
    assert!(!ScopeFilter::Tests.allows("crates/x/src/production.rs"));
    assert!(ScopeFilter::Tests.allows("apps/daemon/tests/fixtures/worker.rs"));
}

#[test]
fn should_count_expect_markers_excluding_build_and_hidden_trees() {
    let repository_root = tempfile::tempdir().expect("the temporary repository should create");
    write_marked_source(repository_root.path(), "crates/example/src/lib.rs", 2);
    // Build output and hidden trees are not repository source.
    write_marked_source(repository_root.path(), "target/debug/out.rs", 1);
    write_marked_source(repository_root.path(), ".git/hooks/pre-commit.rs", 1);
    // Non-Rust files never carry the attribute.
    write_marked_source(repository_root.path(), "docs/notes.txt", 1);

    let marker_count = count_tracked_expect_markers(repository_root.path());

    assert_eq!(marker_count, 2);
}

#[test]
fn should_count_zero_markers_in_a_repository_without_any() {
    let repository_root = tempfile::tempdir().expect("the temporary repository should create");
    write_source(
        repository_root.path(),
        "crates/example/src/lib.rs",
        "fn clean() {}\n",
    );

    let marker_count = count_tracked_expect_markers(repository_root.path());

    assert_eq!(marker_count, 0);
}

/// Writes a source file carrying `marker_line_count` debt-marker lines and
/// one plain declaration, with the attribute built from the marker constant
/// so this test file itself contributes zero lines to the gauge.
fn write_marked_source(repository_root: &Path, relative_path: &str, marker_line_count: usize) {
    let mut source_text = String::new();
    for _ in 0..marker_line_count {
        source_text.push_str(&expect_attribute_fixture());
        source_text.push('\n');
    }
    source_text.push_str("fn filler_declaration() {}\n");
    write_source(repository_root, relative_path, &source_text);
}

fn write_source(repository_root: &Path, relative_path: &str, source_text: &str) {
    let absolute_path = repository_root.join(relative_path);
    fs::create_dir_all(absolute_path.parent().expect("the parent should exist"))
        .expect("the source tree should create");
    fs::write(absolute_path, source_text).expect("the source file should write");
}
