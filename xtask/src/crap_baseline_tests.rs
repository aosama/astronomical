use std::cell::Cell;
use std::collections::BTreeSet;

use serde_json::json;

use crate::crap_baseline::admissions::parse_admissions;
use crate::crap_baseline::verdict::{
    MAXIMUM_NEW_FUNCTION_CYCLOMATIC_COMPLEXITY, judge_delta_report,
};
use crate::crap_baseline::{
    BASELINE_RELATIVE_PATH, CrapBaselineInvocation, CrapReportSource, run_with_source,
};

const LOOKUP_TABLE_FILE: &str = "crates/ipc-protocol/src/worker_event_diagnostics.rs";
const LOOKUP_TABLE_FUNCTION: &str = "WorkerEvent::diagnostic_event_name";
const COMMITTED_BASELINE_TEXT: &str = "{\"committed\": true}\n";

fn lookup_table_label() -> String {
    format!("{LOOKUP_TABLE_FILE}:{LOOKUP_TABLE_FUNCTION}")
}

/// One delta entry shaped like cargo-crap 0.9's `delta-v2` report. A grown
/// entry's baseline score is one complexity step lower, as cargo-crap
/// reports it; a new entry has none.
fn delta_entry(file: &str, function: &str, status: &str, cyclomatic: f64) -> serde_json::Value {
    let crap_score = cyclomatic * cyclomatic + cyclomatic;
    let baseline_crap = match status {
        "new" => serde_json::Value::Null,
        "regressed" => json!((cyclomatic - 1.0) * (cyclomatic - 1.0) + cyclomatic - 1.0),
        _ => json!(crap_score),
    };
    json!({
        "file": file,
        "function": function,
        "line": 1,
        "cyclomatic": cyclomatic,
        "coverage": null,
        "crap": crap_score,
        "crate": "astronomical-example",
        "baseline_crap": baseline_crap,
        "status": status,
    })
}

fn delta_report(entries: Vec<serde_json::Value>, removed_count: usize) -> String {
    let mut removed_entries = Vec::with_capacity(removed_count);
    for removed_index in 0..removed_count {
        removed_entries.push(json!({
            "function": format!("removed_{removed_index}"),
            "file": "src/gone.rs",
            "baseline_crap": 2.0,
        }));
    }
    json!({
        "$schema": "https://raw.githubusercontent.com/minikin/cargo-crap/main/schemas/delta-v2.json",
        "version": "0.9.0",
        "entries": entries,
        "removed": removed_entries,
    })
    .to_string()
}

fn fresh_baseline_report() -> String {
    json!({
        "$schema": "https://raw.githubusercontent.com/minikin/cargo-crap/main/schemas/report-v1.json",
        "version": "0.9.0",
        "entries": [delta_entry("src/parse.rs", "parse_command", "unchanged", 5.0)],
    })
    .to_string()
}

fn admissions_document(admitted_labels: &[&str]) -> String {
    let mut admitted_entries = Vec::new();
    for admitted_label in admitted_labels {
        admitted_entries.push(json!({"function": admitted_label, "reason": "a lookup table"}));
    }
    json!({"admitted_functions": admitted_entries}).to_string()
}

fn no_admissions() -> BTreeSet<String> {
    BTreeSet::new()
}

/// Canned scans; counts the fresh-baseline scans so a test can prove a check
/// never pays for one.
struct CannedReportSource {
    delta_report_text: String,
    fresh_baseline_text: String,
    fresh_baseline_scan_count: Cell<usize>,
}

impl CannedReportSource {
    fn new(delta_report_text: String, fresh_baseline_text: String) -> Self {
        Self {
            delta_report_text,
            fresh_baseline_text,
            fresh_baseline_scan_count: Cell::new(0),
        }
    }
}

impl CrapReportSource for CannedReportSource {
    fn delta_against_committed_baseline(&self) -> Result<String, String> {
        Ok(self.delta_report_text.clone())
    }

    fn fresh_baseline_report(&self) -> Result<String, String> {
        self.fresh_baseline_scan_count
            .set(self.fresh_baseline_scan_count.get() + 1);
        Ok(self.fresh_baseline_text.clone())
    }
}

fn repository_with_committed_baseline() -> tempfile::TempDir {
    let repository_root = tempfile::tempdir().expect("a temporary repository");
    std::fs::write(
        repository_root.path().join(BASELINE_RELATIVE_PATH),
        COMMITTED_BASELINE_TEXT,
    )
    .expect("the baseline writes");
    repository_root
}

fn baseline_text(repository_root: &tempfile::TempDir) -> String {
    std::fs::read_to_string(repository_root.path().join(BASELINE_RELATIVE_PATH))
        .expect("the baseline reads")
}

const WRITE_INVOCATION: CrapBaselineInvocation = CrapBaselineInvocation {
    should_write_baseline: true,
};

#[test]
fn should_refuse_a_baseline_function_that_grew() {
    let report = delta_report(
        vec![delta_entry(
            "src/parse.rs",
            "parse_command",
            "regressed",
            19.0,
        )],
        0,
    );
    let verdict = judge_delta_report(&report, &no_admissions()).expect("the report parses");
    assert!(!verdict.passes());
    assert_eq!(
        verdict.grown_functions,
        vec!["src/parse.rs:parse_command baseline_crap=342 crap=380".to_owned()]
    );
}

#[test]
fn should_accept_a_baseline_function_that_shrank_stayed_or_only_moved() {
    let report = delta_report(
        vec![
            delta_entry("src/parse.rs", "parse_command", "improved", 12.0),
            delta_entry("src/parse.rs", "flag_value", "unchanged", 5.0),
            delta_entry("src/shared/engine.rs", "advance", "moved", 40.0),
        ],
        0,
    );
    let verdict = judge_delta_report(&report, &no_admissions()).expect("the report parses");
    assert!(verdict.passes());
    assert_eq!(verdict.new_function_count, 0);
}

#[test]
fn should_refuse_a_new_function_over_the_cap() {
    let report = delta_report(
        vec![delta_entry(
            "src/route.rs",
            "route_request",
            "new",
            MAXIMUM_NEW_FUNCTION_CYCLOMATIC_COMPLEXITY + 1.0,
        )],
        0,
    );
    let verdict = judge_delta_report(&report, &no_admissions()).expect("the report parses");
    assert!(!verdict.passes());
    assert_eq!(
        verdict.over_cap_new_functions,
        vec!["src/route.rs:route_request cyclomatic=16".to_owned()]
    );
}

#[test]
fn should_accept_a_new_function_under_or_exactly_at_the_cap() {
    let report = delta_report(
        vec![
            delta_entry("src/route.rs", "route_small", "new", 3.0),
            delta_entry(
                "src/route.rs",
                "route_at_limit",
                "new",
                MAXIMUM_NEW_FUNCTION_CYCLOMATIC_COMPLEXITY,
            ),
        ],
        0,
    );
    let verdict = judge_delta_report(&report, &no_admissions()).expect("the report parses");
    assert!(verdict.passes());
    assert_eq!(verdict.new_function_count, 2);
}

#[test]
fn should_let_an_admitted_new_function_over_the_cap_through() {
    let report = delta_report(
        vec![delta_entry(
            LOOKUP_TABLE_FILE,
            LOOKUP_TABLE_FUNCTION,
            "new",
            33.0,
        )],
        0,
    );
    let admissions = BTreeSet::from([lookup_table_label()]);
    let verdict = judge_delta_report(&report, &admissions).expect("the report parses");
    assert!(verdict.passes());
    assert_eq!(verdict.admitted_functions.len(), 1);
}

#[test]
fn should_keep_an_admission_once_its_function_is_in_the_baseline() {
    let report = delta_report(
        vec![delta_entry(
            LOOKUP_TABLE_FILE,
            LOOKUP_TABLE_FUNCTION,
            "unchanged",
            33.0,
        )],
        0,
    );
    let admissions = BTreeSet::from([lookup_table_label()]);
    let verdict = judge_delta_report(&report, &admissions).expect("the report parses");
    assert!(verdict.passes());
}

#[test]
fn should_never_let_an_admission_cover_a_function_that_grew() {
    let report = delta_report(
        vec![delta_entry(
            LOOKUP_TABLE_FILE,
            LOOKUP_TABLE_FUNCTION,
            "regressed",
            34.0,
        )],
        0,
    );
    let admissions = BTreeSet::from([lookup_table_label()]);
    let verdict = judge_delta_report(&report, &admissions).expect("the report parses");
    assert!(!verdict.passes());
    assert_eq!(verdict.grown_functions.len(), 1);
}

#[test]
fn should_refuse_an_admission_naming_no_function() {
    let report = delta_report(
        vec![delta_entry("src/route.rs", "route_small", "new", 3.0)],
        0,
    );
    let admissions = BTreeSet::from(["src/route.rs:route_gone".to_owned()]);
    let verdict = judge_delta_report(&report, &admissions).expect("the report parses");
    assert!(!verdict.passes());
    assert_eq!(
        verdict.stale_admissions,
        vec!["src/route.rs:route_gone".to_owned()]
    );
}

#[test]
fn should_refuse_an_admission_that_covers_several_same_named_functions() {
    let report = delta_report(
        vec![
            delta_entry("src/errors.rs", "EmbedError::from", "new", 20.0),
            delta_entry("src/errors.rs", "EmbedError::from", "new", 18.0),
        ],
        0,
    );
    let admissions = BTreeSet::from(["src/errors.rs:EmbedError::from".to_owned()]);
    let verdict = judge_delta_report(&report, &admissions).expect("the report parses");
    assert!(!verdict.passes());
    assert_eq!(verdict.ambiguous_admissions.len(), 1);
}

#[test]
fn should_list_removed_functions_without_failing() {
    let verdict = judge_delta_report(&delta_report(Vec::new(), 3), &no_admissions())
        .expect("the report parses");
    assert!(verdict.passes());
    assert_eq!(verdict.removed_functions.len(), 3);
    assert_eq!(
        verdict.removed_functions[0],
        "src/gone.rs:removed_0 baseline_crap=2"
    );
}

#[test]
fn should_fail_closed_on_malformed_or_unfamiliar_reports() {
    assert!(judge_delta_report("not json", &no_admissions()).is_err());
    assert!(judge_delta_report(r#"{"entries": []}"#, &no_admissions()).is_err());
    let unknown_status = delta_report(vec![delta_entry("src/a.rs", "a", "renamed", 1.0)], 0);
    assert!(judge_delta_report(&unknown_status, &no_admissions()).is_err());
    let grown_without_baseline_score = json!({
        "entries": [{"file": "src/a.rs", "function": "a", "status": "regressed", "cyclomatic": 3.0, "crap": 12.0}],
        "removed": [],
    })
    .to_string();
    assert!(judge_delta_report(&grown_without_baseline_score, &no_admissions()).is_err());
}

#[test]
fn should_parse_admissions_with_reasons() {
    let admitted_labels = parse_admissions(&admissions_document(&[&lookup_table_label()]))
        .expect("the admissions parse");
    assert_eq!(admitted_labels, BTreeSet::from([lookup_table_label()]));
    assert!(
        parse_admissions(r#"{"admitted_functions": []}"#)
            .expect("an empty list parses")
            .is_empty()
    );
}

#[test]
fn should_refuse_admissions_without_a_reason_file_or_unique_label() {
    let without_reason =
        json!({"admitted_functions": [{"function": lookup_table_label(), "reason": " "}]});
    assert!(parse_admissions(&without_reason.to_string()).is_err());
    let without_file = json!({"admitted_functions": [{"function": "route", "reason": "table"}]});
    assert!(parse_admissions(&without_file.to_string()).is_err());
    let label = lookup_table_label();
    assert!(parse_admissions(&admissions_document(&[&label, &label])).is_err());
    assert!(parse_admissions("[]").is_err());
}

#[test]
fn should_accept_only_the_write_argument() {
    assert_eq!(
        CrapBaselineInvocation::parse(&[]).expect("no arguments is a check"),
        CrapBaselineInvocation::default()
    );
    assert!(
        CrapBaselineInvocation::parse(&["--write".to_owned()])
            .expect("write parses")
            .should_write_baseline
    );
    assert!(CrapBaselineInvocation::parse(&["--admit".to_owned()]).is_err());
}

#[test]
fn should_leave_the_baseline_untouched_when_a_function_grew() {
    let repository_root = repository_with_committed_baseline();
    let report_source = CannedReportSource::new(
        delta_report(
            vec![delta_entry(
                "src/parse.rs",
                "parse_command",
                "regressed",
                19.0,
            )],
            0,
        ),
        fresh_baseline_report(),
    );
    let outcome = run_with_source(
        repository_root.path(),
        &WRITE_INVOCATION,
        &admissions_document(&[]),
        &report_source,
    );
    assert!(outcome.is_err());
    assert_eq!(baseline_text(&repository_root), COMMITTED_BASELINE_TEXT);
}

#[test]
fn should_write_the_fresh_report_byte_for_byte_when_the_check_passes() {
    let repository_root = repository_with_committed_baseline();
    let report_source = CannedReportSource::new(
        delta_report(
            vec![delta_entry(
                LOOKUP_TABLE_FILE,
                LOOKUP_TABLE_FUNCTION,
                "new",
                33.0,
            )],
            1,
        ),
        fresh_baseline_report(),
    );
    run_with_source(
        repository_root.path(),
        &WRITE_INVOCATION,
        &admissions_document(&[&lookup_table_label()]),
        &report_source,
    )
    .expect("an admitted function passes");
    assert_eq!(baseline_text(&repository_root), fresh_baseline_report());
}

#[test]
fn should_refuse_to_write_a_fresh_scan_that_is_not_a_baseline_report() {
    let repository_root = repository_with_committed_baseline();
    let report_source =
        CannedReportSource::new(delta_report(Vec::new(), 0), delta_report(Vec::new(), 0));
    let outcome = run_with_source(
        repository_root.path(),
        &WRITE_INVOCATION,
        &admissions_document(&[]),
        &report_source,
    );
    assert!(outcome.is_err());
    assert_eq!(baseline_text(&repository_root), COMMITTED_BASELINE_TEXT);
}

#[test]
fn should_check_without_scanning_for_or_writing_a_fresh_baseline() {
    let repository_root = repository_with_committed_baseline();
    let report_source =
        CannedReportSource::new(delta_report(Vec::new(), 0), fresh_baseline_report());
    run_with_source(
        repository_root.path(),
        &CrapBaselineInvocation::default(),
        &admissions_document(&[]),
        &report_source,
    )
    .expect("an empty report passes");
    assert_eq!(report_source.fresh_baseline_scan_count.get(), 0);
    assert_eq!(baseline_text(&repository_root), COMMITTED_BASELINE_TEXT);
}
