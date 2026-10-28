//! Unit coverage for the function-length report lane: the diagnostic parser,
//! the scope classifier, the worst-first ordering, and the feature-world set.

use super::function_length::{
    OverBudgetFunction, function_length_lanes, is_test_scope, parse_over_budget_diagnostics,
};

#[test]
fn should_parse_a_single_over_budget_diagnostic() {
    let diagnostic_text = "crates/quantization/src/allocation/mod.rs:177:1: warning: this function has too many lines (142/100)\n";

    let findings = parse_over_budget_diagnostics(diagnostic_text);

    assert_eq!(
        findings,
        vec![OverBudgetFunction {
            code_lines: 142,
            path: "crates/quantization/src/allocation/mod.rs".to_owned(),
            line: 177,
        }]
    );
}

#[test]
fn should_parse_multiple_diagnostics_from_one_stream() {
    let diagnostic_text = "\
    Checking astronomical-config
crates/config/src/model_discovery/artifact_discovery.rs:29:1: warning: this function has too many lines (116/100)
apps/daemon/src/worker.rs:60:1: warning: this function has too many lines (521/100)
";

    let findings = parse_over_budget_diagnostics(diagnostic_text);

    assert_eq!(findings.len(), 2);
    assert_eq!(findings[0].code_lines, 521);
    assert_eq!(findings[1].code_lines, 116);
}

#[test]
fn should_ignore_diagnostics_that_describe_other_lints() {
    let diagnostic_text = "\
crates/config/src/lib.rs:12:5: warning: unused variable: `unused_binding`
crates/config/src/lib.rs:20:1: warning: this function has too many arguments (8/7)
crates/config/src/lib.rs:30:1: error: expected one of `!` or `::`, found `}`
";

    let findings = parse_over_budget_diagnostics(diagnostic_text);

    assert!(findings.is_empty());
}

#[test]
fn should_ignore_successful_compilation_output() {
    let diagnostic_text = "\
    Checking astronomical-model-serving v0.2.84 (/workspace/crates/model-serving)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 4.79s
";

    let findings = parse_over_budget_diagnostics(diagnostic_text);

    assert!(findings.is_empty());
}

#[test]
fn should_ignore_an_over_budget_message_without_a_parseable_count() {
    let diagnostic_text =
        "crates/config/src/lib.rs:12:5: warning: this function has too many lines (many/100)\n";

    let findings = parse_over_budget_diagnostics(diagnostic_text);

    assert!(findings.is_empty());
}

#[test]
fn should_deduplicate_the_same_function_reported_by_two_lanes() {
    let diagnostic_text = "\
crates/model-serving/src/qwen3_5_streaming/model/forward.rs:53:1: warning: this function has too many lines (299/100)
crates/model-serving/src/qwen3_5_streaming/model/forward.rs:53:1: warning: this function has too many lines (299/100)
";

    let findings = parse_over_budget_diagnostics(diagnostic_text);

    assert_eq!(findings.len(), 1);
}

#[test]
fn should_order_the_worst_functions_first() {
    let diagnostic_text = "\
crates/config/src/lib.rs:12:5: warning: this function has too many lines (116/100)
crates/config/src/lib.rs:40:5: warning: this function has too many lines (521/100)
crates/config/src/lib.rs:80:5: warning: this function has too many lines (142/100)
";

    let findings = parse_over_budget_diagnostics(diagnostic_text);

    let mut counts: Vec<u64> = Vec::new();
    for finding in &findings {
        counts.push(finding.code_lines);
    }
    assert_eq!(counts, vec![521, 142, 116]);
}

#[test]
fn should_classify_verification_and_test_files_as_test_scope() {
    assert!(is_test_scope(
        "verification/acceptance/tests/performance/throughput/support.rs"
    ));
    assert!(is_test_scope(
        "apps/daemon/tests/fixtures/scripted_worker.rs"
    ));
    assert!(is_test_scope(
        "crates/model-serving/src/engine_backed_worker/tests/chat/prefill_progress.rs"
    ));
    assert!(is_test_scope(
        "crates/ipc-protocol/src/daemon_protocol_tests.rs"
    ));
}

#[test]
fn should_classify_production_sources_as_production_scope() {
    assert!(!is_test_scope("crates/quantization/src/allocation/mod.rs"));
    assert!(!is_test_scope("apps/daemon/src/worker.rs"));
    // A path that merely contains the word "test" is not a test tree.
    assert!(!is_test_scope(
        "crates/model-serving/src/artifact_validation/latest_path.rs"
    ));
}

#[test]
fn should_cover_every_feature_world_with_the_budget_lint_armed() {
    let lanes = function_length_lanes();

    let mut names: Vec<&str> = Vec::new();
    for lane in &lanes {
        names.push(lane.name);
    }
    assert_eq!(
        names,
        vec!["default-world", "mlx-world", "acceptance-feature-superset"]
    );

    let mlx_lane = &lanes[1];
    let mlx_arguments = mlx_lane.cargo_arguments.join(" ");
    assert!(mlx_arguments.contains("astronomical-model-serving/direct-mlx"));
    assert!(mlx_arguments.contains("astronomical-runtime-integration/mlx"));

    let acceptance_lane = &lanes[2];
    let acceptance_arguments = acceptance_lane.cargo_arguments.join(" ");
    assert!(acceptance_arguments.contains("performance_throughput"));
    assert!(acceptance_arguments.contains("mlx-memory-contract-probe"));
}
