//! Unit coverage for the hard production-scope budget gate: exemption
//! matching, finding classification, argument parsing, and lane selection.

use super::function_length::{FunctionLengthLane, OverBudgetFunction, function_length_lanes};
use super::function_length_debt::{TEST_SCOPE_FREEZE, is_exempted_production_site};
use super::function_length_gate::{
    NamedOverBudgetFinding, TestScopeFreezeViolationKind, check_test_scope_freeze,
    classify_named_findings, parse_gate_arguments, select_lanes_by_names,
};

fn named_finding(
    code_lines: u64,
    path: &str,
    line: usize,
    function_name: &str,
) -> NamedOverBudgetFinding {
    NamedOverBudgetFinding {
        finding: OverBudgetFunction {
            code_lines,
            path: path.to_owned(),
            line,
        },
        function_name: function_name.to_owned(),
    }
}

#[test]
fn should_exempt_the_family_d_declarative_table_sites() {
    // Positive: each listed site matches by path plus name.
    assert!(is_exempted_production_site(
        "crates/model-serving/src/performance_attribution/catalog.rs",
        "identifier",
    ));
    assert!(is_exempted_production_site(
        "crates/model-serving/src/performance_attribution/counter_catalog.rs",
        "identifier",
    ));
    assert!(is_exempted_production_site(
        "crates/model-serving/src/qwen_image_21/tensor_profiles/text_encoder.rs",
        "text_encoder_tensor_profiles",
    ));
    assert!(is_exempted_production_site(
        "crates/model-serving/src/qwen_image_21/tensor_profiles/vae.rs",
        "vae_tensor_profiles",
    ));
}

#[test]
fn should_not_exempt_a_same_name_function_in_a_different_file() {
    // Edge: the exemption list carries two `identifier` tables; a third
    // `identifier` elsewhere must not ride their name.
    assert!(!is_exempted_production_site(
        "crates/model-serving/src/qwen3_5_core/artifacts/artifact.rs",
        "identifier",
    ));
}

#[test]
fn should_not_exempt_the_exempted_file_for_a_different_function() {
    // Edge: a new over-budget function inside an exempted file still fails.
    assert!(!is_exempted_production_site(
        "crates/model-serving/src/performance_attribution/catalog.rs",
        "some_other_function",
    ));
}

#[test]
fn should_not_exempt_an_unresolved_function_name() {
    // Edge: name resolution returning "unknown" (unreadable or moved source)
    // must fail the gate rather than accidentally matching an exemption.
    assert!(!is_exempted_production_site(
        "crates/model-serving/src/performance_attribution/catalog.rs",
        "unknown",
    ));
}

#[test]
fn should_classify_an_unexempted_production_finding_as_a_failure() {
    // Negative: production scope, not exempt — the gate must fail on this.
    let classification = classify_named_findings(&[named_finding(
        104,
        "crates/model-serving/src/some_engine.rs",
        22,
        "forward_layer",
    )]);

    assert_eq!(classification.failures.len(), 1);
    assert_eq!(classification.failures[0].function_name, "forward_layer");
    assert!(classification.exempted.is_empty());
    assert!(classification.test_scope.is_empty());
}

#[test]
fn should_classify_an_exempted_production_finding_as_exempted() {
    // Positive: the Family D table sites report as exempted, not failures.
    let classification = classify_named_findings(&[named_finding(
        234,
        "crates/model-serving/src/performance_attribution/catalog.rs",
        324,
        "identifier",
    )]);

    assert!(classification.failures.is_empty());
    assert_eq!(classification.exempted.len(), 1);
    assert_eq!(classification.exempted[0].finding.code_lines, 234);
    assert!(classification.test_scope.is_empty());
}

#[test]
fn should_classify_test_scope_findings_as_test_scope_regardless_of_exemptions() {
    // Edge: test-scope paths sort into the test bucket even when their name
    // collides with an exempted production function.
    let classification = classify_named_findings(&[
        named_finding(
            150,
            "verification/journeys/some_journey.rs",
            10,
            "run_journey",
        ),
        named_finding(
            150,
            "crates/model-serving/src/some_tests.rs",
            10,
            "identifier",
        ),
    ]);

    assert!(classification.failures.is_empty());
    assert!(classification.exempted.is_empty());
    assert_eq!(classification.test_scope.len(), 2);
}

#[test]
fn should_preserve_arrival_order_inside_each_bucket() {
    // The merged findings arrive worst-first from the lane merge;
    // classification preserves that order within every bucket so the printed
    // list stays ranked without a second sort.
    let classification = classify_named_findings(&[
        named_finding(140, "crates/model-serving/src/engine_a.rs", 5, "forward_a"),
        named_finding(120, "crates/model-serving/src/engine_b.rs", 5, "forward_b"),
    ]);

    assert_eq!(classification.failures[0].finding.code_lines, 140);
    assert_eq!(classification.failures[1].finding.code_lines, 120);
}

#[test]
fn should_reject_a_gate_run_with_no_check_toggle() {
    // Edge: a gate run with no check toggle would vacuously pass, so it is
    // refused rather than allowed to succeed with nothing checked.
    let parse_error = parse_gate_arguments(&[]).expect_err("no check toggle must fail");

    assert!(parse_error.contains("at least one of --fail-on-production or --fail-on-tests"));
}

#[test]
fn should_parse_production_and_test_check_toggles() {
    let arguments = vec![
        "--fail-on-production".to_owned(),
        "--fail-on-tests".to_owned(),
    ];

    let parsed = parse_gate_arguments(&arguments).expect("both toggles should parse");

    assert!(parsed.fail_on_production);
    assert!(parsed.fail_on_tests);
    assert_eq!(parsed.lane_names, None);
}

#[test]
fn should_parse_a_lane_subset() {
    let arguments = vec![
        "--fail-on-production".to_owned(),
        "--lanes".to_owned(),
        "default-world,mlx-world".to_owned(),
    ];

    let parsed = parse_gate_arguments(&arguments).expect("lane subset should parse");

    assert_eq!(
        parsed.lane_names.as_deref(),
        Some("default-world,mlx-world")
    );
    assert!(parsed.fail_on_production);
    assert!(!parsed.fail_on_tests);
}

#[test]
fn should_reject_unknown_gate_arguments() {
    let arguments = vec!["--scope".to_owned(), "production".to_owned()];

    let parse_error = parse_gate_arguments(&arguments).expect_err("unknown argument must fail");

    assert!(parse_error.contains("unknown argument --scope"));
}

#[test]
fn should_reject_a_lanes_flag_without_a_value() {
    let arguments = vec!["--lanes".to_owned()];

    let parse_error = parse_gate_arguments(&arguments).expect_err("missing value must fail");

    assert!(parse_error.contains("--lanes requires a comma-separated lane list"));
}

#[test]
fn should_select_the_requested_lanes_in_canonical_order() {
    let selected = select_lanes_by_names(&function_length_lanes(), "mlx-world,default-world")
        .expect("known lanes should select");

    assert_eq!(selected.len(), 2);
    // Canonical lane order is preserved regardless of the requested order.
    assert_eq!(selected[0].name, "default-world");
    assert_eq!(selected[1].name, "mlx-world");
}

#[test]
fn should_fail_lane_selection_on_an_unknown_lane_name() {
    let selection_error =
        select_lanes_by_names(&function_length_lanes(), "default-world,acceptance-world")
            .expect_err("unknown lane must fail");

    assert!(selection_error.contains("unknown lane acceptance-world"));
    assert!(selection_error.contains("acceptance-feature-superset"));
}

#[test]
fn should_fail_lane_selection_on_an_empty_lane_name_segment() {
    // Edge: a trailing comma must not silently narrow the gate; a zero-lane
    // gate would vacuously pass.
    let selection_error = select_lanes_by_names(&function_length_lanes(), "default-world,")
        .expect_err("empty segment must fail");

    assert!(selection_error.contains("empty name"));
}

#[test]
fn should_fail_lane_selection_on_an_empty_lane_list() {
    let selection_error =
        select_lanes_by_names(&[], "default-world").expect_err("no lanes to select from must fail");

    assert!(selection_error.contains("unknown lane default-world"));
}

#[test]
fn should_carry_the_clippy_arguments_through_lane_cloning() {
    // The gate runs cloned lanes; the clone must retain the cargo arguments
    // or the selected lane would lint nothing.
    let selected = select_lanes_by_names(&function_length_lanes(), "default-world")
        .expect("default lane should select");

    assert_eq!(
        selected[0].cargo_arguments,
        vec!["--workspace", "--all-targets"]
    );
}

#[test]
fn should_expose_exactly_the_four_family_d_exemptions() {
    // Boundary: the exemption list is the complete accepted production debt;
    // this pins its size so every addition is a deliberate, visible change.
    let exemption_count = super::function_length_debt::PRODUCTION_SCOPE_EXEMPTIONS.len();

    assert_eq!(exemption_count, 4);
}

#[test]
fn should_keep_every_exempted_site_present_in_the_repository_lanes() {
    // Boundary: every lane name the hook may request must be a real lane.
    let all_lanes: Vec<FunctionLengthLane> = function_length_lanes();
    let lane_names: Vec<&str> = all_lanes.iter().map(|lane| lane.name).collect();

    assert!(lane_names.contains(&"default-world"));
    assert!(lane_names.contains(&"mlx-world"));
    assert!(lane_names.contains(&"acceptance-feature-superset"));
}

#[test]
fn should_freeze_a_known_test_scope_site_at_its_frozen_count() {
    // Positive: a frozen site reporting exactly its frozen count is clean —
    // the ratchet holds.
    let violations = check_test_scope_freeze(&[named_finding(
        261,
        "apps/daemon/tests/fixtures/idle_worker.rs",
        41,
        "run_fixture",
    )]);

    assert!(violations.is_empty());
}

#[test]
fn should_flag_a_new_test_scope_site_not_in_the_freeze() {
    // Negative: a test-scope site over budget that the freeze does not know is
    // new debt appearing where none may.
    let violations = check_test_scope_freeze(&[named_finding(
        150,
        "verification/journeys/brand_new_journey.rs",
        10,
        "run_brand_new",
    )]);

    assert_eq!(violations.len(), 1);
    assert!(matches!(
        violations[0].kind,
        TestScopeFreezeViolationKind::NewSite
    ));
}

#[test]
fn should_flag_a_frozen_test_scope_site_that_grew() {
    // Negative: frozen at 261 but now 262 — the ratchet pushed the wrong way.
    let violations = check_test_scope_freeze(&[named_finding(
        262,
        "apps/daemon/tests/fixtures/idle_worker.rs",
        41,
        "run_fixture",
    )]);

    assert_eq!(violations.len(), 1);
    assert!(matches!(
        violations[0].kind,
        TestScopeFreezeViolationKind::Grew {
            frozen_code_lines: 261
        }
    ));
}

#[test]
fn should_treat_an_unresolved_test_scope_name_as_a_new_site() {
    // Edge: name resolution returning "unknown" must not ride a stale freeze
    // entry for the same path; fail closed as a new site instead.
    let violations = check_test_scope_freeze(&[named_finding(
        200,
        "apps/daemon/tests/fixtures/idle_worker.rs",
        41,
        "unknown",
    )]);

    assert_eq!(violations.len(), 1);
    assert!(matches!(
        violations[0].kind,
        TestScopeFreezeViolationKind::NewSite
    ));
}

#[test]
fn should_expose_the_frozen_test_scope_baseline_size() {
    // Boundary: pin the baseline so every freeze-list addition is a deliberate,
    // visible change.
    assert_eq!(TEST_SCOPE_FREEZE.len(), 83);
}
