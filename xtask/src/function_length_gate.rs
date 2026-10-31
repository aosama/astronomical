//! Hard function-length gate over the budget, for production scope and — as a
//! ratchet — for test scope.
//!
//! The denied budget lint already fails any build that crosses 100 code
//! lines, but a suppression attribute (`#[expect]` or `#[allow]`) silences
//! it — that is precisely how tracked debt is carried. This gate closes the
//! attribute escape: it force-warns the budget lint, which overrides every
//! suppression attribute, resolves each finding's function name, and fails
//! the run on any production finding outside the accepted exemptions, or —
//! with `--fail-on-tests` — on any test-scope finding that grows past its
//! frozen count or that the freeze list does not name.
//!
//! The accepted-debt lists are deliberate, visible data that cannot be
//! created by editing the function they cover; they live in the debt registry
//! (`function_length_debt`), keyed by source path plus function name, and
//! every entry must cite the decision that accepted the debt. This module
//! owns the gate mechanics: lane running, classification, argument parsing,
//! and outcome reporting.

use std::path::Path;
use std::time::Instant;

use crate::function_length::{
    FunctionLengthLane, OverBudgetFunction, function_length_lanes, is_test_scope,
    merge_world_findings_maximum, resolve_function_name, run_lane_with_forced_budget_lint,
};
use crate::function_length_debt::{frozen_test_scope_code_lines, is_exempted_production_site};

/// The status prefix every gate line carries, matching the report lanes'
/// convention.
const GATE_STATUS_PREFIX: &str = "[function-length-gate]";

/// One over-budget finding with its resolved function name.
#[derive(Clone, Debug)]
pub(crate) struct NamedOverBudgetFinding {
    pub(crate) finding: OverBudgetFunction,
    pub(crate) function_name: String,
}

/// The three buckets every named finding sorts into. Field order in the
/// printed summary is failure count, exempted count, test-scope count.
pub(crate) struct ProductionGateClassification {
    pub(crate) failures: Vec<NamedOverBudgetFinding>,
    pub(crate) exempted: Vec<NamedOverBudgetFinding>,
    pub(crate) test_scope: Vec<NamedOverBudgetFinding>,
}

/// Classifies named findings into gate failures, accepted exemptions, and
/// test-scope sites. Pure: values in, values out, no filesystem and no
/// suppression-attribute knowledge — the force-warn lanes already stripped
/// attributes before findings reached this point.
pub(crate) fn classify_named_findings(
    named_findings: &[NamedOverBudgetFinding],
) -> ProductionGateClassification {
    let mut failures = Vec::new();
    let mut exempted = Vec::new();
    let mut test_scope = Vec::new();
    for named_finding in named_findings {
        if is_test_scope(&named_finding.finding.path) {
            test_scope.push(named_finding.clone());
        } else if is_exempted_production_site(
            &named_finding.finding.path,
            &named_finding.function_name,
        ) {
            exempted.push(named_finding.clone());
        } else {
            failures.push(named_finding.clone());
        }
    }
    ProductionGateClassification {
        failures,
        exempted,
        test_scope,
    }
}

/// The two ways a test-scope site can break the freeze.
#[derive(Debug)]
pub(crate) enum TestScopeFreezeViolationKind {
    /// A test-scope site over budget that is not in the freeze list at all:
    /// new debt appearing where the freeze says none may.
    NewSite,
    /// A known frozen site reporting more code lines than it was frozen at:
    /// the ratchet being pushed the wrong way.
    Grew { frozen_code_lines: u64 },
}

/// One test-scope finding that breaks the freeze, plus how it breaks it.
#[derive(Debug)]
pub(crate) struct TestScopeFreezeViolation {
    pub(crate) finding: NamedOverBudgetFinding,
    pub(crate) kind: TestScopeFreezeViolationKind,
}

/// Checks the frozen test-scope sites against the observed findings. Pure:
/// values in, values out. A finding that is neither frozen nor grown is the
/// success case the ratchet exists to preserve.
pub(crate) fn check_test_scope_freeze(
    test_scope_findings: &[NamedOverBudgetFinding],
) -> Vec<TestScopeFreezeViolation> {
    let mut violations = Vec::new();
    for finding in test_scope_findings {
        match frozen_test_scope_code_lines(&finding.finding.path, &finding.function_name) {
            None => violations.push(TestScopeFreezeViolation {
                finding: finding.clone(),
                kind: TestScopeFreezeViolationKind::NewSite,
            }),
            Some(frozen_code_lines) if finding.finding.code_lines > frozen_code_lines => violations
                .push(TestScopeFreezeViolation {
                    finding: finding.clone(),
                    kind: TestScopeFreezeViolationKind::Grew { frozen_code_lines },
                }),
            Some(_) => {}
        }
    }
    violations
}

/// The gate's caller-supplied knobs. `lane_names` is None for every lane and
/// Some(comma-separated names) for an explicit subset — the pre-commit hook
/// runs the default and MLX worlds locally and leaves the acceptance feature
/// superset to CI, mirroring the untracked-budget check it accompanies.
/// At least one of the two check toggles must be set; a gate run with neither
/// would vacuously pass.
#[derive(Debug)]
pub(crate) struct GateArguments {
    pub(crate) lane_names: Option<String>,
    pub(crate) fail_on_production: bool,
    pub(crate) fail_on_tests: bool,
}

/// Parses `--fail-on-production`, `--fail-on-tests`, and `--lanes
/// NAME[,NAME...]`. Unknown input and a run with no check toggle fail the gate
/// rather than being silently ignored.
pub(crate) fn parse_gate_arguments(arguments: &[String]) -> Result<GateArguments, String> {
    let mut parsed_arguments = GateArguments {
        lane_names: None,
        fail_on_production: false,
        fail_on_tests: false,
    };
    let mut argument_index = 0usize;
    while argument_index < arguments.len() {
        match arguments[argument_index].as_str() {
            "--fail-on-production" => {
                parsed_arguments.fail_on_production = true;
                argument_index += 1;
            }
            "--fail-on-tests" => {
                parsed_arguments.fail_on_tests = true;
                argument_index += 1;
            }
            "--lanes" => {
                let Some(lane_names) = arguments.get(argument_index + 1) else {
                    return Err(format!(
                        "{GATE_STATUS_PREFIX} --lanes requires a comma-separated lane list"
                    ));
                };
                parsed_arguments.lane_names = Some(lane_names.to_owned());
                argument_index += 2;
            }
            unknown_argument => {
                return Err(format!(
                    "{GATE_STATUS_PREFIX} unknown argument {unknown_argument}; supported: --fail-on-production, --fail-on-tests, --lanes NAME[,NAME...]"
                ));
            }
        }
    }
    if !parsed_arguments.fail_on_production && !parsed_arguments.fail_on_tests {
        return Err(format!(
            "{GATE_STATUS_PREFIX} at least one of --fail-on-production or --fail-on-tests is required"
        ));
    }
    Ok(parsed_arguments)
}

/// Selects the named lanes from the full lane set, preserving the canonical
/// lane order regardless of the requested order. Every comma-separated
/// segment must name a known lane, and empty segments are refused, so a typo
/// cannot silently narrow the gate to zero lanes or run it in an order the
/// progress lines cannot be read against.
pub(crate) fn select_lanes_by_names(
    all_lanes: &[FunctionLengthLane],
    requested_names: &str,
) -> Result<Vec<FunctionLengthLane>, String> {
    let mut requested_lane_names: Vec<&str> = Vec::new();
    for requested_name in requested_names.split(',') {
        if requested_name.is_empty() {
            return Err(format!(
                "{GATE_STATUS_PREFIX} lane list contains an empty name: {requested_names}"
            ));
        }
        requested_lane_names.push(requested_name);
    }
    let mut selected_lanes = Vec::new();
    for lane in all_lanes {
        if requested_lane_names.contains(&lane.name) {
            selected_lanes.push(lane.clone());
        }
    }
    for requested_name in &requested_lane_names {
        let mut is_known_lane = false;
        for selected_lane in &selected_lanes {
            if selected_lane.name == *requested_name {
                is_known_lane = true;
                break;
            }
        }
        if !is_known_lane {
            let mut known_lane_names = String::new();
            for lane in all_lanes {
                if !known_lane_names.is_empty() {
                    known_lane_names.push_str(", ");
                }
                known_lane_names.push_str(lane.name);
            }
            return Err(format!(
                "{GATE_STATUS_PREFIX} unknown lane {requested_name}; known lanes: {known_lane_names}"
            ));
        }
    }
    if selected_lanes.is_empty() {
        return Err(format!(
            "{GATE_STATUS_PREFIX} lane list must name at least one lane"
        ));
    }
    Ok(selected_lanes)
}

/// Runs the hard production-scope budget gate: force-warns the budget lint
/// across the selected lanes (attributes cannot suppress it), resolves each
/// finding's function name, classifies the sites, and fails when any
/// production finding is not on the exemption list.
pub(crate) fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    let started_at = Instant::now();
    let parsed_arguments = parse_gate_arguments(arguments)?;
    let all_lanes = function_length_lanes();
    let selected_lanes = match &parsed_arguments.lane_names {
        Some(lane_names) => select_lanes_by_names(&all_lanes, lane_names)?,
        None => all_lanes,
    };
    println!(
        "{GATE_STATUS_PREFIX} status=start lanes={} root={}",
        selected_lanes.len(),
        repository_root.display()
    );

    let mut findings_per_lane: Vec<Vec<OverBudgetFunction>> = Vec::new();
    for (lane_index, lane) in selected_lanes.iter().enumerate() {
        let lane_findings = run_lane_with_forced_budget_lint(
            repository_root,
            GATE_STATUS_PREFIX,
            &lane.cargo_arguments,
        )?;
        println!(
            "{GATE_STATUS_PREFIX} status=progress lane={}/{} name={} findings={} elapsed_seconds={}",
            lane_index + 1,
            selected_lanes.len(),
            lane.name,
            lane_findings.len(),
            started_at.elapsed().as_secs()
        );
        findings_per_lane.push(lane_findings);
    }
    let merged_findings = merge_world_findings_maximum(findings_per_lane);

    // Name resolution is the only filesystem-touching step; it stays in this
    // imperative shell so the classification above stays pure.
    let mut named_findings: Vec<NamedOverBudgetFinding> = Vec::new();
    for finding in &merged_findings {
        let function_name = resolve_function_name(repository_root, finding);
        named_findings.push(NamedOverBudgetFinding {
            finding: finding.clone(),
            function_name,
        });
    }
    let classification = classify_named_findings(&named_findings);

    summarize_gate_result(&parsed_arguments, &classification, started_at)
}

/// Evaluates the enabled checks against the classification, prints every
/// bucket, and returns the gate outcome. Each check runs only when its toggle
/// is on, so `--fail-on-production` alone keeps today's production behavior and
/// `--fail-on-tests` alone freezes test-side debt without gating production.
fn summarize_gate_result(
    parsed_arguments: &GateArguments,
    classification: &ProductionGateClassification,
    started_at: Instant,
) -> Result<(), String> {
    let production_failures: &[NamedOverBudgetFinding] = if parsed_arguments.fail_on_production {
        &classification.failures
    } else {
        &[]
    };
    let test_scope_violations = if parsed_arguments.fail_on_tests {
        check_test_scope_freeze(&classification.test_scope)
    } else {
        Vec::new()
    };

    for exempted_finding in &classification.exempted {
        println!(
            "{GATE_STATUS_PREFIX} status=exempted path={} line={} function={} code_lines={}",
            exempted_finding.finding.path,
            exempted_finding.finding.line,
            exempted_finding.function_name,
            exempted_finding.finding.code_lines
        );
    }
    for failure_finding in production_failures {
        println!(
            "{GATE_STATUS_PREFIX} status=over_budget path={} line={} function={} code_lines={}",
            failure_finding.finding.path,
            failure_finding.finding.line,
            failure_finding.function_name,
            failure_finding.finding.code_lines
        );
    }
    for violation in &test_scope_violations {
        match violation.kind {
            TestScopeFreezeViolationKind::NewSite => {
                println!(
                    "{GATE_STATUS_PREFIX} status=new_test_site path={} line={} function={} code_lines={}",
                    violation.finding.finding.path,
                    violation.finding.finding.line,
                    violation.finding.function_name,
                    violation.finding.finding.code_lines
                );
            }
            TestScopeFreezeViolationKind::Grew { frozen_code_lines } => {
                println!(
                    "{GATE_STATUS_PREFIX} status=test_site_grew path={} line={} function={} code_lines={} frozen_code_lines={}",
                    violation.finding.finding.path,
                    violation.finding.finding.line,
                    violation.finding.function_name,
                    violation.finding.finding.code_lines,
                    frozen_code_lines
                );
            }
        }
    }
    if production_failures.is_empty() && test_scope_violations.is_empty() {
        println!(
            "{GATE_STATUS_PREFIX} status=success over_budget_production=0 test_scope_frozen_violations=0 exempted={} test_scope={} elapsed_seconds={}",
            classification.exempted.len(),
            classification.test_scope.len(),
            started_at.elapsed().as_secs()
        );
        Ok(())
    } else {
        Err(format!(
            "{GATE_STATUS_PREFIX} status=failed over_budget_production={} test_scope_frozen_violations={} exempted={} test_scope={} elapsed_seconds={} hint=split the function, or record an accepting decision on issue #1637 before adding an exemption or freeze entry",
            production_failures.len(),
            test_scope_violations.len(),
            classification.exempted.len(),
            classification.test_scope.len(),
            started_at.elapsed().as_secs()
        ))
    }
}
