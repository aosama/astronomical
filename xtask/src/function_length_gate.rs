//! Hard production-scope gate over the function-length budget.
//!
//! The denied budget lint already fails any build that crosses 100 code
//! lines, but a suppression attribute (`#[expect]` or `#[allow]`) silences
//! it — that is precisely how tracked debt is carried. This gate closes the
//! attribute escape for production sources: it force-warns the budget lint,
//! which overrides every suppression attribute, resolves each finding's
//! function name, and fails the run unless the only production findings are
//! the explicit exemptions below. Test-scope findings are reported for
//! visibility but never fail the gate; test-side length is a separate,
//! owner-tracked decision.
//!
//! An exemption is deliberate, visible, and cannot be created by editing the
//! function it covers: it lives here as data, keyed by source path plus
//! function name, and every entry must cite the decision that accepted the
//! debt.

use std::path::Path;
use std::time::Instant;

use crate::function_length::{
    FunctionLengthLane, OverBudgetFunction, function_length_lanes, is_test_scope,
    merge_world_findings_maximum, resolve_function_name, run_lane_with_forced_budget_lint,
};

/// The status prefix every gate line carries, matching the report lanes'
/// convention.
const GATE_STATUS_PREFIX: &str = "[function-length-gate]";

/// One accepted-debt site: a production function whose code length exceeds
/// the budget by explicit decision. Keyed by source path plus function name
/// (not line number, which drifts with every edit above the site).
pub(crate) struct ProductionScopeExemption {
    pub(crate) path: &'static str,
    pub(crate) function_name: &'static str,
}

/// The complete set of production functions allowed over the 100-code-line
/// budget. The campaign tracked by issue #1637 classified all remaining
/// over-budget production functions as Family D — declarative tables where
/// serialized data is spelled as code; splitting them would obscure the data
/// they encode. The identifier tables enumerate one arm per catalog entry and
/// the tensor-profile builders enumerate one statement per physical tensor.
/// Adding an entry requires an accepting decision on the tracking issue;
/// anything else over budget in production scope fails the gate.
pub(crate) const PRODUCTION_SCOPE_EXEMPTIONS: &[ProductionScopeExemption] = &[
    ProductionScopeExemption {
        path: "crates/model-serving/src/performance_attribution/catalog.rs",
        function_name: "identifier",
    },
    ProductionScopeExemption {
        path: "crates/model-serving/src/performance_attribution/counter_catalog.rs",
        function_name: "identifier",
    },
    ProductionScopeExemption {
        path: "crates/model-serving/src/qwen_image_21/tensor_profiles/text_encoder.rs",
        function_name: "text_encoder_tensor_profiles",
    },
    ProductionScopeExemption {
        path: "crates/model-serving/src/qwen_image_21/tensor_profiles/vae.rs",
        function_name: "vae_tensor_profiles",
    },
];

/// Matches a finding site against the exemption list. Path plus name must
/// both match: two `identifier` tables exist, and the path is what tells
/// them apart. An unresolved name ("unknown") matches nothing, so a moved or
/// mangled source file fails the gate instead of silently keeping its
/// exemption.
pub(crate) fn is_exempted_production_site(source_path: &str, function_name: &str) -> bool {
    for exemption in PRODUCTION_SCOPE_EXEMPTIONS {
        if exemption.path == source_path && exemption.function_name == function_name {
            return true;
        }
    }
    false
}

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

/// The gate's caller-supplied knobs. `lane_names` is None for every lane and
/// Some(comma-separated names) for an explicit subset — the pre-commit hook
/// runs the default and MLX worlds locally and leaves the acceptance feature
/// superset to CI, mirroring the untracked-budget check it accompanies.
#[derive(Debug)]
pub(crate) struct GateArguments {
    pub(crate) lane_names: Option<String>,
}

/// Parses `--lanes NAME[,NAME...]`. Unknown input fails the gate rather than
/// being silently ignored.
pub(crate) fn parse_gate_arguments(arguments: &[String]) -> Result<GateArguments, String> {
    let mut parsed_arguments = GateArguments { lane_names: None };
    let mut argument_index = 0usize;
    while argument_index < arguments.len() {
        match arguments[argument_index].as_str() {
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
                    "{GATE_STATUS_PREFIX} unknown argument {unknown_argument}; supported: --lanes NAME[,NAME...]"
                ));
            }
        }
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

    for exempted_finding in &classification.exempted {
        println!(
            "{GATE_STATUS_PREFIX} status=exempted path={} line={} function={} code_lines={}",
            exempted_finding.finding.path,
            exempted_finding.finding.line,
            exempted_finding.function_name,
            exempted_finding.finding.code_lines
        );
    }
    for failure_finding in &classification.failures {
        println!(
            "{GATE_STATUS_PREFIX} status=over_budget path={} line={} function={} code_lines={}",
            failure_finding.finding.path,
            failure_finding.finding.line,
            failure_finding.function_name,
            failure_finding.finding.code_lines
        );
    }
    if classification.failures.is_empty() {
        println!(
            "{GATE_STATUS_PREFIX} status=success over_budget_production=0 exempted={} test_scope={} elapsed_seconds={}",
            classification.exempted.len(),
            classification.test_scope.len(),
            started_at.elapsed().as_secs()
        );
        Ok(())
    } else {
        Err(format!(
            "{GATE_STATUS_PREFIX} status=failed over_budget_production={} exempted={} test_scope={} elapsed_seconds={} hint=split the function, or record an accepting decision on issue #1637 before adding an exemption entry",
            classification.failures.len(),
            classification.exempted.len(),
            classification.test_scope.len(),
            started_at.elapsed().as_secs()
        ))
    }
}
