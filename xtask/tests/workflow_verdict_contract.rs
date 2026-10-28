//! The verdict fan-in contract, as typed Rust: parses
//! `.github/workflows/ci.yml` and pins the one required check that decides
//! the merge gate from every lane's terminal result (issue #1553), then
//! executes the workflow's real verdict script under every lane-outcome
//! scenario so the decision logic is proven, not merely quoted. The general
//! workflow structure lives in the companion `workflow_structural_contract`
//! test binary; the per-tier lane split in `workflow_lane_split_contract`.

mod contract_support;

use contract_support::{
    find_step_by_name, required_field, required_job, required_text, required_u64, run_text_of,
    steps_of, workflow_contract,
};

/// The lanes the verdict must fan in, in workflow order: the classifier, the
/// macOS build job, the four per-tier test lanes it feeds, the second macOS
/// contract job, and the two ubuntu lanes. A lane removed from this list
/// escapes the merge gate entirely.
const EXPECTED_VERDICT_LANES: [&str; 9] = [
    "detect-changes",
    "rust-core",
    "unit-tests",
    "integration-tests",
    "rest-contracts",
    "e2e-journeys",
    "swift-node",
    "static",
    "config-contracts",
];

#[test]
fn verdict_fan_in_fails_closed_over_every_lane() {
    let workflow = workflow_contract();
    let verdict_job = required_job(&workflow, "verdict");
    assert_eq!(
        required_text(verdict_job, "name", "job verdict"),
        "CI verdict",
        "the required verdict check name changed"
    );
    assert_eq!(
        required_text(verdict_job, "if", "job verdict"),
        "${{ always() }}",
        "the verdict must run even when a lane failed; a skipped verdict satisfies required checks and lets failures merge"
    );
    let verdict_needs = required_field(verdict_job, "needs", "job verdict");
    let needed_lanes: Vec<&str> = verdict_needs
        .as_sequence()
        .expect("verdict needs must be a sequence")
        .iter()
        .map(|lane| lane.as_str().expect("verdict needs must be strings"))
        .collect();
    assert_eq!(
        needed_lanes, EXPECTED_VERDICT_LANES,
        "the verdict must fan in every lane; a lane outside its needs list escapes the merge gate"
    );
    assert_eq!(
        required_u64(verdict_job, "timeout-minutes", "job verdict"),
        2,
        "the verdict exceeded its hard cap"
    );
    assert_eq!(
        required_text(verdict_job, "runs-on", "job verdict"),
        "ubuntu-24.04",
        "the verdict is a decision node, not a macOS workload"
    );
    let verdict_steps = steps_of(verdict_job, "verdict");
    let verdict_step = find_step_by_name(
        &verdict_steps,
        "Fail closed unless every lane is green or legitimately skipped",
    );
    let verdict_text = run_text_of(verdict_step, "verdict");
    for lane in EXPECTED_VERDICT_LANES {
        assert!(
            verdict_text.contains(&format!("needs.{lane}.result")),
            "the verdict stopped consulting {lane}"
        );
    }
    assert!(
        verdict_text.contains("needs.detect-changes.outputs.macos_verification_required"),
        "the verdict lost its classifier input"
    );
    assert!(
        verdict_text.contains("exit 1"),
        "the verdict does not fail closed"
    );
    // A skip is legitimate only under the classifier's decision: the macOS
    // build job and every lane it feeds may skip when macOS verification is
    // not required; each other lane must reach success on its own.
    assert!(
        verdict_text.contains("skipped while classification requires macOS verification"),
        "the verdict lost its skip-legitimacy check"
    );
    for lane in ["static", "config-contracts"] {
        assert!(
            verdict_text.contains(&format!("lane={lane}")),
            "the {lane} lane escaped the verdict"
        );
    }
}

/// A scenario's per-lane results for the six macOS-gated lanes; scenarios
/// omit a lane to mean `success`.
struct Scenario<'a> {
    name: &'static str,
    classification: &'static str,
    macos_required: &'static str,
    macos_lane_results: &'a [(&'static str, &'static str)],
    static_checks: &'static str,
    config_contracts: &'static str,
    expected_pass: bool,
}

const MACOS_GATED_LANES: [&str; 6] = [
    "rust-core",
    "unit-tests",
    "integration-tests",
    "rest-contracts",
    "e2e-journeys",
    "swift-node",
];

#[test]
#[expect(clippy::too_many_lines)]
fn verdict_script_decides_every_scenario_correctly() {
    let workflow = workflow_contract();
    let verdict_job = required_job(&workflow, "verdict");
    let verdict_steps = steps_of(verdict_job, "verdict");
    let verdict_step = find_step_by_name(
        &verdict_steps,
        "Fail closed unless every lane is green or legitimately skipped",
    );
    let verdict_template = run_text_of(verdict_step, "verdict");
    let scenarios = [
        Scenario {
            name: "every lane green",
            classification: "success",
            macos_required: "true",
            macos_lane_results: &[],
            static_checks: "success",
            config_contracts: "success",
            expected_pass: true,
        },
        Scenario {
            name: "classifier-approved macOS skips for a site-only change",
            classification: "success",
            macos_required: "false",
            macos_lane_results: &MACOS_GATED_LANES.map(|lane| (lane, "skipped")),
            static_checks: "success",
            config_contracts: "success",
            expected_pass: true,
        },
        Scenario {
            name: "a failed build job fails the gate",
            classification: "success",
            macos_required: "true",
            macos_lane_results: &[("rust-core", "failure")],
            static_checks: "success",
            config_contracts: "success",
            expected_pass: false,
        },
        Scenario {
            name: "a failed unit-test lane fails the gate",
            classification: "success",
            macos_required: "true",
            macos_lane_results: &[("unit-tests", "failure")],
            static_checks: "success",
            config_contracts: "success",
            expected_pass: false,
        },
        Scenario {
            name: "a failed integration-test lane fails the gate",
            classification: "success",
            macos_required: "true",
            macos_lane_results: &[("integration-tests", "failure")],
            static_checks: "success",
            config_contracts: "success",
            expected_pass: false,
        },
        Scenario {
            name: "a lane skipped while classification requires macOS work",
            classification: "success",
            macos_required: "true",
            macos_lane_results: &[("e2e-journeys", "skipped")],
            static_checks: "success",
            config_contracts: "success",
            expected_pass: false,
        },
        Scenario {
            name: "classification failure fails the gate",
            classification: "failure",
            macos_required: "true",
            macos_lane_results: &[],
            static_checks: "success",
            config_contracts: "success",
            expected_pass: false,
        },
        Scenario {
            name: "a skipped static lane fails the gate",
            classification: "success",
            macos_required: "false",
            macos_lane_results: &MACOS_GATED_LANES.map(|lane| (lane, "skipped")),
            static_checks: "skipped",
            config_contracts: "success",
            expected_pass: false,
        },
        Scenario {
            name: "a cancelled lane fails the gate",
            classification: "success",
            macos_required: "true",
            macos_lane_results: &[("swift-node", "cancelled")],
            static_checks: "success",
            config_contracts: "success",
            expected_pass: false,
        },
        Scenario {
            name: "a failed config-contracts lane fails the gate",
            classification: "success",
            macos_required: "true",
            macos_lane_results: &[],
            static_checks: "success",
            config_contracts: "failure",
            expected_pass: false,
        },
    ];
    for scenario in scenarios {
        let mut scenario_script = verdict_template
            .replace(
                "${{ needs.detect-changes.outputs.macos_verification_required }}",
                scenario.macos_required,
            )
            .replace(
                "${{ needs.detect-changes.result }}",
                scenario.classification,
            )
            .replace("${{ needs.static.result }}", scenario.static_checks)
            .replace(
                "${{ needs.config-contracts.result }}",
                scenario.config_contracts,
            );
        for lane in MACOS_GATED_LANES {
            let lane_result = scenario
                .macos_lane_results
                .iter()
                .find(|(result_lane, _)| *result_lane == lane)
                .map(|(_, lane_result)| *lane_result)
                .unwrap_or("success");
            scenario_script =
                scenario_script.replace(&format!("${{{{ needs.{lane}.result }}}}"), lane_result);
        }
        let script_path = std::env::temp_dir().join("astronomical-verdict-scenario.sh");
        std::fs::write(&script_path, scenario_script)
            .expect("the verdict scenario script should persist");
        let verdict_output = std::process::Command::new("bash")
            .arg(&script_path)
            .output()
            .unwrap_or_else(|failure| {
                panic!("scenario {} should execute: {failure}", scenario.name)
            });
        assert_eq!(
            verdict_output.status.success(),
            scenario.expected_pass,
            "scenario {} decided wrong; stdout:\n{}",
            scenario.name,
            String::from_utf8_lossy(&verdict_output.stdout)
        );
        let verdict_stdout = String::from_utf8_lossy(&verdict_output.stdout);
        if scenario.expected_pass {
            assert!(
                verdict_stdout.contains("[ci-verdict] status=pass"),
                "scenario {} passed without the pass marker",
                scenario.name
            );
        } else {
            assert!(
                verdict_stdout.contains("[ci-verdict] status=fail")
                    || String::from_utf8_lossy(&verdict_output.stderr).contains("status=fail"),
                "scenario {} failed without an attributable reason",
                scenario.name
            );
        }
    }
}
