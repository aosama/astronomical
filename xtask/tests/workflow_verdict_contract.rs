//! The verdict fan-in contract, as typed Rust: parses
//! `.github/workflows/ci.yml` and pins the one required check that decides
//! the merge gate from every lane's terminal result (issue #1553), then
//! executes the workflow's real verdict script under every lane-outcome
//! scenario so the decision logic is proven, not merely quoted. The general
//! workflow structure lives in the companion
//! `workflow_structural_contract` test binary.

mod contract_support;

use contract_support::{
    find_step_by_name, required_field, required_job, required_text, required_u64, run_text_of,
    steps_of, workflow_contract,
};

// The structural half: one required check — "CI verdict" — so lane jobs may
// split or rename without branch-protection coordination. The `if:
// always()` pin is the security-critical piece: a verdict that inherited the
// default success-only condition would be skipped whenever a lane failed,
// and GitHub treats a skipped required check as satisfied — a failing lane
// would merge. The needs pin is the other half: a lane removed from the
// needs list escapes the gate entirely.
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
        needed_lanes,
        [
            "detect-changes",
            "rust-core",
            "swift-node",
            "static",
            "config-contracts"
        ],
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
    for lane_marker in [
        "needs.detect-changes.result",
        "needs.detect-changes.outputs.macos_verification_required",
        "needs.rust-core.result",
        "needs.swift-node.result",
        "needs.static.result",
        "needs.config-contracts.result",
    ] {
        assert!(
            verdict_text.contains(lane_marker),
            "the verdict stopped consulting {lane_marker}"
        );
    }
    assert!(
        verdict_text.contains("exit 1"),
        "the verdict does not fail closed"
    );
    // A skip is legitimate only under the classifier's decision: the macOS
    // lanes may skip when macOS verification is not required; every other
    // lane must reach success on its own.
    assert!(
        verdict_text.contains("skipped while classification requires macOS verification"),
        "the verdict lost its skip-legitimacy check"
    );
    assert!(
        verdict_text.contains("lane=static"),
        "the static lane escaped the verdict"
    );
    assert!(
        verdict_text.contains("lane=config-contracts"),
        "the config-contracts lane escaped the verdict"
    );
}

// The behavioral half: executes the verdict step's real script (pulled from
// the parsed workflow, so it cannot drift) under every lane-outcome
// scenario. A pin proves the script says the right things; this proves it
// decides correctly: the fail-closed paths (lane failure, illegitimate
// skip, classification failure, static not green, a cancelled lane) and the
// two legitimate outcomes (all green; classifier-approved macOS skips for
// site-only changes).
#[test]
fn verdict_script_decides_every_scenario_correctly() {
    let workflow = workflow_contract();
    let verdict_job = required_job(&workflow, "verdict");
    let verdict_steps = steps_of(verdict_job, "verdict");
    let verdict_step = find_step_by_name(
        &verdict_steps,
        "Fail closed unless every lane is green or legitimately skipped",
    );
    let verdict_template = run_text_of(verdict_step, "verdict");
    struct Scenario {
        name: &'static str,
        classification: &'static str,
        macos_required: &'static str,
        rust_core: &'static str,
        swift_node: &'static str,
        static_checks: &'static str,
        config_contracts: &'static str,
        expected_pass: bool,
    }
    let scenarios = [
        Scenario {
            name: "every lane green",
            classification: "success",
            macos_required: "true",
            rust_core: "success",
            swift_node: "success",
            static_checks: "success",
            config_contracts: "success",
            expected_pass: true,
        },
        Scenario {
            name: "classifier-approved macOS skips for a site-only change",
            classification: "success",
            macos_required: "false",
            rust_core: "skipped",
            swift_node: "skipped",
            static_checks: "success",
            config_contracts: "success",
            expected_pass: true,
        },
        Scenario {
            name: "a failed macOS lane fails the gate",
            classification: "success",
            macos_required: "true",
            rust_core: "failure",
            swift_node: "success",
            static_checks: "success",
            config_contracts: "success",
            expected_pass: false,
        },
        Scenario {
            name: "a lane skipped while classification requires macOS work",
            classification: "success",
            macos_required: "true",
            rust_core: "skipped",
            swift_node: "success",
            static_checks: "success",
            config_contracts: "success",
            expected_pass: false,
        },
        Scenario {
            name: "classification failure fails the gate",
            classification: "failure",
            macos_required: "true",
            rust_core: "success",
            swift_node: "success",
            static_checks: "success",
            config_contracts: "success",
            expected_pass: false,
        },
        Scenario {
            name: "a skipped static lane fails the gate",
            classification: "success",
            macos_required: "false",
            rust_core: "skipped",
            swift_node: "skipped",
            static_checks: "skipped",
            config_contracts: "success",
            expected_pass: false,
        },
        Scenario {
            name: "a cancelled lane fails the gate",
            classification: "success",
            macos_required: "true",
            rust_core: "success",
            swift_node: "cancelled",
            static_checks: "success",
            config_contracts: "success",
            expected_pass: false,
        },
        Scenario {
            name: "a failed config-contracts lane fails the gate",
            classification: "success",
            macos_required: "true",
            rust_core: "success",
            swift_node: "success",
            static_checks: "success",
            config_contracts: "failure",
            expected_pass: false,
        },
    ];
    for scenario in scenarios {
        let scenario_script = verdict_template
            .replace(
                "${{ needs.detect-changes.outputs.macos_verification_required }}",
                scenario.macos_required,
            )
            .replace(
                "${{ needs.detect-changes.result }}",
                scenario.classification,
            )
            .replace("${{ needs.rust-core.result }}", scenario.rust_core)
            .replace("${{ needs.swift-node.result }}", scenario.swift_node)
            .replace("${{ needs.static.result }}", scenario.static_checks)
            .replace(
                "${{ needs.config-contracts.result }}",
                scenario.config_contracts,
            );
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
