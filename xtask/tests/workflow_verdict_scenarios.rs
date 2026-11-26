//! The lane-outcome scenarios the verdict script is executed under: each
//! names the results every lane reported and whether the gate must pass, and
//! the script from the workflow's own verdict step is run through `bash` for
//! each so the decision logic is proven, not merely quoted. Kept beside the
//! verdict contract so that file stays a short list of the facts it pins.

use crate::{MACOS_GATED_LANES, with_untouched_linux_lanes_skipped};

/// A scenario's per-lane results for the seven macOS-gated lanes; scenarios
/// omit a lane to mean `success`.
pub(crate) struct Scenario<'a> {
    name: &'static str,
    classification: &'static str,
    macos_required: &'static str,
    macos_lane_results: &'a [(&'static str, &'static str)],
    static_checks: &'static str,
    config_contracts: &'static str,
    notices_required: &'static str,
    notices_freshness: &'static str,
    expected_pass: bool,
}

/// Executes the verdict script template under every scenario and asserts the
/// gate decided each one as the scenario expects.
pub(crate) fn assert_verdict_script_decides_every_scenario(verdict_template: &str) {
    let all_macos_lanes_skipped = MACOS_GATED_LANES.map(|lane| (lane, "skipped"));
    let mut scenarios = green_run_and_test_lane_failure_scenarios(&all_macos_lanes_skipped);
    scenarios.extend(skipped_lane_and_classification_scenarios(
        &all_macos_lanes_skipped,
    ));
    scenarios.extend(cancelled_lane_and_notices_scenarios());
    for scenario in scenarios {
        let scenario_script = scenario_verdict_script(verdict_template, &scenario);
        let verdict_output = run_scenario_script(&scenario, &scenario_script);
        assert_scenario_decided_correctly(&scenario, &verdict_output);
    }
}

fn green_run_and_test_lane_failure_scenarios<'a>(
    all_macos_lanes_skipped: &'a [(&'static str, &'static str)],
) -> Vec<Scenario<'a>> {
    vec![
        Scenario {
            name: "every lane green",
            classification: "success",
            macos_required: "true",
            macos_lane_results: &[],
            static_checks: "success",
            config_contracts: "success",
            notices_required: "false",
            notices_freshness: "skipped",
            expected_pass: true,
        },
        Scenario {
            name: "classifier-approved macOS skips for a site-only change",
            classification: "success",
            macos_required: "false",
            macos_lane_results: all_macos_lanes_skipped,
            static_checks: "success",
            config_contracts: "success",
            notices_required: "false",
            notices_freshness: "skipped",
            expected_pass: true,
        },
        Scenario {
            name: "a failed build job fails the gate",
            classification: "success",
            macos_required: "true",
            macos_lane_results: &[("rust-core", "failure")],
            static_checks: "success",
            config_contracts: "success",
            notices_required: "false",
            notices_freshness: "skipped",
            expected_pass: false,
        },
        Scenario {
            name: "a failed unit-test lane fails the gate",
            classification: "success",
            macos_required: "true",
            macos_lane_results: &[("unit-tests", "failure")],
            static_checks: "success",
            config_contracts: "success",
            notices_required: "false",
            notices_freshness: "skipped",
            expected_pass: false,
        },
        Scenario {
            name: "a failed integration-test lane fails the gate",
            classification: "success",
            macos_required: "true",
            macos_lane_results: &[("integration-tests", "failure")],
            static_checks: "success",
            config_contracts: "success",
            notices_required: "false",
            notices_freshness: "skipped",
            expected_pass: false,
        },
    ]
}

fn skipped_lane_and_classification_scenarios<'a>(
    all_macos_lanes_skipped: &'a [(&'static str, &'static str)],
) -> Vec<Scenario<'a>> {
    vec![
        Scenario {
            name: "a lane skipped while classification requires macOS work",
            classification: "success",
            macos_required: "true",
            macos_lane_results: &[("e2e-journeys", "skipped")],
            static_checks: "success",
            config_contracts: "success",
            notices_required: "false",
            notices_freshness: "skipped",
            expected_pass: false,
        },
        Scenario {
            name: "a failed neural-engine lane fails the gate",
            classification: "success",
            macos_required: "true",
            macos_lane_results: &[("neural-engine", "failure")],
            static_checks: "success",
            config_contracts: "success",
            notices_required: "false",
            notices_freshness: "skipped",
            expected_pass: false,
        },
        Scenario {
            name: "a neural-engine lane skipped while classification requires macOS work",
            classification: "success",
            macos_required: "true",
            macos_lane_results: &[("neural-engine", "skipped")],
            static_checks: "success",
            config_contracts: "success",
            notices_required: "false",
            notices_freshness: "skipped",
            expected_pass: false,
        },
        Scenario {
            name: "classification failure fails the gate",
            classification: "failure",
            macos_required: "true",
            macos_lane_results: &[],
            static_checks: "success",
            config_contracts: "success",
            notices_required: "false",
            notices_freshness: "skipped",
            expected_pass: false,
        },
        Scenario {
            name: "a skipped static lane fails the gate",
            classification: "success",
            macos_required: "false",
            macos_lane_results: all_macos_lanes_skipped,
            static_checks: "skipped",
            config_contracts: "success",
            notices_required: "false",
            notices_freshness: "skipped",
            expected_pass: false,
        },
    ]
}

fn cancelled_lane_and_notices_scenarios() -> Vec<Scenario<'static>> {
    vec![
        Scenario {
            name: "a cancelled lane fails the gate",
            classification: "success",
            macos_required: "true",
            macos_lane_results: &[("swift-node", "cancelled")],
            static_checks: "success",
            config_contracts: "success",
            notices_required: "false",
            notices_freshness: "skipped",
            expected_pass: false,
        },
        Scenario {
            name: "a failed config-contracts lane fails the gate",
            classification: "success",
            macos_required: "true",
            macos_lane_results: &[],
            static_checks: "success",
            config_contracts: "failure",
            notices_required: "false",
            notices_freshness: "skipped",
            expected_pass: false,
        },
        Scenario {
            name: "a lockfile change with fresh notices passes",
            classification: "success",
            macos_required: "true",
            macos_lane_results: &[],
            static_checks: "success",
            config_contracts: "success",
            notices_required: "true",
            notices_freshness: "success",
            expected_pass: true,
        },
        Scenario {
            name: "a lockfile change with stale notices fails the gate",
            classification: "success",
            macos_required: "true",
            macos_lane_results: &[],
            static_checks: "success",
            config_contracts: "success",
            notices_required: "true",
            notices_freshness: "failure",
            expected_pass: false,
        },
        Scenario {
            name: "a notices lane skipped while classification requires it fails the gate",
            classification: "success",
            macos_required: "true",
            macos_lane_results: &[],
            static_checks: "success",
            config_contracts: "success",
            notices_required: "true",
            notices_freshness: "skipped",
            expected_pass: false,
        },
        Scenario {
            name: "a cancelled notices lane fails the gate",
            classification: "success",
            macos_required: "true",
            macos_lane_results: &[],
            static_checks: "success",
            config_contracts: "success",
            notices_required: "true",
            notices_freshness: "cancelled",
            expected_pass: false,
        },
    ]
}

fn scenario_verdict_script(verdict_template: &str, scenario: &Scenario) -> String {
    let mut scenario_script = verdict_template
        .replace(
            "${{ needs.detect-changes.outputs.macos_verification_required }}",
            scenario.macos_required,
        )
        .replace(
            "${{ needs.detect-changes.outputs.swift_node_required }}",
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
        )
        .replace(
            "${{ needs.detect-changes.outputs.notices_inputs_changed }}",
            scenario.notices_required,
        )
        .replace(
            "${{ needs.notices-freshness.result }}",
            scenario.notices_freshness,
        );
    // These scenarios change no catalog or plugin input, so those lanes skip.
    scenario_script = with_untouched_linux_lanes_skipped(scenario_script);
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
    scenario_script
}

fn run_scenario_script(scenario: &Scenario, scenario_script: &str) -> std::process::Output {
    let script_path = std::env::temp_dir().join("astronomical-verdict-scenario.sh");
    std::fs::write(&script_path, scenario_script)
        .expect("the verdict scenario script should persist");
    std::process::Command::new("bash")
        .arg(&script_path)
        .output()
        .unwrap_or_else(|failure| panic!("scenario {} should execute: {failure}", scenario.name))
}

fn assert_scenario_decided_correctly(scenario: &Scenario, verdict_output: &std::process::Output) {
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
