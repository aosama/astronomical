//! The lane-split contract, as typed Rust: pins the build-once, run-per-tier
//! shape of `.github/workflows/ci.yml` (issue #1553 phase 2b) — the macOS
//! build job packs one nextest archive per tier, each per-tier lane checks
//! out the source and runs its filters against its own archive without
//! compiling, and cargo-nextest is provisioned through one SHA-pinned,
//! digest-verifying install action. Companion binaries: the general structure
//! (`workflow_structural_contract`) and the merge-gate fan-in
//! (`workflow_verdict_contract`).

mod contract_support;

use contract_support::{
    find_step_by_name, is_pinned_action_reference, required_field, required_job, required_text,
    required_u64, run_text_of, steps_of, workflow_contract,
};
use serde_yaml::Value;

const TIER_LANES: [(&str, &str); 4] = [
    ("unit-tests", "Unit tests"),
    ("integration-tests", "Integration tests"),
    ("rest-contracts", "REST contracts"),
    ("e2e-journeys", "End-to-end journeys"),
];

#[test]
fn build_job_packs_one_archive_per_tier_lane() {
    let workflow = workflow_contract();
    let build_steps = steps_of(required_job(&workflow, "rust-core"), "rust-core");
    let archive_step = find_step_by_name(&build_steps, "Archive test binaries");
    let archive_text = run_text_of(archive_step, "archive-test-binaries");
    for archive_alias in [
        "cargo nextest-archive-integration",
        "cargo nextest-archive-rest-api",
        "cargo nextest-archive-e2e",
    ] {
        assert!(
            archive_text.contains(archive_alias),
            "the archive step lost {archive_alias}"
        );
    }
    // The build job no longer executes tests: a cargo test or nextest run
    // line here would mean the monolith crept back.
    assert!(
        !archive_text.contains("cargo test ")
            && !archive_text.contains("cargo nextest-integration"),
        "the build job still executes tests instead of only building and archiving"
    );
    for upload_name in [
        "Upload integration test archive",
        "Upload REST test archive",
        "Upload E2E test archive",
    ] {
        let upload_step = find_step_by_name(&build_steps, upload_name);
        let upload_configuration = upload_step
            .get("with")
            .expect("an archive upload must configure its inputs");
        assert_eq!(
            required_text(upload_configuration, "if-no-files-found", upload_name),
            "error",
            "{upload_name} must fail when the archive is missing"
        );
        assert_eq!(
            required_u64(upload_configuration, "retention-days", upload_name),
            1,
            "{upload_name} archives are transient and must expire quickly"
        );
        assert_eq!(
            required_u64(upload_configuration, "compression-level", upload_name),
            0,
            "{upload_name} payload is already zstd-compressed"
        );
    }
}

#[test]
fn archive_aliases_carry_the_selections_the_compile_step_builds() {
    let configuration =
        std::fs::read_to_string(contract_support::repository_root().join(".cargo/config.toml"))
            .expect("the canonical Cargo configuration should be readable");
    let alias_selection = |alias_prefix: &str| -> String {
        configuration
            .lines()
            .find(|line| line.starts_with(alias_prefix))
            .unwrap_or_else(|| panic!("the {alias_prefix} alias should exist"))
            .trim_start_matches(alias_prefix)
            .trim_matches('"')
            .to_owned()
    };
    let compile_selection = alias_selection("test-integration = ");
    let archive_selection = alias_selection("nextest-archive-integration = ");
    // Compare from the first package flag onward: the run aliases carry
    // execution-only flags (--no-fail-fast) that archiving does not.
    fn selection_tail(alias_value: &str) -> &str {
        &alias_value[alias_value
            .find("-p ")
            .expect("the alias should select packages")..]
    }
    // If the selections drifted, the lanes would run filters against
    // binaries the compile step's timing and native-progress attribution
    // never saw.
    assert_eq!(
        selection_tail(&compile_selection),
        selection_tail(&archive_selection),
        "the nextest-archive-integration alias must carry the identical selection as test-integration"
    );
    assert!(compile_selection.contains("-p astronomical-memory-policy "));
    assert!(compile_selection.contains("--lib --test integration_tests"));
}

#[test]
fn every_tier_lane_runs_its_own_archive_without_compiling() {
    let workflow = workflow_contract();
    for (lane_id, lane_name) in TIER_LANES {
        let lane = required_job(&workflow, lane_id);
        assert_eq!(
            required_text(lane, "name", &format!("job {lane_id}")),
            lane_name,
            "the {lane_id} lane must report its tier name"
        );
        assert_eq!(
            required_text(lane, "runs-on", &format!("job {lane_id}")),
            "macos-26",
            "the {lane_id} lane runs the compiled macOS binaries"
        );
        assert!(
            required_u64(lane, "timeout-minutes", &format!("job {lane_id}")) <= 10,
            "the {lane_id} lane exceeded its hard cap"
        );
        // No explicit `if`: with the default success() semantics a failed
        // build skips every lane and a legitimately skipped build skips them
        // with it — the verdict owns the fail-closed distinction.
        assert!(
            lane.get("if").is_none(),
            "the {lane_id} lane must inherit the default success() gate"
        );
        let lane_steps = steps_of(lane, lane_id);
        for step in &lane_steps {
            if let Some(step_text) = step.get("run").and_then(Value::as_str) {
                assert!(
                    !step_text
                        .lines()
                        .any(|line| line.trim_start().starts_with("cargo test")
                            || line.trim_start().starts_with("cargo build")),
                    "the {lane_id} lane compiles instead of running its archive"
                );
            }
        }
        let run_step = find_step_by_name(
            &lane_steps,
            &format!("Run {lane_name} from the build archive"),
        );
        let run_text = run_text_of(run_step, lane_id);
        assert!(
            run_text.contains("--archive-file"),
            "the {lane_id} lane runs without its build archive"
        );
    }
}

/// The direct-MLX runtime tests need the metallib: the archive layout shifts
/// the beside-the-executable lookup, so the unit lane must download the
/// builder's metallib and point the environment override at it — and the
/// build job must upload it.
fn assert_unit_lane_ships_the_native_metallib(
    unit_steps: &[&Value],
    unit_text: &str,
    build_steps: &[&Value],
) {
    assert!(
        unit_text.contains("export ASTRONOMICAL_MLX_METALLIB_PATH="),
        "the unit lane lost its metallib override"
    );
    let mut downloads_native_metallib = false;
    for step in unit_steps {
        let uses_text = step.get("uses").and_then(Value::as_str);
        let artifact_name = step
            .get("with")
            .and_then(|with| with.get("name"))
            .and_then(Value::as_str);
        let downloads_from_actions = match uses_text {
            Some(uses_text) => uses_text.contains("actions/download-artifact"),
            None => false,
        };
        if downloads_from_actions && artifact_name == Some("native-metallib") {
            downloads_native_metallib = true;
        }
    }
    assert!(
        downloads_native_metallib,
        "the unit lane does not download the native metallib artifact"
    );
    let mut uploads_native_metallib = false;
    for step in build_steps {
        if let Some("Upload native metallib") = step.get("name").and_then(Value::as_str) {
            uploads_native_metallib = true;
        }
    }
    assert!(
        uploads_native_metallib,
        "the build job does not upload the native metallib artifact"
    );
}

#[test]
fn every_tier_lane_applies_its_kind_and_archive_filters() {
    let workflow = workflow_contract();
    // Tier-specific filters: unit tests take the lib kind plus the harness
    // fixtures archive, integration tests take the test kind, and REST and
    // E2E run their whole archives.
    let unit_steps = steps_of(required_job(&workflow, "unit-tests"), "unit-tests");
    let unit_text = run_text_of(
        find_step_by_name(&unit_steps, "Run Unit tests from the build archive"),
        "unit",
    );
    assert!(
        unit_text.contains("-E 'kind(lib)'"),
        "the unit lane lost its lib-kind filter"
    );
    let build_steps = steps_of(required_job(&workflow, "rust-core"), "rust-core");
    assert_unit_lane_ships_the_native_metallib(&unit_steps, unit_text, &build_steps);
    let integration_steps = steps_of(
        required_job(&workflow, "integration-tests"),
        "integration-tests",
    );
    let integration_text = run_text_of(
        find_step_by_name(
            &integration_steps,
            "Run Integration tests from the build archive",
        ),
        "integration",
    );
    assert!(
        integration_text.contains("-E 'kind(test)'"),
        "the integration lane lost its test-kind filter"
    );
    let rest_steps = steps_of(required_job(&workflow, "rest-contracts"), "rest-contracts");
    let rest_text = run_text_of(
        find_step_by_name(&rest_steps, "Run REST contracts from the build archive"),
        "rest",
    );
    // The REST lane runs its whole archive (issue #1728): a filtered subset
    // skipped 157 of 172 contracts and let fixture drift pass CI.
    assert!(
        rest_text.contains("rest-api.tar.zst") && rest_text.contains("--no-fail-fast"),
        "the REST lane is not running its whole archive with --no-fail-fast"
    );
    // Only the command line is inspected: the step's comment explains the
    // filter it replaced and would trip a whole-text search.
    let mut rest_command_lines = Vec::new();
    for line in rest_text.lines() {
        if line.trim_start().starts_with("cargo-nextest") {
            rest_command_lines.push(line);
        }
    }
    assert_eq!(
        rest_command_lines.len(),
        1,
        "the REST lane should run exactly one cargo-nextest command"
    );
    assert!(
        !rest_command_lines[0].contains(" -E ") && !rest_command_lines[0].contains(" library"),
        "the REST lane filters its archive again instead of running every contract"
    );
    let e2e_steps = steps_of(required_job(&workflow, "e2e-journeys"), "e2e-journeys");
    let e2e_text = run_text_of(
        find_step_by_name(&e2e_steps, "Run End-to-end journeys from the build archive"),
        "e2e",
    );
    assert!(
        e2e_text.contains("e2e.tar.zst"),
        "the E2E lane is not running its own archive"
    );
}

const NEXTEST_INSTALL_ACTION: &str = "taiki-e/install-action";
const NEXTEST_TOOL_PREFIX: &str = "cargo-nextest@";

/// `cargo-nextest@0.9.149` pins an exact dotted release; a bare tool name or
/// a partial version would float to whatever the action manifest calls latest.
fn is_exact_nextest_tool_pin(tool_text: &str) -> bool {
    let Some(pinned_version) = tool_text.strip_prefix(NEXTEST_TOOL_PREFIX) else {
        return false;
    };
    let version_components: Vec<&str> = pinned_version.split('.').collect();
    let has_three_components = version_components.len() == 3;
    let mut every_component_is_numeric = true;
    for version_component in &version_components {
        let is_numeric = !version_component.is_empty()
            && version_component.bytes().all(|byte| byte.is_ascii_digit());
        every_component_is_numeric = every_component_is_numeric && is_numeric;
    }
    has_three_components && every_component_is_numeric
}

#[test]
fn should_accept_only_exact_three_component_nextest_pins() {
    assert!(is_exact_nextest_tool_pin("cargo-nextest@0.9.149"));
    assert!(!is_exact_nextest_tool_pin("cargo-nextest"));
    assert!(!is_exact_nextest_tool_pin("cargo-nextest@"));
    assert!(!is_exact_nextest_tool_pin("cargo-nextest@0.9"));
    assert!(!is_exact_nextest_tool_pin("cargo-nextest@0.9.x"));
    assert!(!is_exact_nextest_tool_pin("cargo-nextest@0.9.149.1"));
    assert!(!is_exact_nextest_tool_pin("sccache@0.9.149"));
}

// cargo-nextest is provisioned by install-action, which verifies the release
// digest it carries inside the pinned action commit. The build job and every
// tier lane must use the identical SHA-pinned reference and exact tool pin so
// archives are produced and consumed by the same verified binary, and the
// cargo-binstall fallback stays disabled so a missing release fails instead
// of compiling or fetching an unverified binary.
#[test]
fn should_provision_cargo_nextest_through_one_sha_pinned_verified_install_action() {
    let workflow = workflow_contract();
    let mut first_provision_reference: Option<(String, String)> = None;
    for job_id in [
        "rust-core",
        "unit-tests",
        "integration-tests",
        "rest-contracts",
        "e2e-journeys",
    ] {
        let job_steps = steps_of(required_job(&workflow, job_id), job_id);
        let provision_step = find_step_by_name(&job_steps, "Provision pinned cargo-nextest");
        let provision_reference = assert_verified_nextest_provisioning(provision_step, job_id);
        match &first_provision_reference {
            None => first_provision_reference = Some(provision_reference),
            Some(expected_reference) => assert_eq!(
                &provision_reference, expected_reference,
                "job {job_id} provisions a different action or nextest release than rust-core"
            ),
        }
    }
    assert_nextest_configuration_keeps_archive_budgets();
}

/// Returns the (action reference, tool pin) pair so the caller can require
/// every job to agree on both.
fn assert_verified_nextest_provisioning(provision_step: &Value, job_id: &str) -> (String, String) {
    let uses_text = required_text(provision_step, "uses", job_id);
    assert!(
        is_pinned_action_reference(uses_text, NEXTEST_INSTALL_ACTION),
        "job {job_id} must provision cargo-nextest through a SHA-pinned {NEXTEST_INSTALL_ACTION}, got {uses_text}"
    );
    let provision_inputs = required_field(provision_step, "with", job_id);
    let tool_text = required_text(provision_inputs, "tool", job_id);
    assert!(
        is_exact_nextest_tool_pin(tool_text),
        "job {job_id} must pin an exact cargo-nextest release, got {tool_text}"
    );
    assert_eq!(
        required_text(provision_inputs, "fallback", job_id),
        "none",
        "job {job_id} must disable the unverified install fallback"
    );
    assert!(
        provision_inputs.get("checksum").is_none(),
        "job {job_id} must keep install-action checksum verification at its enabled default"
    );
    (uses_text.to_owned(), tool_text.to_owned())
}

// The per-test hang budget stays configured for archive runs.
fn assert_nextest_configuration_keeps_archive_budgets() {
    let nextest_configuration =
        std::fs::read_to_string(contract_support::repository_root().join(".config/nextest.toml"))
            .expect("the nextest configuration should be readable");
    assert!(
        nextest_configuration.contains("slow-timeout")
            && nextest_configuration.contains("terminate-after")
            && nextest_configuration.contains("archive.include"),
        "the nextest configuration lost its hang budget or its archive-included binaries"
    );
}

/// Cargo unifies features per command, so a lane's archive built with a different
/// package list recompiles what the build step just compiled.
#[test]
fn should_compile_and_archive_every_lane_with_one_package_list() {
    let compile_selection = package_list_of_alias("test-integration = ");
    assert!(compile_selection.contains("-p astronomical-rest-contract"));
    assert!(compile_selection.contains("-p astronomical-e2e"));
    for archive_alias in [
        "nextest-archive-integration = ",
        "nextest-archive-rest-api = ",
        "nextest-archive-e2e = ",
    ] {
        assert_eq!(
            package_list_of_alias(archive_alias),
            compile_selection,
            "{archive_alias}must carry the compile step's package list"
        );
    }
    let workflow = workflow_contract();
    let compile_steps = steps_of(required_job(&workflow, "rust-core"), "rust-core");
    let compile_step = find_step_by_name(&compile_steps, "Compile integration lane tests");
    let compile_text = run_text_of(compile_step, "compile-integration");
    assert!(
        compile_text
            .contains("cargo test-integration --timings --no-run --test rest_api_tests --test e2e"),
        "the integration, REST and E2E lanes must compile in one command"
    );
}

fn package_list_of_alias(alias_prefix: &str) -> String {
    let cargo_config =
        std::fs::read_to_string(contract_support::repository_root().join(".cargo/config.toml"))
            .expect("the cargo config should be readable");
    let mut packages = Vec::new();
    for config_line in cargo_config.lines() {
        if !config_line.starts_with(alias_prefix) {
            continue;
        }
        let mut words = config_line.split_whitespace();
        while let Some(word) = words.next() {
            if word == "-p" {
                packages.push(words.next().expect("-p takes a package"));
            }
        }
    }
    assert!(!packages.is_empty(), "{alias_prefix} should list packages");
    packages.join(" -p ")
}

/// These two compile with their own features, so they rebuild crates. In the build job
/// every test lane would wait on them; removing them outright would lose the proof
/// that the daemon builds as a release does and the fixture library's tests.
#[test]
fn should_keep_the_standalone_builds_beside_the_build_job_rather_than_in_it() {
    let workflow = workflow_contract();
    let parallel_steps = steps_of(required_job(&workflow, "neural-engine"), "neural-engine");
    let daemon_step = find_step_by_name(&parallel_steps, "Build the daemon on its own features");
    assert!(
        run_text_of(daemon_step, "daemon-standalone-build")
            .contains("cargo build --locked --bins -p astronomical-daemon")
    );
    let fixture_step = find_step_by_name(&parallel_steps, "Run the shared test-fixture unit tests");
    assert!(
        run_text_of(fixture_step, "test-fixture-unit-tests")
            .contains("cargo test-test-harness-fixtures")
    );
    let build_steps = steps_of(required_job(&workflow, "rust-core"), "rust-core");
    let compile_step = find_step_by_name(&build_steps, "Compile integration lane tests");
    let compile_text = run_text_of(compile_step, "compile-integration-tests");
    assert!(!compile_text.contains("--bins"));
    assert!(!compile_text.contains("test-test-harness-fixtures"));
    let parallel_needs = required_field(
        required_job(&workflow, "neural-engine"),
        "needs",
        "job neural-engine",
    );
    assert_eq!(
        parallel_needs.as_str(),
        Some("detect-changes"),
        "waiting on the build job would put these back on the critical path"
    );
}

fn workflow_contract_targets(xtask_manifest_text: &str) -> Vec<&str> {
    let mut contract_targets = Vec::new();
    for manifest_line in xtask_manifest_text.lines() {
        let Some(after_name) = manifest_line.strip_prefix("name = \"") else {
            continue;
        };
        let target_name = after_name.trim_end_matches('"');
        if target_name.starts_with("workflow_") && target_name.ends_with("_contract") {
            contract_targets.push(target_name);
        }
    }
    contract_targets
}

fn contract_targets_missing_from_lane<'a>(
    contract_targets: &[&'a str],
    lane_text: &str,
) -> Vec<&'a str> {
    let mut missing_targets = Vec::new();
    for contract_target in contract_targets {
        if !lane_runs_test_target(lane_text, contract_target) {
            missing_targets.push(*contract_target);
        }
    }
    missing_targets
}

/// Line by line, so a commented-out invocation does not count as running the target.
fn lane_runs_test_target(lane_text: &str, test_target: &str) -> bool {
    let invocation_ending = format!("--test {test_target}");
    for lane_line in lane_text.lines() {
        let trimmed_line = lane_line.trim();
        if !trimmed_line.starts_with('#') && trimmed_line.ends_with(&invocation_ending) {
            return true;
        }
    }
    false
}

/// The lane names its targets one by one, so a new contract passes locally and then
/// never runs in CI. That happened to the xtask binary contract.
#[test]
fn should_run_every_workflow_contract_on_the_config_lane() {
    let xtask_manifest_text =
        std::fs::read_to_string(contract_support::repository_root().join("xtask/Cargo.toml"))
            .expect("the xtask manifest should be readable");
    let contract_targets = workflow_contract_targets(&xtask_manifest_text);
    assert!(contract_targets.contains(&"workflow_xtask_binary_contract"));
    let workflow = workflow_contract();
    let lane_steps = steps_of(
        required_job(&workflow, "config-contracts"),
        "config-contracts",
    );
    let lane_step = find_step_by_name(&lane_steps, "Run workflow and cache contracts");
    let lane_text = run_text_of(lane_step, "config-contracts");
    assert_eq!(
        contract_targets_missing_from_lane(&contract_targets, lane_text),
        Vec::<&str>::new()
    );
}

#[test]
fn should_report_a_contract_the_lane_leaves_out() {
    let contract_targets = ["workflow_listed_contract", "workflow_forgotten_contract"];
    assert_eq!(
        contract_targets_missing_from_lane(
            &contract_targets,
            "cargo test --test workflow_listed_contract\n"
        ),
        vec!["workflow_forgotten_contract"]
    );
}

#[test]
fn should_not_count_a_commented_out_contract_as_running() {
    assert!(!lane_runs_test_target(
        "# cargo test --test workflow_listed_contract\n",
        "workflow_listed_contract"
    ));
    assert!(lane_runs_test_target(
        "  cargo test --locked -p xtask --test workflow_listed_contract",
        "workflow_listed_contract"
    ));
}

#[test]
fn should_read_only_workflow_contracts_from_the_manifest() {
    let manifest_text = "name = \"xtask\"\n\
                         name = \"integration_gh_shim_journeys\"\n\
                         name = \"workflow_listed_contract\"\n";
    assert_eq!(
        workflow_contract_targets(manifest_text),
        vec!["workflow_listed_contract"]
    );
}

/// Only `[[test]]` sections: the package and any binary also have a `name` line.
fn xtask_test_targets(xtask_manifest_text: &str) -> Vec<&str> {
    let mut test_targets = Vec::new();
    let mut is_in_test_section = false;
    for manifest_line in xtask_manifest_text.lines() {
        if manifest_line.starts_with('[') {
            is_in_test_section = manifest_line.trim() == "[[test]]";
            continue;
        }
        if !is_in_test_section {
            continue;
        }
        if let Some(after_name) = manifest_line.strip_prefix("name = \"") {
            test_targets.push(after_name.trim_end_matches('"'));
        }
    }
    test_targets
}

/// `autotests = false` means a target runs only where a lane names it, so a new one
/// passes locally and then never runs in CI. Five targets sat like that.
#[test]
fn should_run_every_xtask_test_target_in_ci() {
    let xtask_manifest_text =
        std::fs::read_to_string(contract_support::repository_root().join("xtask/Cargo.toml"))
            .expect("the xtask manifest should be readable");
    let test_targets = xtask_test_targets(&xtask_manifest_text);
    assert!(test_targets.contains(&"integration_gh_shim_journeys"));
    let workflow = workflow_contract();
    let config_steps = steps_of(
        required_job(&workflow, "config-contracts"),
        "config-contracts",
    );
    let config_step = find_step_by_name(&config_steps, "Run workflow and cache contracts");
    let neural_engine_steps = steps_of(required_job(&workflow, "neural-engine"), "neural-engine");
    let xtask_tests_step =
        find_step_by_name(&neural_engine_steps, "Run the xtask unit and journey tests");
    let lanes_text = format!(
        "{}\n{}",
        run_text_of(config_step, "config-contracts"),
        run_text_of(xtask_tests_step, "xtask-unit-tests")
    );
    assert_eq!(
        contract_targets_missing_from_lane(&test_targets, &lanes_text),
        Vec::<&str>::new()
    );
}

#[test]
fn should_read_only_test_sections_from_the_manifest() {
    let manifest_text = "[package]\nname = \"xtask\"\n\n\
                         [[bin]]\nname = \"xtask\"\n\n\
                         [[test]]\nname = \"listed_journeys\"\npath = \"tests/listed.rs\"\n";
    assert_eq!(xtask_test_targets(manifest_text), vec!["listed_journeys"]);
}

#[test]
fn should_find_no_test_targets_in_a_manifest_without_test_sections() {
    let manifest_text = "[package]\nname = \"xtask\"\n";
    assert_eq!(xtask_test_targets(manifest_text), Vec::<&str>::new());
}
