//! The lane-split contract, as typed Rust: pins the build-once, run-per-tier
//! shape of `.github/workflows/ci.yml` (issue #1553 phase 2b) — the macOS
//! build job packs one nextest archive per tier, each per-tier lane checks
//! out the source and runs its filters against its own archive without
//! compiling, and cargo-nextest is provisioned through a shared composite
//! action whose pin is verified. Companion binaries: the general structure
//! (`workflow_structural_contract`) and the merge-gate fan-in
//! (`workflow_verdict_contract`).

mod contract_support;

use contract_support::{
    find_step_by_name, required_job, required_text, required_u64, run_text_of, steps_of,
    workflow_contract,
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
        "cargo nextest-archive-test-harness",
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
        "Upload test-harness test archive",
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
fn tier_lanes_run_their_own_archives_without_compiling() {
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

    // Tier-specific filters: unit tests take the lib kind plus the harness
    // fixtures archive, integration tests take the test kind, REST keeps
    // its library subset and exact-name wire contract, and E2E runs the
    // whole archive.
    let unit_steps = steps_of(required_job(&workflow, "unit-tests"), "unit-tests");
    let unit_text = run_text_of(
        find_step_by_name(&unit_steps, "Run Unit tests from the build archive"),
        "unit",
    );
    assert!(
        unit_text.contains("-E 'kind(lib)'") && unit_text.contains("test-harness.tar.zst"),
        "the unit lane lost its lib-kind filter or the harness-fixtures archive"
    );
    // The direct-MLX runtime tests need the metallib: the archive layout
    // shifts the beside-the-executable lookup, so the unit lane must
    // download the builder's metallib and point the environment override
    // at it — and the build job must upload it.
    assert!(
        unit_text.contains("export ASTRONOMICAL_MLX_METALLIB_PATH="),
        "the unit lane lost its metallib override"
    );
    assert!(
        unit_steps.iter().any(|step| {
            step.get("uses")
                .and_then(Value::as_str)
                .map(|uses_text| uses_text.contains("actions/download-artifact"))
                .unwrap_or(false)
                && step
                    .get("with")
                    .and_then(|with| with.get("name"))
                    .and_then(Value::as_str)
                    .map(|name| name == "native-metallib")
                    .unwrap_or(false)
        }),
        "the unit lane does not download the native metallib artifact"
    );
    let build_steps = steps_of(required_job(&workflow, "rust-core"), "rust-core");
    assert!(
        build_steps.iter().any(|step| {
            step.get("name")
                .and_then(Value::as_str)
                .map(|name| name == "Upload native metallib")
                .unwrap_or(false)
        }),
        "the build job does not upload the native metallib artifact"
    );
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
    assert!(
        rest_text.contains(" library"),
        "the REST lane lost its library-subset filter"
    );
    assert!(
        rest_text.contains("-E 'test(=rest_api::application::image_status::"),
        "the REST lane lost its exact-name wire-contract filter"
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

// cargo-nextest is provisioned by a shared composite action so the build job
// and every tier lane install the same pinned, digest-verified binary.
#[test]
fn cargo_nextest_is_provisioned_by_a_pinned_verified_composite_action() {
    let action_text = std::fs::read_to_string(
        contract_support::repository_root()
            .join(".github/actions/provision-cargo-nextest/action.yml"),
    )
    .expect("the cargo-nextest composite action should exist");
    assert!(
        action_text.contains("default: \"0.9.149\""),
        "the composite action lost its pinned nextest version"
    );
    assert!(
        action_text.contains("47b9941765a9ff94e304a3fe5444ac1c7593da655525b8b69f8f2b31a9217507"),
        "the composite action lost its pinned nextest digest"
    );
    for verification_marker in ["shasum -a 256", "$NEXTEST_SHA256", "exit 1", "$GITHUB_PATH"] {
        assert!(
            action_text.contains(verification_marker),
            "the cargo-nextest composite action lost {verification_marker}"
        );
    }
    let workflow = workflow_contract();
    for job_id in [
        "rust-core",
        "unit-tests",
        "integration-tests",
        "rest-contracts",
        "e2e-journeys",
    ] {
        let job_steps = steps_of(required_job(&workflow, job_id), job_id);
        let provision_step = find_step_by_name(&job_steps, "Provision pinned cargo-nextest");
        assert_eq!(
            required_text(provision_step, "uses", job_id),
            "./.github/actions/provision-cargo-nextest",
            "job {job_id} must provision cargo-nextest through the shared composite action"
        );
    }
    // The per-test hang budget stays configured for archive runs.
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
