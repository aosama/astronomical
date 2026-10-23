//! Offline contract journeys for `cargo xtask disposable-target`, the
//! typed port of the retired `scripts/run-in-disposable-cargo-target.sh`.
//!
//! These journeys absorb the two retired shell contracts
//! (`test-cargo-artifact-lifecycle-contract.sh` and
//! `test-cargo-artifact-cleanup-signal-contract.sh`): the complete
//! disposable Cargo-target journey with a minimal real Cargo build, callers
//! entering from any directory, nested lanes sharing one owner, owned
//! targets removed after success, failure, or interruption, unowned
//! symlinks refused, and interruption during storage measurement
//! retaining exactly one marker-owned target.

mod disposable_journey_support;

use disposable_journey_support::process::{
    LIFECYCLE_MARKER_NAME, du_kibibytes, exit_status_code, process_is_alive, send_terminate,
    arm_termination_watchdog, sccache_is_on_path,
};
use disposable_journey_support::sandbox::{
    DisposableSandbox, assert_lane_root_is_empty, assert_path_is_absent, canonicalized,
    capture_fixture_body, file_text, journey_log_text, make_executable, run_journey,
    sanitized_runner_command, wait_for_file,
};
use std::path::{Path, PathBuf};
use std::process::Stdio;

#[test]
fn should_complete_journey_from_external_directory_and_remove_owned_target() {
    let sandbox = DisposableSandbox::install();
    let record_path = sandbox.root_directory.join("target-record");
    let fixture_path =
        sandbox.write_executable("capture-fixture.sh", &capture_fixture_body());
    let mut command_builder = sanitized_runner_command(&sandbox);
    command_builder
        .env("RUSTC_WRAPPER", "")
        .arg("--lane")
        .arg("direct-mlx")
        .arg("--")
        .arg(&fixture_path)
        .arg(&record_path);
    let journey_output = command_builder
        .output()
        .expect("the disposable-target journey should run");
    let log_text = journey_log_text(&journey_output);

    assert!(journey_output.status.success(), "journey log:\n{log_text}");
    let recorded_target = file_text(&record_path).trim().to_owned();
    let expected_lane_prefix = sandbox
        .lane_root_directory
        .join("astronomical-cargo-direct-mlx.")
        .to_string_lossy()
        .into_owned();
    assert!(
        recorded_target.starts_with(&expected_lane_prefix),
        "disposable Cargo target escaped its lane root: {recorded_target}"
    );
    let recorded_working_directory = file_text(&sandbox.root_directory.join("target-record.working-directory"))
        .trim()
        .to_owned();
    assert_eq!(
        recorded_working_directory,
        canonicalized(&sandbox.repository_directory).to_string_lossy(),
        "the disposable command must run from the repository root"
    );
    let recorded_rustc_wrapper = file_text(&sandbox.root_directory.join("target-record.rustc-wrapper"))
        .trim()
        .to_owned();
    let expected_rustc_wrapper = if sccache_is_on_path() {
        "sccache"
    } else {
        "none"
    };
    assert_eq!(
        recorded_rustc_wrapper, expected_rustc_wrapper,
        "an empty RUSTC_WRAPPER must fall back to sccache detection"
    );
    for expected_status in ["start", "cleanup-start", "removing", "removed"] {
        assert!(
            log_text.contains(&format!("status={expected_status}")),
            "the lifecycle log should report status={expected_status}: {log_text}"
        );
    }
    assert_path_is_absent(Path::new(&recorded_target));
    assert_lane_root_is_empty(&sandbox.lane_root_directory);
}

#[test]
fn should_use_canonical_default_temporary_root_when_lane_root_is_unset() {
    let sandbox = DisposableSandbox::install();
    let record_path = sandbox.root_directory.join("default-root-record");
    let fixture_path =
        sandbox.write_executable("capture-fixture.sh", &capture_fixture_body());
    let mut command_builder = sanitized_runner_command(&sandbox);
    command_builder
        .env_remove("ASTRONOMICAL_CARGO_LANE_ROOT")
        .env_remove("TMPDIR")
        .arg("--lane")
        .arg("default-root")
        .arg("--")
        .arg(&fixture_path)
        .arg(&record_path);
    let journey_output = command_builder
        .output()
        .expect("the disposable-target journey should run");
    assert!(
        journey_output.status.success(),
        "journey log:\n{}",
        journey_log_text(&journey_output)
    );
    let recorded_target = file_text(&record_path).trim().to_owned();
    assert!(
        recorded_target.starts_with('/')
            && recorded_target.contains("/astronomical-cargo-default-root."),
        "the default temporary Cargo target must be canonical and absolute: {recorded_target}"
    );
    assert_path_is_absent(Path::new(&recorded_target));
}

#[test]
fn should_build_a_real_cargo_package_and_not_reuse_targets_across_version_churn() {
    let sandbox = DisposableSandbox::install();
    std::fs::create_dir_all(sandbox.repository_directory.join("src"))
        .expect("the fixture package source directory should be creatable");
    std::fs::write(
        sandbox.repository_directory.join("src/main.rs"),
        "fn main() {\n    println!(\"Romeo and Juliet\");\n}\n",
    )
    .expect("the fixture package main should be writable");
    let package_manifest = sandbox.repository_directory.join("Cargo.toml");
    let write_manifest = |package_version: &str| {
        std::fs::write(
            &package_manifest,
            format!(
                "[package]\nname = \"cargo-lifecycle-fixture\"\nversion = \"{package_version}\"\nedition = \"2024\"\n"
            ),
        )
        .expect("the fixture package manifest should be writable");
    };
    write_manifest("0.2.10");
    let build_and_record_fixture_path = sandbox.write_executable(
        "build-and-record-fixture.sh",
        &format!(
            "#!/bin/sh\n\
             set -eu\n\
             printf '%s\\n' \"${{CARGO_TARGET_DIR:?}}\" > \"{record}\"\n\
             printf '%s\\n' \"$PWD/Cargo.toml\" > \"{record}.manifest-path\"\n\
             exec cargo build --verbose\n",
            record = sandbox.root_directory.join("target-record").display()
        ),
    );

    let (first_journey, first_log) = run_journey(
        &sandbox,
        "version-churn",
        &build_and_record_fixture_path,
    );
    assert!(
        first_journey.status.success(),
        "first journey log:\n{first_log}"
    );
    let first_target = file_text(&sandbox.root_directory.join("target-record"))
        .trim()
        .to_owned();
    assert!(
        first_target.contains("astronomical-cargo-version-churn."),
        "the first journey should own a version-churn lane target: {first_target}"
    );

    write_manifest("0.2.11");
    let (second_journey, second_log) = run_journey(
        &sandbox,
        "version-churn",
        &build_and_record_fixture_path,
    );
    assert!(
        second_journey.status.success(),
        "second journey log:\n{second_log}"
    );
    let second_target = file_text(&sandbox.root_directory.join("target-record"))
        .trim()
        .to_owned();
    assert_ne!(
        first_target, second_target,
        "package-version journeys unexpectedly reused one disposable target"
    );
    assert_path_is_absent(Path::new(&first_target));
    assert_path_is_absent(Path::new(&second_target));
    assert_lane_root_is_empty(&sandbox.lane_root_directory);
}

#[test]
fn should_keep_steady_state_storage_bounded_across_journeys() {
    let sandbox = DisposableSandbox::install();
    let bounded_storage_root = sandbox.root_directory.join("bounded-storage");
    let retained_target_directory = bounded_storage_root.join("retained-target");
    let bounded_lane_root = bounded_storage_root.join("cargo-lanes");
    std::fs::create_dir_all(retained_target_directory.join("debug"))
        .expect("the retained target fixture should be creatable");
    std::fs::create_dir_all(&bounded_lane_root).expect("the bounded lane root should be creatable");
    std::fs::write(
        retained_target_directory.join("debug/artifact"),
        "retained-artifact\n",
    )
    .expect("the retained artifact fixture should be writable");
    let baseline_allocated_kibibytes = du_kibibytes(&bounded_storage_root);

    let record_path = sandbox.root_directory.join("bounded-target-record");
    let fixture_path =
        sandbox.write_executable("capture-fixture.sh", &capture_fixture_body());
    for _ in 0..2 {
        let mut command_builder = sanitized_runner_command(&sandbox);
        command_builder
            .env("ASTRONOMICAL_CARGO_LANE_ROOT", &bounded_lane_root)
            .arg("--lane")
            .arg("bounded-storage")
            .arg("--")
            .arg(&fixture_path)
            .arg(&record_path);
        let journey_output = command_builder
            .output()
            .expect("the bounded-storage journey should run");
        assert!(
            journey_output.status.success(),
            "bounded-storage journey log:\n{}",
            journey_log_text(&journey_output)
        );
    }
    let retained_allocated_kibibytes = du_kibibytes(&bounded_storage_root);
    assert!(
        retained_allocated_kibibytes <= baseline_allocated_kibibytes * 2,
        "retained artifacts exceeded twice the clean steady-state baseline: \
         {retained_allocated_kibibytes} > {} kibibytes",
        baseline_allocated_kibibytes * 2
    );
    assert_lane_root_is_empty(&bounded_lane_root);
}

#[test]
fn should_reuse_owner_target_for_nested_lane() {
    let sandbox = DisposableSandbox::install();
    let record_path = sandbox.root_directory.join("nested-target-record");
    let capture_fixture_path =
        sandbox.write_executable("capture-fixture.sh", &capture_fixture_body());
    let nested_fixture_path = sandbox.write_executable(
        "nested-fixture.sh",
        &format!(
            "#!/bin/sh\n\
             set -eu\n\
             printf '%s\\n' \"${{CARGO_TARGET_DIR:?}}\" > \"$1\"\n\
             \"${{XTASK_BINARY:?}}\" disposable-target --lane nested-acceptance -- \\\n\
                 \"{capture_fixture}\" \"{nested_record}\"\n",
            capture_fixture = capture_fixture_path.display(),
            nested_record = sandbox
                .root_directory
                .join("nested-target-record.nested")
                .display()
        ),
    );
    let mut command_builder = sanitized_runner_command(&sandbox);
    command_builder
        .env("XTASK_BINARY", env!("CARGO_BIN_EXE_xtask"))
        .arg("--lane")
        .arg("outer-acceptance")
        .arg("--")
        .arg(&nested_fixture_path)
        .arg(&record_path);
    let journey_output = command_builder
        .output()
        .expect("the nested-lane journey should run");
    let log_text = journey_log_text(&journey_output);

    assert!(
        journey_output.status.success(),
        "nested-lane journey log:\n{log_text}"
    );
    assert!(
        log_text.contains("status=reused"),
        "the nested lane should report target reuse: {log_text}"
    );
    let outer_target = file_text(&record_path).trim().to_owned();
    let nested_target = file_text(&sandbox.root_directory.join("nested-target-record.nested"))
        .trim()
        .to_owned();
    assert_eq!(
        outer_target, nested_target,
        "the nested Cargo lane created a second target owner"
    );
    assert_path_is_absent(Path::new(&outer_target));
    assert_lane_root_is_empty(&sandbox.lane_root_directory);
}

#[test]
fn should_preserve_child_failure_status_and_clean() {
    let sandbox = DisposableSandbox::install();
    let record_path = sandbox.root_directory.join("failure-target-record");
    let fixture_path = sandbox.write_executable(
        "failing-fixture.sh",
        &format!(
            "#!/bin/sh\n\
             set -eu\n\
             printf '%s\\n' \"${{CARGO_TARGET_DIR:?}}\" > \"{record}\"\n\
             mkdir -p \"${{CARGO_TARGET_DIR}}/debug\"\n\
             exit 23\n",
            record = record_path.display()
        ),
    );
    let (journey_output, log_text) = run_journey(&sandbox, "failed-acceptance", &fixture_path);
    assert_eq!(
        exit_status_code(&journey_output),
        23,
        "the lifecycle must preserve the child failure status: {log_text}"
    );
    let recorded_target = file_text(&record_path).trim().to_owned();
    assert_path_is_absent(Path::new(&recorded_target));
    assert_lane_root_is_empty(&sandbox.lane_root_directory);
}

#[test]
fn should_stop_background_descendants_after_successful_command() {
    let sandbox = DisposableSandbox::install();
    let record_path = sandbox.root_directory.join("background-target-record");
    let fixture_path = sandbox.write_executable(
        "background-fixture.sh",
        &format!(
            "#!/bin/sh\n\
             set -eu\n\
             printf '%s\\n' \"${{CARGO_TARGET_DIR:?}}\" > \"{record}\"\n\
             mkdir -p \"${{CARGO_TARGET_DIR}}/debug\"\n\
             sleep 30 &\n\
             descendant_process_id=$!\n\
             printf '%s\\n' \"$descendant_process_id\" > \"{record}.descendant-process\"\n",
            record = record_path.display()
        ),
    );
    let (journey_output, log_text) = run_journey(&sandbox, "background-descendant", &fixture_path);
    assert!(
        journey_output.status.success(),
        "the background-descendant journey should succeed: {log_text}"
    );
    let descendant_process_id: i32 = file_text(
        &sandbox
            .root_directory
            .join("background-target-record.descendant-process"),
    )
    .trim()
    .parse()
    .expect("the descendant process id should be numeric");
    assert!(
        !process_is_alive(descendant_process_id),
        "the successful journey left descendant process {descendant_process_id} running"
    );
    let recorded_target = file_text(&record_path).trim().to_owned();
    assert_path_is_absent(Path::new(&recorded_target));
    assert_lane_root_is_empty(&sandbox.lane_root_directory);
}

#[test]
fn should_clean_owned_target_and_stop_descendants_when_interrupted() {
    let sandbox = DisposableSandbox::install();
    let record_path = sandbox.root_directory.join("interruption-target-record");
    let fixture_path = sandbox.write_executable(
        "interrupting-fixture.sh",
        &format!(
            "#!/bin/sh\n\
             set -eu\n\
             printf '%s\\n' \"${{CARGO_TARGET_DIR:?}}\" > \"{record}\"\n\
             mkdir -p \"${{CARGO_TARGET_DIR}}/debug\"\n\
             trap '' TERM\n\
             sleep 30 &\n\
             descendant_process_id=$!\n\
             printf '%s\\n' \"$descendant_process_id\" > \"{record}.descendant-process\"\n\
             wait \"$descendant_process_id\"\n",
            record = record_path.display()
        ),
    );
    let mut command_builder = sanitized_runner_command(&sandbox);
    command_builder
        .arg("--lane")
        .arg("interrupted-acceptance")
        .arg("--")
        .arg(&fixture_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let journey_process = command_builder
        .spawn()
        .expect("the interrupted journey should spawn");
    wait_for_file(
        &sandbox
            .root_directory
            .join("interruption-target-record.descendant-process"),
    );
    send_terminate(journey_process.id());
    arm_termination_watchdog(journey_process.id());
    let journey_output = journey_process
        .wait_with_output()
        .expect("the interrupted journey should finish");
    let log_text = journey_log_text(&journey_output);
    assert_eq!(
        exit_status_code(&journey_output),
        143,
        "an interrupted lifecycle must exit 143: {log_text}"
    );
    assert!(
        log_text.contains("status=interrupted signal=TERM"),
        "the interrupted journey should report its signal: {log_text}"
    );
    let descendant_process_id: i32 = file_text(
        &sandbox
            .root_directory
            .join("interruption-target-record.descendant-process"),
    )
    .trim()
    .parse()
    .expect("the descendant process id should be numeric");
    assert!(
        !process_is_alive(descendant_process_id),
        "the interrupted journey left descendant process {descendant_process_id} running"
    );
    let recorded_target = file_text(&record_path).trim().to_owned();
    assert_path_is_absent(Path::new(&recorded_target));
    assert_lane_root_is_empty(&sandbox.lane_root_directory);
}

#[test]
fn should_refuse_nested_reuse_without_target_ownership() {
    let sandbox = DisposableSandbox::install();
    let fixture_path = sandbox.write_executable("capture-fixture.sh", &capture_fixture_body());
    let mut command_builder = sanitized_runner_command(&sandbox);
    command_builder
        // A disposable lifecycle marker without its owning CARGO_TARGET_DIR
        // must be refused instead of silently creating a second owner.
        .env("ASTRONOMICAL_CARGO_TARGET_LIFECYCLE", "disposable")
        .env_remove("CARGO_TARGET_DIR")
        .arg("--lane")
        .arg("orphaned-acceptance")
        .arg("--")
        .arg(&fixture_path);
    let journey_output = command_builder
        .output()
        .expect("the orphaned nested journey should run");
    let log_text = journey_log_text(&journey_output);
    assert_eq!(
        exit_status_code(&journey_output),
        1,
        "an orphaned nested lifecycle must exit 1: {log_text}"
    );
    assert!(
        log_text.contains("nested disposable lifecycle is missing its target ownership"),
        "the orphaned nested lifecycle should name its refusal: {log_text}"
    );
    assert_lane_root_is_empty(&sandbox.lane_root_directory);
}

#[test]
fn should_refuse_cleanup_through_an_unowned_symlink() {
    let sandbox = DisposableSandbox::install();
    let unowned_directory = sandbox.root_directory.join("unowned");
    std::fs::create_dir_all(&unowned_directory).expect("the unowned directory should be creatable");
    std::fs::write(unowned_directory.join("evidence"), "preserve\n")
        .expect("the unowned evidence fixture should be writable");
    let record_path = sandbox.root_directory.join("symlink-target-record");
    let fixture_path = sandbox.write_executable(
        "symlink-fixture.sh",
        &format!(
            "#!/bin/sh\n\
             set -eu\n\
             printf '%s\\n' \"${{CARGO_TARGET_DIR:?}}\" > \"{record}\"\n\
             rm -rf \"${{CARGO_TARGET_DIR}}\"\n\
             ln -s \"{unowned_directory}\" \"${{CARGO_TARGET_DIR}}\"\n",
            record = record_path.display(),
            unowned_directory = unowned_directory.display()
        ),
    );
    let (journey_output, log_text) = run_journey(&sandbox, "unsafe-cleanup", &fixture_path);
    assert_eq!(
        exit_status_code(&journey_output),
        1,
        "the lifecycle accepted an unowned replacement path: {log_text}"
    );
    assert!(
        log_text.contains("status=cleanup-refused"),
        "the refused cleanup should be reported: {log_text}"
    );
    assert_eq!(
        file_text(&unowned_directory.join("evidence")),
        "preserve\n",
        "the lifecycle removed content through an unowned symlink"
    );
    let replaced_target = file_text(&record_path).trim().to_owned();
    assert!(
        Path::new(&replaced_target).is_symlink(),
        "the symlink refusal fixture did not replace the owned target"
    );
    std::fs::remove_file(&replaced_target)
        .expect("the replacement symlink should be removable");
    assert_lane_root_is_empty(&sandbox.lane_root_directory);
}

#[test]
fn should_reject_an_invalid_lane_name_without_touching_the_lane_root() {
    let sandbox = DisposableSandbox::install();
    let fixture_path = sandbox.write_executable("capture-fixture.sh", &capture_fixture_body());
    let (journey_output, log_text) = run_journey(&sandbox, "../escape", &fixture_path);
    assert_eq!(
        exit_status_code(&journey_output),
        2,
        "an invalid lane name must exit 2: {log_text}"
    );
    assert_lane_root_is_empty(&sandbox.lane_root_directory);
}

#[test]
fn should_retain_one_owned_target_when_measurement_is_interrupted() {
    let sandbox = DisposableSandbox::install();
    let shim_directory = sandbox.root_directory.join("shim-bin");
    std::fs::create_dir_all(&shim_directory).expect("the shim directory should be creatable");
    let cleanup_ready_path = sandbox.root_directory.join("cleanup-ready");
    let du_shim_path = {
        let du_shim_path = shim_directory.join("du");
        std::fs::write(
            &du_shim_path,
            &format!(
                "#!/bin/sh\n\
                 set -eu\n\
                 printf '4\\t%s\\n' \"$2\"\n\
                 printf '%s\\n' ready > \"{cleanup_ready}\"\n\
                 exec sleep 30\n",
                cleanup_ready = cleanup_ready_path.display()
            ),
        )
        .expect("the du shim should be writable");
        make_executable(&du_shim_path);
        du_shim_path
    };
    let _ = du_shim_path;
    let record_path = sandbox.root_directory.join("signal-target-record");
    let fixture_path =
        sandbox.write_executable("capture-fixture.sh", &capture_fixture_body());
    let mut command_builder = sanitized_runner_command(&sandbox);
    command_builder
        .arg("--lane")
        .arg("cleanup-signal")
        .arg("--")
        .arg(&fixture_path)
        .arg(&record_path)
        .env(
            "PATH",
            format!(
                "{}:{}",
                shim_directory.display(),
                std::env::var("PATH").unwrap_or_default()
            ),
        )
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let journey_process = command_builder
        .spawn()
        .expect("the cleanup-signal journey should spawn");
    wait_for_file(&cleanup_ready_path);
    send_terminate(journey_process.id());
    arm_termination_watchdog(journey_process.id());
    let journey_output = journey_process
        .wait_with_output()
        .expect("the cleanup-signal journey should finish");
    let log_text = journey_log_text(&journey_output);
    assert_eq!(
        exit_status_code(&journey_output),
        143,
        "the interrupted measurement returned an unexpected status: {log_text}"
    );
    assert!(
        log_text.contains("status=cleanup-skipped-retained"),
        "the interrupted measurement did not report retained cleanup ownership: {log_text}"
    );
    let original_target = file_text(&record_path).trim().to_owned();
    let original_target_name = Path::new(&original_target)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let retained_target_directories: Vec<PathBuf> = std::fs::read_dir(&sandbox.lane_root_directory)
        .expect("the lane root should be listable after retention")
        .map(|lane_entry| {
            lane_entry
                .expect("the lane entry should be readable")
                .path()
        })
        .filter(|lane_entry_path| {
            lane_entry_path
                .file_name()
                .map(|entry_name| {
                    entry_name
                        .to_string_lossy()
                        .starts_with(&format!("{original_target_name}.removing."))
                })
                .unwrap_or(false)
        })
        .collect();
    assert_eq!(
        retained_target_directories.len(),
        1,
        "interruption during measurement must retain exactly one handoff target: {log_text}"
    );
    let retained_target_directory = &retained_target_directories[0];
    assert!(
        retained_target_directory.join(LIFECYCLE_MARKER_NAME).is_file(),
        "the retained target must keep its lifecycle marker"
    );
    let _ = std::fs::remove_dir_all(retained_target_directory);
    assert_lane_root_is_empty(&sandbox.lane_root_directory);
}
