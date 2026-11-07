//! The cached xtask binary, as typed Rust: parses the `provision-xtask` action and
//! `.github/workflows/ci.yml`, and pins what keeps a stale or missing binary out of CI.
//! A stale binary is the failure that matters, because CI would stay green while running
//! tooling older than the commit under test.

mod contract_support;

use contract_support::{
    provision_xtask_action_contract, repository_root, required_field, required_text,
    workflow_contract,
};
use serde_yaml::Value;
use serial_test::serial;
use std::collections::BTreeSet;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;

const XTASK_FOLDER: &str = "xtask";
const PROVISION_ACTION_REFERENCE: &str = "./.github/actions/provision-xtask";
const CHECKOUT_ACTION_PREFIX: &str = "actions/checkout@";

fn manifest_text_of(package_folder: &str) -> String {
    std::fs::read_to_string(repository_root().join(package_folder).join("Cargo.toml"))
        .expect("the package manifest should be readable")
}

fn path_value_of(dependency_line: &str) -> Option<&str> {
    let after_path_key = dependency_line.split("path = \"").nth(1)?;
    after_path_key.split('"').next()
}

fn dependency_name_of(dependency_line: &str) -> &str {
    let name_end = dependency_line
        .find(['=', '.', ' '])
        .unwrap_or(dependency_line.len());
    &dependency_line[..name_end]
}

/// Sections whose crates link into the binary; dev-dependencies do not.
fn is_linked_dependency_section(section_header: &str) -> bool {
    section_header == "[dependencies]" || section_header == "[build-dependencies]"
}

/// Path dependencies a manifest links. A direct entry is relative to that manifest; an
/// entry inherited from the workspace table is relative to the repository root, marked
/// by a leading `/` so the walk can resolve each from the right folder.
fn linked_path_dependencies(manifest_text: &str, workspace_manifest_text: &str) -> Vec<String> {
    let mut relative_paths = Vec::new();
    let mut is_in_linked_section = false;
    for manifest_line in manifest_text.lines() {
        if manifest_line.starts_with('[') {
            is_in_linked_section = is_linked_dependency_section(manifest_line.trim());
            continue;
        }
        if !is_in_linked_section {
            continue;
        }
        if let Some(direct_path) = path_value_of(manifest_line) {
            relative_paths.push(direct_path.to_owned());
        } else if let Some(inherited_path) =
            workspace_dependency_path(workspace_manifest_text, manifest_line)
        {
            relative_paths.push(inherited_path);
        }
    }
    relative_paths
}

fn workspace_dependency_path(
    workspace_manifest_text: &str,
    dependency_line: &str,
) -> Option<String> {
    if !dependency_line.contains("workspace = true") {
        return None;
    }
    let dependency_name = dependency_name_of(dependency_line);
    let mut is_in_workspace_table = false;
    for workspace_line in workspace_manifest_text.lines() {
        if workspace_line.starts_with('[') {
            is_in_workspace_table = workspace_line.trim() == "[workspace.dependencies]";
            continue;
        }
        if is_in_workspace_table && dependency_name_of(workspace_line) == dependency_name {
            return path_value_of(workspace_line)
                .map(str::to_owned)
                .map(mark_as_repository_relative);
        }
    }
    None
}

fn mark_as_repository_relative(workspace_path: String) -> String {
    format!("/{workspace_path}")
}

fn repository_relative_folder(package_folder: &str, dependency_path: &str) -> String {
    let root = repository_root();
    let base_folder = match dependency_path.strip_prefix('/') {
        Some(_) => root.clone(),
        None => root.join(package_folder),
    };
    let resolved_folder = base_folder
        .join(dependency_path.trim_start_matches('/'))
        .canonicalize()
        .expect("a path dependency should resolve");
    let folder_inside_repository: &Path = resolved_folder
        .strip_prefix(&root)
        .expect("a path dependency should sit inside the repository");
    folder_inside_repository.to_string_lossy().into_owned()
}

/// The folder of every package the xtask binary links from this repository.
fn folders_linked_into_xtask() -> BTreeSet<String> {
    let workspace_manifest_text = std::fs::read_to_string(repository_root().join("Cargo.toml"))
        .expect("the workspace manifest should be readable");
    let mut linked_folders = BTreeSet::new();
    let mut folders_to_visit = vec![XTASK_FOLDER.to_owned()];
    while let Some(package_folder) = folders_to_visit.pop() {
        if !linked_folders.insert(package_folder.clone()) {
            continue;
        }
        let manifest_text = manifest_text_of(&package_folder);
        for relative_path in linked_path_dependencies(&manifest_text, &workspace_manifest_text) {
            folders_to_visit.push(repository_relative_folder(&package_folder, &relative_path));
        }
    }
    linked_folders
}

fn folders_missing_from_key(key_text: &str, folders: &BTreeSet<String>) -> Vec<String> {
    let mut missing_folders = Vec::new();
    for package_folder in folders {
        if !key_text.contains(&format!("'{package_folder}/**'")) {
            missing_folders.push(package_folder.clone());
        }
    }
    missing_folders
}

fn cache_step_with_operation<'a>(action_steps: &[&'a Value], operation: &str) -> &'a Value {
    for action_step in action_steps {
        let Some(step_inputs) = action_step.get("with") else {
            continue;
        };
        if step_inputs.get("operation").and_then(Value::as_str) == Some(operation) {
            return action_step;
        }
    }
    panic!("the action has no cache step with operation {operation}");
}

fn cache_key_of(cache_step: &Value) -> &str {
    required_text(
        required_field(cache_step, "with", "the cache step"),
        "key",
        "the cache step inputs",
    )
}

fn action_steps(action: &Value) -> Vec<&Value> {
    required_field(
        required_field(action, "runs", "the action"),
        "steps",
        "the action runs",
    )
    .as_sequence()
    .expect("the action steps must be a sequence")
    .iter()
    .collect()
}

/// True for a line that runs the binary from PATH rather than through cargo.
fn runs_xtask_directly(run_text: &str) -> bool {
    for run_line in run_text.lines() {
        let trimmed_line = run_line.trim_start();
        if trimmed_line.starts_with("xtask ") {
            return true;
        }
    }
    false
}

fn step_uses(job_step: &Value, action_prefix: &str) -> bool {
    match job_step.get("uses").and_then(Value::as_str) {
        Some(uses_text) => uses_text.starts_with(action_prefix),
        None => false,
    }
}

#[test]
fn should_key_the_binary_on_every_folder_it_links() {
    let action = provision_xtask_action_contract();
    let steps = action_steps(&action);
    let restore_key = cache_key_of(cache_step_with_operation(&steps, "restore"));
    let linked_folders = folders_linked_into_xtask();
    assert!(
        linked_folders.contains("crates/native-identity"),
        "the walk should follow xtask's path dependencies"
    );
    assert_eq!(
        folders_missing_from_key(restore_key, &linked_folders),
        Vec::<String>::new(),
        "a change in a linked folder must move the key, or CI runs a stale binary"
    );
    for build_input in [
        "Cargo.lock",
        "Cargo.toml",
        "rust-toolchain.toml",
        ".cargo/config.toml",
    ] {
        assert!(
            restore_key.contains(&format!("'{build_input}'")),
            "{build_input} changes what the binary is built from"
        );
    }
}

#[test]
fn should_report_a_linked_folder_the_key_leaves_out() {
    let linked_folders = BTreeSet::from(["xtask".to_owned(), "crates/new-helper".to_owned()]);
    assert_eq!(
        folders_missing_from_key("hashFiles('xtask/**')", &linked_folders),
        vec!["crates/new-helper".to_owned()]
    );
}

#[test]
fn should_not_count_a_dev_dependency_as_linked() {
    let manifest_text = "[dependencies]\nlinked = { path = \"../linked\" }\n\n\
                         [dev-dependencies]\ntest-only = { path = \"../test-only\" }\n";
    assert_eq!(
        linked_path_dependencies(manifest_text, ""),
        vec!["../linked".to_owned()]
    );
}

#[test]
fn should_resolve_a_path_dependency_inherited_from_the_workspace() {
    let manifest_text = "[dependencies]\nshared.workspace = true\nserde.workspace = true\n";
    let workspace_text =
        "[workspace.dependencies]\nshared = { path = \"crates/shared\" }\nserde = \"1\"\n";
    assert_eq!(
        linked_path_dependencies(manifest_text, workspace_text),
        vec!["/crates/shared".to_owned()]
    );
}

#[test]
fn should_restore_and_save_the_binary_under_one_key() {
    let action = provision_xtask_action_contract();
    let steps = action_steps(&action);
    assert_eq!(
        cache_key_of(cache_step_with_operation(&steps, "restore")),
        cache_key_of(cache_step_with_operation(&steps, "save")),
        "a save under another key would never be restored"
    );
}

/// A pull request's binary is built from unreviewed sources; saving it could hand it
/// to another run with the same key.
#[test]
fn should_save_the_binary_only_when_built_on_the_default_branch() {
    let action = provision_xtask_action_contract();
    let steps = action_steps(&action);
    let save_condition = required_text(
        cache_step_with_operation(&steps, "save"),
        "if",
        "the save step",
    );
    assert!(save_condition.contains("github.ref_name == github.event.repository.default_branch"));
    assert!(save_condition.contains("was_built == 'true'"));
}

/// A step reaching for `xtask` before the action ran would fail with "command not found"
/// only on the path that reaches it, and a local action needs the checkout to exist.
#[test]
fn should_provision_after_checkout_and_before_every_direct_xtask_command() {
    let workflow = workflow_contract();
    let jobs = required_field(&workflow, "jobs", "the workflow")
        .as_mapping()
        .expect("the jobs must be a mapping");
    let mut provisioning_job_count = 0;
    for (job_key, job) in jobs {
        let job_id = job_key.as_str().expect("a job id is text");
        let Some(job_steps) = job.get("steps").and_then(Value::as_sequence) else {
            continue;
        };
        let mut has_checkout = false;
        let mut has_provisioned = false;
        for job_step in job_steps {
            let run_text = job_step.get("run").and_then(Value::as_str).unwrap_or("");
            if runs_xtask_directly(run_text) {
                assert!(
                    has_provisioned,
                    "job {job_id} runs xtask before provisioning"
                );
            }
            if step_uses(job_step, CHECKOUT_ACTION_PREFIX) {
                has_checkout = true;
            }
            if step_uses(job_step, PROVISION_ACTION_REFERENCE) {
                assert!(has_checkout, "job {job_id} provisions before its checkout");
                has_provisioned = true;
                provisioning_job_count += 1;
            }
        }
    }
    assert_eq!(
        provisioning_job_count, 2,
        "the classifier and the macOS build provision the binary"
    );
}

#[test]
fn should_tell_a_direct_xtask_command_from_one_through_cargo() {
    assert!(runs_xtask_directly("set -e\n  xtask classify-scope\n"));
    assert!(!runs_xtask_directly("cargo xtask check-architecture\n"));
    assert!(!runs_xtask_directly("# xtask prune-caches runs later\n"));
    assert!(!runs_xtask_directly(""));
}

const BUILD_STEP_ID: &str = "xtask-binary-build";
const RESTORED_BINARY_RELATIVE_PATH: &str = ".astronomical-ci/xtask-binary/xtask";
/// The real binary's usage exit, which the probe's `--help` always gets.
const USAGE_EXIT_CODE: i32 = 2;
const CRASH_EXIT_CODE: i32 = 139;
const CARGO_SUCCESS: i32 = 0;
const CARGO_FAILURE: i32 = 101;
/// Marks the binary the stub cargo built, to tell it from one that was restored.
const STUB_BUILT_MARKER: &str = "stub-built";

enum RestoredBinary {
    Missing,
    NotExecutable,
    Exits(i32),
}

struct ProvisionStepOutcome {
    exit_code: Option<i32>,
    step_outputs: String,
    recorded_path: String,
    installed_binary_text: Option<String>,
    is_installed_binary_executable: bool,
}

fn write_executable_script(script_path: &Path, script_body: &str) {
    std::fs::write(script_path, format!("#!/bin/bash\n{script_body}\n")).expect("script writes");
    std::fs::set_permissions(script_path, std::fs::Permissions::from_mode(0o755))
        .expect("script becomes executable");
}

fn provision_step_script() -> String {
    let action = provision_xtask_action_contract();
    for action_step in action_steps(&action) {
        if action_step.get("id").and_then(Value::as_str) == Some(BUILD_STEP_ID) {
            return required_text(action_step, "run", "the build step").to_owned();
        }
    }
    panic!("the action has no step with id {BUILD_STEP_ID}");
}

fn place_restored_binary(installed_binary: &Path, restored_binary: &RestoredBinary) {
    if matches!(restored_binary, RestoredBinary::Missing) {
        return;
    }
    std::fs::create_dir_all(installed_binary.parent().expect("the binary has a folder"))
        .expect("the binary folder should exist");
    match restored_binary {
        RestoredBinary::Missing => {}
        RestoredBinary::NotExecutable => {
            std::fs::write(installed_binary, "not a program\n").expect("the file writes");
        }
        RestoredBinary::Exits(exit_code) => {
            write_executable_script(installed_binary, &format!("exit {exit_code}"));
        }
    }
}

fn write_stub_cargo(stub_folder: &Path, cargo_exit_code: i32) {
    write_executable_script(
        &stub_folder.join("cargo"),
        &format!(
            "[ {cargo_exit_code} -eq 0 ] || exit {cargo_exit_code}\n\
             mkdir -p target/debug\n\
             printf '#!/bin/bash\\n# {STUB_BUILT_MARKER}\\nexit 2\\n' > target/debug/xtask\n\
             chmod +x target/debug/xtask"
        ),
    );
}

/// Runs the action's own script under the exact shell GitHub uses for a `shell: bash`
/// step. Text contracts cannot see this: `-e` survives `set -uo pipefail`, so a probe
/// exiting 2 once ended the script and failed every run that restored the binary.
/// Callers run serially: each writes a script and executes it straight away, and a fork
/// in a parallel test can briefly hold it open for writing, which Linux reports as
/// "Text file busy" and would flake these tests.
fn run_provision_step(
    restored_binary: &RestoredBinary,
    cargo_exit_code: i32,
) -> ProvisionStepOutcome {
    let sandbox = tempfile::tempdir().expect("the sandbox should exist");
    let home_folder = sandbox.path().join("home");
    let work_folder = sandbox.path().join("work");
    let stub_folder = sandbox.path().join("stub");
    for sandbox_folder in [&home_folder, &work_folder, &stub_folder] {
        std::fs::create_dir_all(sandbox_folder).expect("the folder should exist");
    }
    let installed_binary = home_folder.join(RESTORED_BINARY_RELATIVE_PATH);
    place_restored_binary(&installed_binary, restored_binary);
    write_stub_cargo(&stub_folder, cargo_exit_code);
    let step_script = sandbox.path().join("step.sh");
    std::fs::write(&step_script, provision_step_script()).expect("the step script writes");
    let outputs_file = sandbox.path().join("outputs");
    let path_file = sandbox.path().join("path");
    std::fs::write(&outputs_file, "").expect("the outputs file writes");
    std::fs::write(&path_file, "").expect("the path file writes");
    let inherited_path = std::env::var("PATH").expect("the test has a PATH");
    let step_status = Command::new("bash")
        .args(["--noprofile", "--norc", "-e", "-o", "pipefail"])
        .arg(&step_script)
        .current_dir(&work_folder)
        // bash reads this even under --norc, so it must not reach the sandboxed shell.
        .env_remove("BASH_ENV")
        .env("HOME", &home_folder)
        .env(
            "PATH",
            format!("{}:{inherited_path}", stub_folder.display()),
        )
        .env("GITHUB_OUTPUT", &outputs_file)
        .env("GITHUB_PATH", &path_file)
        .status()
        .expect("bash should start");
    ProvisionStepOutcome {
        exit_code: step_status.code(),
        step_outputs: std::fs::read_to_string(&outputs_file).expect("the outputs read"),
        recorded_path: std::fs::read_to_string(&path_file).expect("the path file reads"),
        installed_binary_text: std::fs::read_to_string(&installed_binary).ok(),
        is_installed_binary_executable: is_executable_file(&installed_binary),
    }
}

fn is_executable_file(file_path: &Path) -> bool {
    match std::fs::metadata(file_path) {
        Ok(file_metadata) => file_metadata.permissions().mode() & 0o111 != 0,
        Err(_) => false,
    }
}

fn assert_binary_was_built(outcome: &ProvisionStepOutcome) {
    assert_eq!(outcome.exit_code, Some(0));
    assert!(outcome.step_outputs.contains("was_built=true"));
    assert!(outcome.recorded_path.contains("xtask-binary"));
    let installed_text = outcome
        .installed_binary_text
        .as_deref()
        .expect("the binary should be installed");
    assert!(
        installed_text.contains(STUB_BUILT_MARKER),
        "the freshly built binary must replace whatever was restored"
    );
    assert!(
        outcome.is_installed_binary_executable,
        "a rebuilt binary that cannot execute would fail the next xtask command"
    );
}

fn assert_restored_binary_was_kept(outcome: &ProvisionStepOutcome) {
    assert_eq!(
        outcome.exit_code,
        Some(0),
        "this exit must not end the step"
    );
    assert!(outcome.step_outputs.contains("was_built=false"));
    assert!(outcome.recorded_path.contains("xtask-binary"));
    let installed_text = outcome
        .installed_binary_text
        .as_deref()
        .expect("the restored binary should remain");
    assert!(!installed_text.contains(STUB_BUILT_MARKER));
}

#[test]
#[serial(provision_step)]
fn should_accept_a_restored_binary_that_exits_with_its_usage_code() {
    let outcome = run_provision_step(&RestoredBinary::Exits(USAGE_EXIT_CODE), CARGO_FAILURE);
    assert_restored_binary_was_kept(&outcome);
}

#[test]
#[serial(provision_step)]
fn should_accept_a_restored_binary_that_exits_cleanly() {
    let outcome = run_provision_step(&RestoredBinary::Exits(0), CARGO_FAILURE);
    assert_restored_binary_was_kept(&outcome);
}

#[test]
#[serial(provision_step)]
fn should_build_the_binary_when_none_was_restored() {
    let outcome = run_provision_step(&RestoredBinary::Missing, CARGO_SUCCESS);
    assert_binary_was_built(&outcome);
}

#[test]
#[serial(provision_step)]
fn should_rebuild_a_restored_binary_that_crashes() {
    let outcome = run_provision_step(&RestoredBinary::Exits(CRASH_EXIT_CODE), CARGO_SUCCESS);
    assert_binary_was_built(&outcome);
}

#[test]
#[serial(provision_step)]
fn should_rebuild_a_restored_file_that_is_not_executable() {
    let outcome = run_provision_step(&RestoredBinary::NotExecutable, CARGO_SUCCESS);
    assert_binary_was_built(&outcome);
}

#[test]
#[serial(provision_step)]
fn should_fail_the_step_with_cargos_exit_code_when_the_rebuild_fails() {
    let outcome = run_provision_step(&RestoredBinary::Missing, CARGO_FAILURE);
    assert_eq!(
        outcome.exit_code,
        Some(CARGO_FAILURE),
        "CI must not continue without a binary"
    );
    assert_eq!(outcome.step_outputs, "");
    assert_eq!(outcome.recorded_path, "");
    assert_eq!(outcome.installed_binary_text, None);
}
