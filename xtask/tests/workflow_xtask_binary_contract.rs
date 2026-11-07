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
use std::collections::BTreeSet;
use std::path::Path;

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
