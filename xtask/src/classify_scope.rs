//! `cargo xtask classify-scope` — owns the event-specific CI authority
//! policy so pull-request verification and default-branch cache publication
//! cannot drift across embedded workflow code. The verdict logic is pure and
//! unit-tested beside this source; the git and environment edges stay thin
//! (issue #1250).

use std::path::Path;
use std::process::Command;

const ZERO_GIT_SHA: &str = "0000000000000000000000000000000000000000";

/// Pathspecs mirroring native source-identity ownership plus the Rust
/// toolchain (it participates in full hosted compatibility). Kept in sync
/// with the identity policy in `astronomical-native-identity`.
const NATIVE_INPUT_PATHSPECS: &[&str] = &[
    "crates/runtime-integration/build.rs",
    "crates/mlx-c-rust/build.rs",
    "crates/runtime-integration/build_native_linking.rs",
    "crates/runtime-integration/build_native_store.rs",
    "crates/runtime-integration/build_native_store_manifest.rs",
    "crates/runtime-integration/build_native_compile.rs",
    "crates/runtime-integration/native-build-store-schema-version",
    "crates/runtime-integration/native",
    "crates/native-identity/src",
    "third-party/native-dependency-manifest.cmake",
    "third-party/pins",
    "third-party/patches",
    "rust-toolchain.toml",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CiEvent {
    PullRequest,
    Push,
    WorkflowDispatch,
}

impl CiEvent {
    fn from_event_name(event_name: &str) -> Result<Self, String> {
        match event_name {
            "pull_request" => Ok(Self::PullRequest),
            "push" => Ok(Self::Push),
            "workflow_dispatch" => Ok(Self::WorkflowDispatch),
            other => Err(format!("unsupported CI event: {other}")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangeScopeVerdict {
    pub code_changed: bool,
    pub native_inputs_changed: bool,
    pub macos_verification_required: bool,
    pub classification: String,
    pub changed_file_count: usize,
}

/// Classifies one page/site artifact change as static-only. POSIX `case`
/// pattern semantics: `*` spans `/`, so `site/*` covers nested paths.
pub fn is_static_only_path(changed_path: &str) -> bool {
    changed_path.starts_with("site/")
        || changed_path.ends_with(".md")
        || changed_path.starts_with(".github/assets/")
        || changed_path == ".github/workflows/pages.yml"
}

/// The pure verdict: identical inputs give identical outputs, no environment
/// access. PRs and pushes (PR merges) both gate on code changes so that
/// default-branch pushes save native-build and sccache caches for future PRs
/// to restore; static-only changes still skip.
pub fn classify_change_scope(
    event: CiEvent,
    changed_paths: &[String],
    native_inputs_changed: bool,
) -> ChangeScopeVerdict {
    match event {
        CiEvent::WorkflowDispatch => ChangeScopeVerdict {
            code_changed: true,
            native_inputs_changed: false,
            macos_verification_required: true,
            classification: "manual_run".to_owned(),
            changed_file_count: 0,
        },
        CiEvent::PullRequest | CiEvent::Push => {
            let code_changed = changed_paths
                .iter()
                .any(|changed_path| !is_static_only_path(changed_path));
            let macos_verification_required = code_changed;
            let classification = if !code_changed {
                "static_only"
            } else if native_inputs_changed {
                "native_inputs"
            } else {
                "code_or_build"
            };
            ChangeScopeVerdict {
                code_changed,
                native_inputs_changed,
                macos_verification_required,
                classification: classification.to_owned(),
                changed_file_count: changed_paths.len(),
            }
        }
    }
}

/// Fail-safe verdict for an unresolvable change range (fresh history, forced
/// pushes): verify everything, publish caches, and label the run so the gap
/// is visible in the classification log. Every flag is forced true — the
/// range is unknown, so no gate may stay closed by accident.
pub(crate) fn fail_safe_verdict() -> ChangeScopeVerdict {
    ChangeScopeVerdict {
        code_changed: true,
        native_inputs_changed: true,
        macos_verification_required: true,
        classification: "unknown_history".to_owned(),
        changed_file_count: 0,
    }
}

pub fn run(repository_root: &Path) -> Result<(), String> {
    let started_at = std::time::Instant::now();
    let event_name = require_environment_value("EVENT_NAME")?;
    let github_output_path = require_environment_value("GITHUB_OUTPUT")?;
    let event = CiEvent::from_event_name(&event_name)?;

    eprintln!("[ci-change-scope] status=start event={event_name}");

    let verdict = match event {
        CiEvent::WorkflowDispatch => classify_change_scope(event, &[], false),
        CiEvent::PullRequest | CiEvent::Push => {
            let (base_sha, head_sha) = match event {
                CiEvent::PullRequest => (
                    optional_environment_value("PULL_REQUEST_BASE_SHA"),
                    optional_environment_value("PULL_REQUEST_HEAD_SHA"),
                ),
                _ => (
                    optional_environment_value("PUSH_BEFORE_SHA"),
                    optional_environment_value("CURRENT_SHA"),
                ),
            };
            match resolve_verdict_from_history(repository_root, base_sha, head_sha) {
                Some(history_verdict) => history_verdict,
                None => fail_safe_verdict(),
            }
        }
    };

    publish_verdict(&github_output_path, &event_name, &verdict, started_at);
    Ok(())
}

type HistoryVerdict = Option<ChangeScopeVerdict>;

fn resolve_verdict_from_history(
    repository_root: &Path,
    base_sha: Option<String>,
    head_sha: Option<String>,
) -> HistoryVerdict {
    let base_sha = base_sha?;
    let head_sha = head_sha?;
    if base_sha == ZERO_GIT_SHA {
        return None;
    }
    if !commit_exists(repository_root, &base_sha) || !commit_exists(repository_root, &head_sha) {
        return None;
    }
    let changed_paths = changed_paths_between(repository_root, &base_sha, &head_sha)?;
    let native_inputs_changed =
        native_inputs_changed_between(repository_root, &base_sha, &head_sha)?;
    Some(classify_change_scope(
        CiEvent::PullRequest,
        &changed_paths,
        native_inputs_changed,
    ))
}

fn commit_exists(repository_root: &Path, commit_sha: &str) -> bool {
    Command::new("git")
        .arg("-C")
        .arg(repository_root)
        .args(["cat-file", "-e", &format!("{commit_sha}^{{commit}}")])
        .output()
        .map(|git_output| git_output.status.success())
        .unwrap_or(false)
}

fn changed_paths_between(
    repository_root: &Path,
    base_sha: &str,
    head_sha: &str,
) -> Option<Vec<String>> {
    let git_output = Command::new("git")
        .arg("-C")
        .arg(repository_root)
        .args(["diff", "--name-only", "--no-renames", base_sha, head_sha])
        .output()
        .ok()?;
    if !git_output.status.success() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&git_output.stdout)
            .lines()
            .filter(|line| !line.is_empty())
            .map(str::to_owned)
            .collect(),
    )
}

fn native_inputs_changed_between(
    repository_root: &Path,
    base_sha: &str,
    head_sha: &str,
) -> Option<bool> {
    let git_status = Command::new("git")
        .arg("-C")
        .arg(repository_root)
        .args(["diff", "--quiet", "--no-renames", base_sha, head_sha, "--"])
        .args(NATIVE_INPUT_PATHSPECS)
        .status()
        .ok()?;
    let exit_code = git_status.code();
    match exit_code {
        Some(0) => Some(false),
        Some(1) => Some(true),
        _ => None,
    }
}

fn publish_verdict(
    github_output_path: &str,
    event_name: &str,
    verdict: &ChangeScopeVerdict,
    started_at: std::time::Instant,
) {
    let mut github_output = std::fs::OpenOptions::new()
        .append(true)
        .open(github_output_path)
        .expect("GITHUB_OUTPUT should be appendable");
    use std::io::Write;
    writeln!(github_output, "code_changed={}", verdict.code_changed).unwrap();
    writeln!(
        github_output,
        "native_inputs_changed={}",
        verdict.native_inputs_changed
    )
    .unwrap();
    writeln!(
        github_output,
        "macos_verification_required={}",
        verdict.macos_verification_required
    )
    .unwrap();
    eprintln!(
        "[ci-change-scope] status=complete event={event_name} classification={} changed_files={} code_changed={} native_inputs_changed={} macos_verification_required={} elapsed_seconds={}",
        verdict.classification,
        verdict.changed_file_count,
        verdict.code_changed,
        verdict.native_inputs_changed,
        verdict.macos_verification_required,
        started_at.elapsed().as_secs()
    );
}

fn require_environment_value(variable_name: &str) -> Result<String, String> {
    optional_environment_value(variable_name).ok_or_else(|| format!("{variable_name} is required"))
}

fn optional_environment_value(variable_name: &str) -> Option<String> {
    std::env::var(variable_name)
        .ok()
        .filter(|value| !value.is_empty())
}
