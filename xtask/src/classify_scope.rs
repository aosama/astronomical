//! `cargo xtask classify-scope` — owns the event-specific CI authority
//! policy so pull-request verification and default-branch cache publication
//! cannot drift across embedded workflow code. The verdict logic is pure and
//! unit-tested beside this source; the git and environment edges stay thin
//! (issue #1250).

use std::path::Path;
use std::process::Command;

use crate::verified_tree::{is_tree_already_verified, last_default_branch_run_commit};
use crate::version_only_bump::{VersionBumpTexts, is_version_only_bump};

const ZERO_GIT_SHA: &str = "0000000000000000000000000000000000000000";

/// Pathspecs mirroring native source-identity ownership plus the Rust
/// toolchain (it participates in the full hosted toolchain environment). Kept
/// in sync with the identity policy in `astronomical-native-identity`.
const NATIVE_INPUT_PATHSPECS: &[&str] = &[
    "crates/runtime-integration/build.rs",
    // Kept for transition classification: the path matches the bridge
    // deletion itself, so it keeps the crate's former in-repo name
    // (mlx-c-rust); the live build script now lives in the published
    // mlx-bridge crate outside this repository.
    "crates/mlx-c-rust/build.rs",
    "crates/native-build-orchestration/src/native_linking.rs",
    "crates/native-build-orchestration/src/native_store.rs",
    "crates/native-build-orchestration/src/native_store/manifest.rs",
    "crates/native-build-orchestration/src/native_compile.rs",
    "crates/native-build-orchestration/src/native-build-store-schema-version",
    "crates/runtime-integration/native",
    "crates/native-identity/src",
    "third-party/native-dependency-manifest.cmake",
    "third-party/pins",
    "third-party/patches",
    "rust-toolchain.toml",
];

/// Paths whose change can alter the text `cargo xtask notices` generates:
/// the resolved dependency graph, the pinned cargo-about, its configuration
/// and template, and the committed artifact itself. Deliberately narrower
/// than the digest inputs in `notices.rs`: workspace crates are `private`
/// (ignored by about.toml), so their manifests and license files only move
/// the generated text through `Cargo.lock`, and gating on every manifest edit
/// would pay a harvest for workspace version bumps that cannot change it.
const NOTICES_FRESHNESS_INPUT_PATHS: &[&str] = &[
    "Cargo.lock",
    "third-party/tool-pins/Cargo.toml",
    "third-party/about.toml",
    "third-party/rust-dependency-notices.hbs",
    "third-party/RUST_DEPENDENCY_NOTICES",
    "third-party/RUST_DEPENDENCY_NOTICES.digest",
];

/// Paths whose change gives a default-branch run something to publish: the
/// compiler, download and Swift caches are keyed on these, and pull requests can
/// only restore what the default branch saved. Code-only merges saved nothing in
/// every default-branch run measured, so only these still need that run. The
/// workflow and cache action join them because they define the cache keys. What the
/// cached xtask binary is keyed on stays out on purpose: those paths alone touched 11 of
/// 60 merges, each a full run of about 12 job-minutes, while a missed binary costs about
/// 9 seconds to rebuild until the next full run saves it.
const CACHE_INPUT_PATH_PREFIXES: &[&str] = &[
    "Cargo.lock",
    "rust-toolchain.toml",
    "third-party/tool-pins/",
    "third-party/native-dependency-manifest.cmake",
    "third-party/pins/",
    ".github/workflows/ci.yml",
    ".github/actions/",
    "apps/astronomical-menu/",
    "apps/thin-talk/",
];

/// Everything the Swift and Node lane builds or reads. It compiles no Rust and reads
/// nothing else, so a change outside these cannot change what the lane reports.
const SWIFT_NODE_INPUT_PATH_PREFIXES: &[&str] = &[
    "apps/astronomical-menu/",
    "apps/thin-talk/",
    "apps/daemon/console/",
    "scripts/internal/",
    "scripts/ci/",
    "site/schemas/",
    ".node-version",
    ".github/workflows/ci.yml",
    ".github/actions/",
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
    pub swift_node_required: bool,
    pub notices_inputs_changed: bool,
    pub classification: String,
    pub changed_file_count: usize,
}

/// Classifies one page/site artifact change as static-only. POSIX `case`
/// pattern semantics: `*` spans `/`, so `site/*` covers nested paths.
pub fn is_static_only_path(changed_path: &str) -> bool {
    is_published_content_path(changed_path) || is_repository_housekeeping_path(changed_path)
}

fn is_published_content_path(changed_path: &str) -> bool {
    changed_path.starts_with("site/")
        || changed_path.ends_with(".md")
        || changed_path.starts_with(".github/assets/")
        || changed_path == ".github/workflows/pages.yml"
}

/// None of these reach a build or a test: a fresh checkout has nothing to ignore.
fn is_repository_housekeeping_path(changed_path: &str) -> bool {
    changed_path == ".github/dependabot.yml"
        || changed_path == ".gitignore"
        || changed_path.ends_with("/.gitignore")
}

/// True when a change to this path gives a default-branch run a cache to save.
pub fn is_cache_input_path(changed_path: &str) -> bool {
    for cache_input_prefix in CACHE_INPUT_PATH_PREFIXES {
        if changed_path.starts_with(cache_input_prefix) {
            return true;
        }
    }
    false
}

/// Whether a default-branch push could skip the macOS lanes if its exact tree
/// already passed them, judged from what it changed. Asked first because the other
/// questions cost network calls.
pub fn may_reuse_verified_tree(
    event: CiEvent,
    verdict: &ChangeScopeVerdict,
    changed_paths: &[String],
) -> bool {
    if event != CiEvent::Push || !verdict.macos_verification_required {
        return false;
    }
    if verdict.native_inputs_changed {
        return false;
    }
    for changed_path in changed_paths {
        if is_cache_input_path(changed_path) {
            return false;
        }
    }
    true
}

/// The verdict for a push whose exact tree already passed every macOS lane on its
/// pull request. Those lanes would only repeat that proof on identical bytes.
pub fn verdict_for_verified_tree(verdict: ChangeScopeVerdict) -> ChangeScopeVerdict {
    ChangeScopeVerdict {
        macos_verification_required: false,
        swift_node_required: false,
        classification: "already_verified_tree".to_owned(),
        ..verdict
    }
}

/// True when the changed path can make the committed dependency notices
/// stale, so the pull request must prove them fresh before it merges.
pub fn is_notices_freshness_input_path(changed_path: &str) -> bool {
    NOTICES_FRESHNESS_INPUT_PATHS.contains(&changed_path)
}

/// True when a change to this path can change what the Swift and Node lane checks.
pub fn is_swift_node_input_path(changed_path: &str) -> bool {
    for swift_node_input_prefix in SWIFT_NODE_INPUT_PATH_PREFIXES {
        if changed_path.starts_with(swift_node_input_prefix) {
            return true;
        }
    }
    false
}

fn any_swift_node_input(changed_paths: &[String]) -> bool {
    for changed_path in changed_paths {
        if is_swift_node_input_path(changed_path) {
            return true;
        }
    }
    false
}

fn any_notices_freshness_input(changed_paths: &[String]) -> bool {
    for changed_path in changed_paths {
        if is_notices_freshness_input_path(changed_path) {
            return true;
        }
    }
    false
}

/// The pure verdict: identical inputs give identical outputs, no environment
/// access. PRs and pushes (PR merges) both gate on code changes; static-only
/// changes still skip, and `verdict_for_verified_tree` narrows a push further.
///
/// `is_version_only_bump` is a content proof made by the git edge (see
/// `version_only_bump`), never a path guess: a proven bump is identical to
/// the commit main already verified apart from the version string, so the
/// macOS lanes would only re-prove it.
pub fn classify_change_scope(
    event: CiEvent,
    changed_paths: &[String],
    native_inputs_changed: bool,
    is_version_only_bump: bool,
) -> ChangeScopeVerdict {
    match event {
        CiEvent::WorkflowDispatch => ChangeScopeVerdict {
            code_changed: true,
            native_inputs_changed: false,
            macos_verification_required: true,
            swift_node_required: true,
            notices_inputs_changed: true,
            classification: "manual_run".to_owned(),
            changed_file_count: 0,
        },
        CiEvent::PullRequest => {
            classify_changed_paths(changed_paths, native_inputs_changed, is_version_only_bump)
        }
        CiEvent::Push => with_swift_lane_on_every_full_run(classify_changed_paths(
            changed_paths,
            native_inputs_changed,
            is_version_only_bump,
        )),
    }
}

/// A full default-branch run keeps the Swift lane even with no Swift input changed,
/// because only that run saves its caches and the run's record tells later merges that
/// every cache is current. Without this, seven quiet days evict the Swift caches while
/// the record still says otherwise. `verdict_for_verified_tree` still skips it on a merge.
fn with_swift_lane_on_every_full_run(verdict: ChangeScopeVerdict) -> ChangeScopeVerdict {
    ChangeScopeVerdict {
        swift_node_required: verdict.macos_verification_required,
        ..verdict
    }
}

fn classify_changed_paths(
    changed_paths: &[String],
    native_inputs_changed: bool,
    is_version_only_bump: bool,
) -> ChangeScopeVerdict {
    let code_changed = changed_paths
        .iter()
        .any(|changed_path| !is_static_only_path(changed_path));
    let classification = if !code_changed {
        "static_only"
    } else if is_version_only_bump {
        "version_only_bump"
    } else if native_inputs_changed {
        "native_inputs"
    } else {
        "code_or_build"
    };
    ChangeScopeVerdict {
        code_changed,
        native_inputs_changed,
        macos_verification_required: code_changed && !is_version_only_bump,
        swift_node_required: code_changed
            && !is_version_only_bump
            && any_swift_node_input(changed_paths),
        notices_inputs_changed: any_notices_freshness_input(changed_paths),
        classification: classification.to_owned(),
        changed_file_count: changed_paths.len(),
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
        swift_node_required: true,
        notices_inputs_changed: true,
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
        CiEvent::WorkflowDispatch => classify_change_scope(event, &[], false, false),
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
            match resolve_verdict_from_history(repository_root, event, base_sha, head_sha) {
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
    event: CiEvent,
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
    let is_version_only_bump =
        version_only_bump_between(repository_root, &base_sha, &head_sha, &changed_paths);
    let verdict = classify_change_scope(
        CiEvent::PullRequest,
        &changed_paths,
        native_inputs_changed,
        is_version_only_bump,
    );
    Some(reuse_verified_tree_when_proven(
        repository_root,
        event,
        verdict,
        &changed_paths,
        &head_sha,
    ))
}

fn reuse_verified_tree_when_proven(
    repository_root: &Path,
    event: CiEvent,
    verdict: ChangeScopeVerdict,
    changed_paths: &[String],
    head_sha: &str,
) -> ChangeScopeVerdict {
    if may_reuse_verified_tree(event, &verdict, changed_paths)
        && is_tree_already_verified(repository_root, head_sha)
        && are_default_branch_caches_current(repository_root, head_sha)
    {
        return verdict_for_verified_tree(verdict);
    }
    verdict
}

/// Whether the last default-branch run that really ran the macOS lanes published
/// caches for this commit's inputs. The range starts at that run rather than the
/// last push, because a push that skipped those lanes published nothing: a version
/// bump changes `Cargo.lock` and skips them, and the merge after it must publish.
fn are_default_branch_caches_current(repository_root: &Path, head_sha: &str) -> bool {
    let Some(last_run_commit) = last_default_branch_run_commit() else {
        return false;
    };
    let is_current = commit_exists(repository_root, &last_run_commit)
        && !has_cache_input_change(repository_root, &last_run_commit, head_sha);
    eprintln!("[ci-change-scope] default_branch_caches_current={is_current}");
    is_current
}

/// Anything unreadable counts as a change, which only costs a full run.
fn has_cache_input_change(repository_root: &Path, base_sha: &str, head_sha: &str) -> bool {
    let Some(changed_paths) = changed_paths_between(repository_root, base_sha, head_sha) else {
        return true;
    };
    for changed_path in &changed_paths {
        if is_cache_input_path(changed_path) {
            return true;
        }
    }
    native_inputs_changed_between(repository_root, base_sha, head_sha) != Some(false)
}

/// Reads the base and head texts a version bump rewrites and asks the pure
/// proof. The files are read only when both appear in the diff, and any read
/// failure answers false: when the proof cannot be made, verify everything.
fn version_only_bump_between(
    repository_root: &Path,
    base_sha: &str,
    head_sha: &str,
    changed_paths: &[String],
) -> bool {
    if !changed_paths
        .iter()
        .any(|changed_path| changed_path == "Cargo.toml")
        || !changed_paths
            .iter()
            .any(|changed_path| changed_path == "Cargo.lock")
    {
        return false;
    }
    let Some(base_manifest) = file_text_at(repository_root, base_sha, "Cargo.toml") else {
        return false;
    };
    let Some(head_manifest) = file_text_at(repository_root, head_sha, "Cargo.toml") else {
        return false;
    };
    let Some(base_lockfile) = file_text_at(repository_root, base_sha, "Cargo.lock") else {
        return false;
    };
    let Some(head_lockfile) = file_text_at(repository_root, head_sha, "Cargo.lock") else {
        return false;
    };
    is_version_only_bump(
        changed_paths,
        &VersionBumpTexts {
            base_manifest: &base_manifest,
            head_manifest: &head_manifest,
            base_lockfile: &base_lockfile,
            head_lockfile: &head_lockfile,
        },
    )
}

fn file_text_at(repository_root: &Path, commit_sha: &str, file_path: &str) -> Option<String> {
    let git_output = Command::new("git")
        .arg("-C")
        .arg(repository_root)
        .args(["show", &format!("{commit_sha}:{file_path}")])
        .output()
        .ok()?;
    if !git_output.status.success() {
        return None;
    }
    String::from_utf8(git_output.stdout).ok()
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
    writeln!(
        github_output,
        "swift_node_required={}",
        verdict.swift_node_required
    )
    .unwrap();
    writeln!(
        github_output,
        "notices_inputs_changed={}",
        verdict.notices_inputs_changed
    )
    .unwrap();
    eprintln!(
        "[ci-change-scope] status=complete event={event_name} classification={} changed_files={} code_changed={} native_inputs_changed={} macos_verification_required={} notices_inputs_changed={} elapsed_seconds={}",
        verdict.classification,
        verdict.changed_file_count,
        verdict.code_changed,
        verdict.native_inputs_changed,
        verdict.macos_verification_required,
        verdict.notices_inputs_changed,
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
