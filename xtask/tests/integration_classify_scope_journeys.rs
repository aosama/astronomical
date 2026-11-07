//! `cargo xtask classify-scope` deciding a default-branch push, through a real git
//! history and a shimmed `gh`. The decision skips macOS verification, so its process
//! boundary is the subject: every path that cannot prove the tree passed must leave
//! the full run in place, and a shim stands in so real GitHub state is never read.

use std::path::{Path, PathBuf};
use std::process::Command;

struct GhAnswers {
    tree_listing: String,
    run_listing: String,
}

const NO_RECORDS: &str = r#"{"total_count":0,"artifacts":[]}"#;

struct PushScenario {
    _sandbox: tempfile::TempDir,
    repository: PathBuf,
    shim_directory: PathBuf,
    call_marker: PathBuf,
    output_file: PathBuf,
    base_sha: String,
    head_sha: String,
}

fn git(repository: &Path, arguments: &[&str]) -> String {
    let git_output = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args([
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
        ])
        .args(arguments)
        .output()
        .expect("git runs");
    assert!(
        git_output.status.success(),
        "git {arguments:?} failed: {}",
        String::from_utf8_lossy(&git_output.stderr)
    );
    String::from_utf8_lossy(&git_output.stdout)
        .trim()
        .to_owned()
}

/// A repository whose last commit changes `changed_path`.
fn scenario_changing(changed_path: &str) -> PushScenario {
    let sandbox = tempfile::tempdir().expect("a temporary directory");
    let repository = sandbox.path().join("repository");
    std::fs::create_dir_all(&repository).expect("the repository folder is created");
    git(&repository, &["init", "--quiet"]);
    std::fs::write(repository.join("first.txt"), "first").expect("the file is written");
    git(&repository, &["add", "."]);
    git(&repository, &["commit", "--quiet", "-m", "first"]);
    let base_sha = git(&repository, &["rev-parse", "HEAD"]);
    let changed_file = repository.join(changed_path);
    std::fs::create_dir_all(changed_file.parent().expect("a parent folder"))
        .expect("the parent folder is created");
    std::fs::write(&changed_file, "changed").expect("the file is written");
    git(&repository, &["add", "."]);
    git(&repository, &["commit", "--quiet", "-m", "second"]);
    let head_sha = git(&repository, &["rev-parse", "HEAD"]);
    let shim_directory = sandbox.path().join("shim-bin");
    std::fs::create_dir_all(&shim_directory).expect("the shim folder is created");
    let call_marker = sandbox.path().join("gh-was-called");
    let output_file = sandbox.path().join("github-output");
    std::fs::write(&output_file, "").expect("the output file is created");
    PushScenario {
        _sandbox: sandbox,
        repository,
        shim_directory,
        call_marker,
        output_file,
        base_sha,
        head_sha,
    }
}

impl PushScenario {
    /// Moves the push's range forward by one commit that changes `changed_path`.
    fn commit_on_top(&mut self, changed_path: &str) {
        let changed_file = self.repository.join(changed_path);
        std::fs::create_dir_all(changed_file.parent().expect("a parent folder"))
            .expect("the parent folder is created");
        std::fs::write(&changed_file, "changed on top").expect("the file is written");
        git(&self.repository, &["add", "."]);
        git(&self.repository, &["commit", "--quiet", "-m", "on top"]);
        self.base_sha.clone_from(&self.head_sha);
        self.head_sha = git(&self.repository, &["rev-parse", "HEAD"]);
    }

    fn head_tree(&self) -> String {
        git(&self.repository, &["rev-parse", "HEAD^{tree}"])
    }

    /// Answers only the exact queries the classifier should make, so a broken query
    /// fails these journeys instead of quietly never skipping again. The shim's
    /// commands must stay shell builtins, because they are all it needs.
    fn install_gh(&self, answers: &GhAnswers, exit_code: i32) {
        let shim_script = format!(
            "#!/bin/sh\n: > '{marker}'\ncase \"$2\" in\n  'repos/owner/repository/actions/artifacts?name=verified-tree-{tree}&per_page=10') printf '%s\\n' '{tree_listing}' ;;\n  'repos/owner/repository/actions/artifacts?name=default-branch-macos-run&per_page=10') printf '%s\\n' '{run_listing}' ;;\n  *) exit 99 ;;\nesac\nexit {exit_code}\n",
            marker = self.call_marker.display(),
            tree = self.head_tree(),
            tree_listing = answers.tree_listing,
            run_listing = answers.run_listing,
        );
        let script_path = self.shim_directory.join("gh");
        std::fs::write(&script_path, shim_script).expect("the shim is written");
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script_path, std::fs::Permissions::from_mode(0o755))
            .expect("the shim is executable");
    }

    fn tree_record(&self, expired: bool, head_repository_id: u64) -> String {
        serde_json::json!({"artifacts": [{
            "name": format!("verified-tree-{}", self.head_tree()),
            "expired": expired,
            "workflow_run": {"repository_id": 7, "head_repository_id": head_repository_id}
        }]})
        .to_string()
    }

    /// A live record of a default-branch macOS run at this commit.
    fn run_record(&self, commit: &str) -> String {
        serde_json::json!({"artifacts": [{
            "name": "default-branch-macos-run",
            "expired": false,
            "created_at": "2026-11-06T10:00:00Z",
            "workflow_run": {
                "repository_id": 7,
                "head_repository_id": 7,
                "head_sha": commit
            }
        }]})
        .to_string()
    }

    /// Every proof a push needs in order to skip: its tree passed, and the default
    /// branch last ran the macOS lanes at the push's base.
    fn all_proofs(&self) -> GhAnswers {
        GhAnswers {
            tree_listing: self.tree_record(false, 7),
            run_listing: self.run_record(&self.base_sha),
        }
    }

    /// The published outputs of one classification.
    fn classify(&self, event_name: &str) -> String {
        let inherited_path = std::env::var("PATH").expect("PATH is set");
        let classify_output = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .arg("--repository-root")
            .arg(&self.repository)
            .arg("classify-scope")
            .env(
                "PATH",
                format!("{}:{inherited_path}", self.shim_directory.display()),
            )
            .env("EVENT_NAME", event_name)
            .env("PUSH_BEFORE_SHA", &self.base_sha)
            .env("CURRENT_SHA", &self.head_sha)
            .env("PULL_REQUEST_BASE_SHA", &self.base_sha)
            .env("PULL_REQUEST_HEAD_SHA", &self.head_sha)
            .env("GITHUB_OUTPUT", &self.output_file)
            .env("GITHUB_REPOSITORY", "owner/repository")
            .output()
            .expect("the classifier runs");
        assert!(
            classify_output.status.success(),
            "the classifier failed: {}",
            String::from_utf8_lossy(&classify_output.stderr)
        );
        std::fs::read_to_string(&self.output_file).expect("the outputs are readable")
    }

    fn was_gh_called(&self) -> bool {
        self.call_marker.exists()
    }
}

const MACOS_REQUIRED: &str = "macos_verification_required=true";
const MACOS_NOT_REQUIRED: &str = "macos_verification_required=false";

const CODE_PATH: &str = "crates/config/src/lib.rs";

#[test]
fn should_skip_the_macos_lanes_for_a_code_push_whose_tree_already_passed() {
    let scenario = scenario_changing(CODE_PATH);
    scenario.install_gh(&scenario.all_proofs(), 0);

    let outputs = scenario.classify("push");

    assert!(outputs.contains(MACOS_NOT_REQUIRED), "{outputs}");
    assert!(outputs.contains("code_changed=true"), "{outputs}");
}

#[test]
fn should_run_in_full_when_no_record_of_the_tree_exists() {
    let scenario = scenario_changing(CODE_PATH);
    let mut answers = scenario.all_proofs();
    answers.tree_listing = NO_RECORDS.to_owned();
    scenario.install_gh(&answers, 0);

    let outputs = scenario.classify("push");

    assert!(outputs.contains(MACOS_REQUIRED), "{outputs}");
    assert!(scenario.was_gh_called());
}

#[test]
fn should_run_in_full_when_the_tree_record_expired() {
    let scenario = scenario_changing(CODE_PATH);
    let mut answers = scenario.all_proofs();
    answers.tree_listing = scenario.tree_record(true, 7);
    scenario.install_gh(&answers, 0);

    assert!(scenario.classify("push").contains(MACOS_REQUIRED));
}

#[test]
fn should_run_in_full_when_the_tree_record_came_from_a_fork() {
    let scenario = scenario_changing(CODE_PATH);
    let mut answers = scenario.all_proofs();
    answers.tree_listing = scenario.tree_record(false, 8);
    scenario.install_gh(&answers, 0);

    assert!(scenario.classify("push").contains(MACOS_REQUIRED));
}

#[test]
fn should_run_in_full_when_github_cannot_answer() {
    let scenario = scenario_changing(CODE_PATH);
    scenario.install_gh(&scenario.all_proofs(), 1);

    assert!(scenario.classify("push").contains(MACOS_REQUIRED));
}

/// Its record's expiry is what refreshes the caches after a quiet spell.
#[test]
fn should_run_in_full_when_no_recent_default_branch_run_is_on_record() {
    let scenario = scenario_changing(CODE_PATH);
    let mut answers = scenario.all_proofs();
    answers.run_listing = NO_RECORDS.to_owned();
    scenario.install_gh(&answers, 0);

    assert!(scenario.classify("push").contains(MACOS_REQUIRED));
}

#[test]
fn should_run_in_full_when_the_recorded_commit_is_not_in_this_history() {
    let scenario = scenario_changing(CODE_PATH);
    let mut answers = scenario.all_proofs();
    answers.run_listing = scenario.run_record("0123456789012345678901234567890123456789");
    scenario.install_gh(&answers, 0);

    assert!(scenario.classify("push").contains(MACOS_REQUIRED));
}

/// A version bump changes `Cargo.lock` and skips the macOS lanes, so the merge after
/// it is the one that must publish the new caches, though it changes only code.
#[test]
fn should_run_in_full_after_a_skipped_push_left_a_cache_unpublished() {
    let mut scenario = scenario_changing("Cargo.lock");
    let last_macos_run = scenario.base_sha.clone();
    scenario.commit_on_top(CODE_PATH);
    let answers = GhAnswers {
        tree_listing: scenario.tree_record(false, 7),
        run_listing: scenario.run_record(&last_macos_run),
    };
    scenario.install_gh(&answers, 0);

    let outputs = scenario.classify("push");

    assert!(outputs.contains(MACOS_REQUIRED), "{outputs}");
}

#[test]
fn should_skip_again_once_the_default_branch_has_published_that_cache() {
    let mut scenario = scenario_changing("Cargo.lock");
    let publishing_run = scenario.head_sha.clone();
    scenario.commit_on_top(CODE_PATH);
    let answers = GhAnswers {
        tree_listing: scenario.tree_record(false, 7),
        run_listing: scenario.run_record(&publishing_run),
    };
    scenario.install_gh(&answers, 0);

    assert!(scenario.classify("push").contains(MACOS_NOT_REQUIRED));
}

#[test]
fn should_run_in_full_without_asking_when_a_cache_input_changed() {
    let scenario = scenario_changing("Cargo.lock");
    scenario.install_gh(&scenario.all_proofs(), 0);

    let outputs = scenario.classify("push");

    assert!(outputs.contains(MACOS_REQUIRED), "{outputs}");
    assert!(
        !scenario.was_gh_called(),
        "no record can excuse a cache to publish"
    );
}

#[test]
fn should_run_in_full_without_asking_on_a_first_push_with_no_earlier_commit() {
    let mut scenario = scenario_changing(CODE_PATH);
    scenario.base_sha = "0".repeat(40);
    let answers = GhAnswers {
        tree_listing: scenario.tree_record(false, 7),
        run_listing: NO_RECORDS.to_owned(),
    };
    scenario.install_gh(&answers, 0);

    let outputs = scenario.classify("push");

    assert!(outputs.contains(MACOS_REQUIRED), "{outputs}");
    assert!(!scenario.was_gh_called());
}

#[test]
fn should_never_skip_a_pull_request_whatever_the_records_say() {
    let scenario = scenario_changing(CODE_PATH);
    scenario.install_gh(&scenario.all_proofs(), 0);

    let outputs = scenario.classify("pull_request");

    assert!(outputs.contains(MACOS_REQUIRED), "{outputs}");
    assert!(!scenario.was_gh_called());
}

#[test]
fn should_skip_the_macos_lanes_for_a_gitignore_change_without_asking() {
    let scenario = scenario_changing(".gitignore");
    scenario.install_gh(&scenario.all_proofs(), 0);

    let outputs = scenario.classify("pull_request");

    assert!(outputs.contains(MACOS_NOT_REQUIRED), "{outputs}");
    assert!(outputs.contains("code_changed=false"), "{outputs}");
    assert!(!scenario.was_gh_called());
}
