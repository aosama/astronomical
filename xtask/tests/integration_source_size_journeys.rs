//! `cargo xtask source-size --staged`, the pre-commit hook's check, through a
//! real git index. The staged scan reads the repository of the directory it
//! runs in, as the hook calls it from any checkout, so each journey runs the
//! built binary inside a sandbox repository. Ported from the staged scenarios
//! of the former `verification/tooling_contracts/test-source-size-check.sh`
//! (issue #2635).

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

/// Runs git in the sandbox; git's own messages reach the test output.
fn git(repository: &Path, arguments: &[&str]) {
    let git_status = Command::new("git")
        .current_dir(repository)
        .args(arguments)
        .status()
        .expect("git runs");
    assert!(git_status.success(), "git {arguments:?} failed");
}

fn sandbox_repository() -> tempfile::TempDir {
    let sandbox = tempfile::tempdir().expect("a sandbox should be creatable");
    git(sandbox.path(), &["init", "--quiet"]);
    sandbox
}

/// Writes a file of `line_count` copies of `line_text` and stages it.
fn stage_file(repository: &Path, relative_path: &str, line_count: usize, line_text: &str) {
    let staged_file = repository.join(relative_path);
    fs::create_dir_all(staged_file.parent().expect("a staged file has a folder"))
        .expect("the staged folder should be creatable");
    fs::write(&staged_file, format!("{line_text}\n").repeat(line_count))
        .expect("the staged file should be writable");
    git(repository, &["add", relative_path]);
}

fn run_staged_scan(repository: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["source-size", "--staged"])
        .current_dir(repository)
        .output()
        .expect("the xtask binary runs")
}

#[test]
fn should_pass_staged_sources_at_the_limit_in_every_language() {
    let sandbox = sandbox_repository();
    stage_file(sandbox.path(), "boundary.rs", 700, "// staged Rust line");
    stage_file(sandbox.path(), "boundary.py", 700, "# staged Python line");
    stage_file(
        sandbox.path(),
        "Boundary.swift",
        700,
        "// staged Swift line",
    );
    let scan_output = run_staged_scan(sandbox.path());
    assert!(
        scan_output.status.success(),
        "{}",
        String::from_utf8_lossy(&scan_output.stdout)
    );
    assert!(String::from_utf8_lossy(&scan_output.stdout).contains("mode=staged files=3"));
}

#[test]
fn should_judge_the_staged_content_not_the_working_file() {
    let sandbox = sandbox_repository();
    stage_file(sandbox.path(), "boundary.rs", 700, "// staged Rust line");
    let grown_working_file = format!("{}// unstaged line\n", "// staged Rust line\n".repeat(700));
    fs::write(sandbox.path().join("boundary.rs"), grown_working_file)
        .expect("the working file should be writable");
    assert!(run_staged_scan(sandbox.path()).status.success());
}

#[test]
fn should_skip_staged_documentation_generated_code_and_vendored_assets() {
    let sandbox = sandbox_repository();
    stage_file(sandbox.path(), "oversized.md", 900, "documentation line");
    stage_file(
        sandbox.path(),
        "Sources/App/Resources/web/vendor/bundle.min.js",
        900,
        "var vendored=1;",
    );
    let generated_python = format!(
        "# @generated deterministic binding\n{}",
        "x = 1\n".repeat(900)
    );
    fs::write(sandbox.path().join("generated.py"), generated_python)
        .expect("the generated file should be writable");
    git(sandbox.path(), &["add", "generated.py"]);
    assert!(run_staged_scan(sandbox.path()).status.success());
}

#[test]
fn should_fail_a_staged_source_over_the_limit_and_name_it() {
    let sandbox = sandbox_repository();
    stage_file(sandbox.path(), "too_long.cpp", 701, "// staged C++ line");
    let scan_output = run_staged_scan(sandbox.path());
    assert!(!scan_output.status.success());
    assert!(
        String::from_utf8_lossy(&scan_output.stdout).contains("limit=700 path=too_long.cpp"),
        "{}",
        String::from_utf8_lossy(&scan_output.stdout)
    );
}
