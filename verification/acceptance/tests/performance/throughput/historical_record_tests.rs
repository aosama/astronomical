//! Regression coverage for the durable throughput-history ledger resolution.

use std::path::Path;

use super::*;

#[test]
fn should_anchor_the_default_history_ledger_to_the_workspace_root_regardless_of_launch_directory() {
    // The resolution used to be relative to the test process's working
    // directory, and the standard acceptance launcher starts binaries with the
    // package directory as their working directory, so measurements forked
    // into a doubled nested copy of the ledger path instead of the
    // source-controlled file. The compile-time manifest anchor makes the
    // absolute path itself the regression signal: a launch-directory-relative
    // resolution can never produce one.
    let resolved_ledger = resolve_history_log_path(None);

    assert!(
        resolved_ledger.is_absolute(),
        "the default ledger path must be anchored, not launch-directory relative: {resolved_ledger:?}"
    );

    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let expected_ledger = workspace_root.join(DEFAULT_HISTORY_RELATIVE_PATH);
    assert_eq!(
        resolved_ledger, expected_ledger,
        "the default ledger must resolve to the source-controlled history file"
    );

    let canonical_ledger = resolved_ledger
        .canonicalize()
        .expect("the source-controlled history ledger should exist on disk");
    let canonical_ledger_text = canonical_ledger.to_string_lossy();
    assert!(
        !canonical_ledger_text.contains("verification/acceptance/verification/acceptance"),
        "the resolved ledger must not live in a doubled nested copy of the package tree: {canonical_ledger_text}"
    );
    assert!(
        canonical_ledger.is_file(),
        "the canonical ledger must be the existing source-controlled file: {canonical_ledger_text}"
    );
}

#[test]
fn should_prefer_the_explicit_history_path_over_the_workspace_default() {
    let override_ledger =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("target/selected-throughput-history.jsonl");
    let resolved_ledger = resolve_history_log_path(Some(&override_ledger));
    assert_eq!(
        resolved_ledger, override_ledger,
        "an explicit ledger override must take precedence over the anchored default"
    );
}

#[test]
fn should_preserve_spaces_in_the_explicit_history_path() {
    let configured_history_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/measurement evidence/selected-throughput-history.jsonl");

    assert_eq!(
        resolve_history_log_path(Some(&configured_history_path)),
        configured_history_path
    );
}

#[test]
fn should_record_actual_worker_digest_and_dirty_source_state() {
    let mut run_context = TestRunContext {
        source_commit: Some("fixture-commit".to_owned()),
        source_is_dirty: true,
        ..TestRunContext::default()
    };
    run_context
        .workspace_binary_sha256
        .insert("astronomical-inference-worker".to_owned(), "ab".repeat(32));
    let build_identity = worker_build_identity(&run_context).expect("worker identity");
    assert_eq!(build_identity.source_commit, "fixture-commit");
    assert!(build_identity.source_is_dirty);
    assert_eq!(build_identity.executable_sha256, "ab".repeat(32));
}

#[test]
fn should_reject_missing_source_commit_or_worker_identity_before_measurement() {
    let mut run_context = TestRunContext::default();
    assert!(worker_build_identity(&run_context).is_err());
    run_context.source_commit = Some("fixture-commit".to_owned());
    assert!(worker_build_identity(&run_context).is_err());
}
