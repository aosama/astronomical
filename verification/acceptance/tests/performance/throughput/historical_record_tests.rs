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
    let resolved_ledger = default_history_log_path();

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
fn should_prefer_the_history_environment_variable_over_the_workspace_default() {
    // The override exists so a machine can divert measurements to a scratch
    // ledger without touching the source-controlled one; presence wins, which
    // matches the historical semantics of the resolution (an empty value is
    // honored as the empty relative path rather than silently ignored).
    let override_directory =
        tempfile::tempdir().expect("the override ledger directory should be created");
    let override_ledger = override_directory.path().join("throughput-history.jsonl");

    // SAFETY: this test reads the previous value and restores it before
    // returning; the acceptance binaries run this module's tests in one
    // process, so the window without the variable is this assertion only.
    let previous_override = std::env::var_os(HISTORY_ENV_VAR);
    unsafe {
        std::env::set_var(HISTORY_ENV_VAR, &override_ledger);
    }
    let resolved_ledger = history_log_path();
    match previous_override {
        Some(previous) => unsafe {
            std::env::set_var(HISTORY_ENV_VAR, previous);
        },
        None => unsafe {
            std::env::remove_var(HISTORY_ENV_VAR);
        },
    }

    assert_eq!(
        resolved_ledger, override_ledger,
        "an explicit ledger override must take precedence over the anchored default"
    );
}
