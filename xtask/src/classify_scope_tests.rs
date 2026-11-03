use crate::classify_scope::{
    CiEvent, classify_change_scope, is_notices_freshness_input_path, is_static_only_path,
};

fn verdict_for(
    event: CiEvent,
    changed_paths: &[&str],
    native_inputs_changed: bool,
) -> (bool, bool, bool, String, usize) {
    let owned_paths: Vec<String> = changed_paths
        .iter()
        .map(|path| (*path).to_owned())
        .collect();
    let verdict = classify_change_scope(event, &owned_paths, native_inputs_changed);
    (
        verdict.code_changed,
        verdict.native_inputs_changed,
        verdict.macos_verification_required,
        verdict.classification,
        verdict.changed_file_count,
    )
}

#[test]
fn static_only_paths_span_site_docs_assets_and_the_pages_workflow() {
    assert!(is_static_only_path("site/index.html"));
    assert!(is_static_only_path("site/guides/deep/page.html"));
    assert!(is_static_only_path("docs/anything.md"));
    assert!(is_static_only_path(".github/assets/banner.png"));
    assert!(is_static_only_path(".github/workflows/pages.yml"));
    assert!(!is_static_only_path(".github/workflows/ci.yml"));
    assert!(!is_static_only_path("scripts/ci/ci-step-timing.sh"));
    assert!(!is_static_only_path("docs/readme.txt"));
}

#[test]
fn static_only_changes_skip_macos_verification_and_cache_publication() {
    assert_eq!(
        verdict_for(CiEvent::PullRequest, &["README.md", "site/a.html"], false),
        (false, false, false, "static_only".to_owned(), 2)
    );
}

#[test]
fn code_changes_require_verification_on_both_pr_and_push() {
    for event in [CiEvent::PullRequest, CiEvent::Push] {
        assert_eq!(
            verdict_for(event, &["crates/config/src/lib.rs"], false),
            (true, false, true, "code_or_build".to_owned(), 1)
        );
    }
}

#[test]
fn native_input_changes_are_labelled_separately() {
    assert_eq!(
        verdict_for(
            CiEvent::Push,
            &["third-party/pins/mlx.cmake", "README.md"],
            true
        ),
        (true, true, true, "native_inputs".to_owned(), 2)
    );
}

#[test]
fn mixed_static_and_code_changes_count_as_code() {
    assert_eq!(
        verdict_for(
            CiEvent::PullRequest,
            &["README.md", "apps/daemon/src/main.rs"],
            false
        ),
        (true, false, true, "code_or_build".to_owned(), 2)
    );
}

#[test]
fn manual_dispatch_verifies_everything_without_native_rebuild_assumptions() {
    assert_eq!(
        verdict_for(CiEvent::WorkflowDispatch, &[], false),
        (true, false, true, "manual_run".to_owned(), 0)
    );
}

#[test]
fn unknown_history_fails_safe_with_every_gate_open() {
    // Pinned by the CI coordination contract journey: an unresolvable range
    // must force code_changed true — deriving it from an empty path list
    // once silently closed the verification gate.
    let verdict = crate::classify_scope::fail_safe_verdict();
    assert!(verdict.code_changed);
    assert!(verdict.native_inputs_changed);
    assert!(verdict.macos_verification_required);
    assert!(verdict.notices_inputs_changed);
    assert_eq!(verdict.classification, "unknown_history");
    assert_eq!(verdict.changed_file_count, 0);
}

fn notices_inputs_changed_for(event: CiEvent, changed_paths: &[&str]) -> bool {
    let mut owned_paths: Vec<String> = Vec::new();
    for changed_path in changed_paths {
        owned_paths.push((*changed_path).to_owned());
    }
    classify_change_scope(event, &owned_paths, false).notices_inputs_changed
}

#[test]
fn should_flag_notices_inputs_when_a_dependency_bump_rewrites_the_lockfile() {
    // The Dependabot `rust-dependencies` group bump that merged green and
    // turned the nightly notices check red touched exactly these two files.
    for event in [CiEvent::PullRequest, CiEvent::Push] {
        assert!(notices_inputs_changed_for(
            event,
            &["Cargo.lock", "crates/config/Cargo.toml"]
        ));
    }
}

#[test]
fn should_flag_every_notices_generation_input_and_the_committed_artifact() {
    for notices_input_path in [
        "Cargo.lock",
        "third-party/tool-pins/Cargo.toml",
        "third-party/about.toml",
        "third-party/rust-dependency-notices.hbs",
        "third-party/RUST_DEPENDENCY_NOTICES",
        "third-party/RUST_DEPENDENCY_NOTICES.digest",
    ] {
        assert!(
            is_notices_freshness_input_path(notices_input_path),
            "{notices_input_path} must trigger the notices freshness lane"
        );
    }
}

#[test]
fn should_not_flag_notices_inputs_for_code_manifest_or_lookalike_paths() {
    // Workspace crates are private to cargo-about, so their manifests and
    // sources cannot change the notices text without also moving Cargo.lock.
    for event in [CiEvent::PullRequest, CiEvent::Push] {
        assert!(!notices_inputs_changed_for(
            event,
            &[
                "crates/config/Cargo.toml",
                "apps/daemon/src/main.rs",
                "Cargo.toml",
                "xtask/Cargo.lock",
                "third-party/tool-pins/lib.rs",
                "docs/Cargo.lock.md",
            ]
        ));
    }
}

#[test]
fn should_not_flag_notices_inputs_for_an_empty_change_range() {
    assert!(!notices_inputs_changed_for(CiEvent::PullRequest, &[]));
}

#[test]
fn should_flag_notices_inputs_on_manual_dispatch() {
    // A manual run verifies everything, the notices lane included.
    assert!(notices_inputs_changed_for(CiEvent::WorkflowDispatch, &[]));
}
