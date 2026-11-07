use crate::classify_scope::{
    CiEvent, classify_change_scope, is_cache_input_path, is_notices_freshness_input_path,
    is_static_only_path, is_swift_node_input_path, may_reuse_verified_tree,
    verdict_for_verified_tree,
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
    let verdict = classify_change_scope(event, &owned_paths, native_inputs_changed, false);
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
    classify_change_scope(event, &owned_paths, false, false).notices_inputs_changed
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

fn verdict_with_bump_proof(
    changed_paths: &[&str],
    is_version_only_bump: bool,
) -> (bool, bool, String) {
    let mut owned_paths = Vec::new();
    for changed_path in changed_paths {
        owned_paths.push((*changed_path).to_owned());
    }
    let verdict = classify_change_scope(
        CiEvent::PullRequest,
        &owned_paths,
        false,
        is_version_only_bump,
    );
    (
        verdict.macos_verification_required,
        verdict.notices_inputs_changed,
        verdict.classification,
    )
}

#[test]
fn should_skip_macos_verification_for_a_proven_version_only_bump() {
    assert_eq!(
        verdict_with_bump_proof(&["Cargo.toml", "Cargo.lock"], true),
        (false, true, "version_only_bump".to_owned())
    );
}

#[test]
fn should_still_require_macos_verification_without_the_content_proof() {
    assert_eq!(
        verdict_with_bump_proof(&["Cargo.toml", "Cargo.lock"], false),
        (true, true, "code_or_build".to_owned())
    );
}

#[test]
fn should_never_let_a_manual_run_skip_macos_verification() {
    let verdict = classify_change_scope(CiEvent::WorkflowDispatch, &[], false, true);

    assert!(verdict.macos_verification_required);
    assert_eq!(verdict.classification, "manual_run");
}

#[test]
fn should_treat_repository_housekeeping_files_as_static() {
    assert!(is_static_only_path(".gitignore"));
    assert!(is_static_only_path("apps/thin-talk/web/.gitignore"));
    assert!(is_static_only_path(".github/dependabot.yml"));
}

#[test]
fn should_not_treat_lookalikes_of_housekeeping_files_as_static() {
    assert!(!is_static_only_path(".gitignored"));
    assert!(!is_static_only_path("src/not.gitignore"));
    assert!(!is_static_only_path(".gitattributes"));
    assert!(!is_static_only_path(".github/dependabot.yml.bak"));
}

#[test]
fn should_call_every_cache_key_input_a_cache_input() {
    for cache_input in [
        "Cargo.lock",
        "rust-toolchain.toml",
        "third-party/tool-pins/Cargo.toml",
        "third-party/native-dependency-manifest.cmake",
        "third-party/pins/mlx.cmake",
        ".github/workflows/ci.yml",
        ".github/actions/astronomical-cache/action.yml",
        "apps/astronomical-menu/Sources/Menu.swift",
        "apps/astronomical-menu/Package.resolved",
        "apps/thin-talk/Tests/Canvas.swift",
        "apps/thin-talk/web/package-lock.json",
    ] {
        assert!(is_cache_input_path(cache_input), "{cache_input}");
    }
}

#[test]
fn should_not_call_ordinary_code_a_cache_input() {
    for ordinary_path in [
        "crates/model-serving/src/lib.rs",
        "apps/daemon/src/main.rs",
        "apps/daemon/console/console.js",
        "third-party/about.toml",
        "Cargo.toml",
        ".cargo/config.toml",
        "xtask/src/main.rs",
        "crates/native-identity/src/lib.rs",
        "crates/test-harness/src/lib.rs",
    ] {
        assert!(!is_cache_input_path(ordinary_path), "{ordinary_path}");
    }
}

fn paths(changed_paths: &[&str]) -> Vec<String> {
    let mut owned_paths = Vec::new();
    for changed_path in changed_paths {
        owned_paths.push((*changed_path).to_owned());
    }
    owned_paths
}

fn code_verdict(changed_paths: &[String]) -> crate::classify_scope::ChangeScopeVerdict {
    classify_change_scope(CiEvent::Push, changed_paths, false, false)
}

#[test]
fn should_ask_about_a_verified_tree_for_a_code_only_push() {
    let changed_paths = paths(&["crates/config/src/lib.rs", "README.md"]);
    assert!(may_reuse_verified_tree(
        CiEvent::Push,
        &code_verdict(&changed_paths),
        &changed_paths
    ));
}

#[test]
fn should_never_reuse_a_verified_tree_on_a_pull_request_or_manual_run() {
    let changed_paths = paths(&["crates/config/src/lib.rs"]);
    for event in [CiEvent::PullRequest, CiEvent::WorkflowDispatch] {
        assert!(!may_reuse_verified_tree(
            event,
            &code_verdict(&changed_paths),
            &changed_paths
        ));
    }
}

#[test]
fn should_run_in_full_when_a_push_has_a_cache_to_publish() {
    let changed_paths = paths(&["crates/config/src/lib.rs", "Cargo.lock"]);
    assert!(!may_reuse_verified_tree(
        CiEvent::Push,
        &code_verdict(&changed_paths),
        &changed_paths
    ));
}

#[test]
fn should_run_in_full_when_native_inputs_changed() {
    let changed_paths = paths(&["crates/runtime-integration/native/shim.c"]);
    let verdict = classify_change_scope(CiEvent::Push, &changed_paths, true, false);
    assert!(!may_reuse_verified_tree(
        CiEvent::Push,
        &verdict,
        &changed_paths
    ));
}

#[test]
fn should_not_ask_when_macos_verification_was_never_required() {
    let changed_paths = paths(&["README.md"]);
    assert!(!may_reuse_verified_tree(
        CiEvent::Push,
        &code_verdict(&changed_paths),
        &changed_paths
    ));
}

#[test]
fn should_skip_only_the_macos_lanes_for_a_verified_tree() {
    let changed_paths = paths(&["crates/config/src/lib.rs", "Cargo.lock"]);
    let verdict = verdict_for_verified_tree(code_verdict(&changed_paths));
    assert!(!verdict.macos_verification_required);
    assert!(!verdict.swift_node_required);
    assert!(verdict.code_changed);
    assert!(verdict.notices_inputs_changed);
    assert_eq!(verdict.classification, "already_verified_tree");
    assert_eq!(verdict.changed_file_count, 2);
}

#[test]
fn should_call_everything_the_swift_and_node_lane_reads_an_input() {
    for swift_node_input in [
        "apps/astronomical-menu/Sources/AstronomicalMenuCore/Menu.swift",
        "apps/astronomical-menu/tests/AstronomicalMenuContractTests/Fixtures/full.json",
        "apps/thin-talk/web/src/app.ts",
        "apps/daemon/console/console.js",
        "scripts/internal/render-macos-app-icon.swift",
        "scripts/ci/ci-step-timing.sh",
        "site/schemas/config/v1/astronomical-config.schema.json",
        ".node-version",
        ".github/workflows/ci.yml",
        ".github/actions/assert-pinned-xcode/action.yml",
    ] {
        assert!(
            is_swift_node_input_path(swift_node_input),
            "{swift_node_input}"
        );
    }
}

#[test]
fn should_not_call_rust_code_a_swift_and_node_input() {
    for rust_path in [
        "crates/model-serving/src/lib.rs",
        "apps/daemon/src/main.rs",
        "apps/daemon/tests/rest_api/application/contracts.rs",
        "apps/astronomical-cli/src/main.rs",
        "Cargo.lock",
        "xtask/src/main.rs",
    ] {
        assert!(!is_swift_node_input_path(rust_path), "{rust_path}");
    }
}

#[test]
fn should_skip_only_the_swift_and_node_lane_for_a_rust_change() {
    let verdict = classify_change_scope(
        CiEvent::PullRequest,
        &paths(&["crates/config/src/lib.rs"]),
        false,
        false,
    );
    assert!(verdict.macos_verification_required);
    assert!(!verdict.swift_node_required);
}

#[test]
fn should_run_the_swift_and_node_lane_for_a_mixed_change() {
    let verdict = classify_change_scope(
        CiEvent::PullRequest,
        &paths(&["crates/config/src/lib.rs", "apps/daemon/console/console.js"]),
        false,
        false,
    );
    assert!(verdict.swift_node_required);
}

#[test]
fn should_require_the_swift_and_node_lane_when_the_range_is_unknown_or_manual() {
    assert!(crate::classify_scope::fail_safe_verdict().swift_node_required);
    assert!(
        classify_change_scope(CiEvent::WorkflowDispatch, &[], false, false).swift_node_required
    );
}

#[test]
fn should_not_require_the_swift_and_node_lane_for_a_version_bump_or_static_change() {
    let bump_paths = paths(&["Cargo.toml", "Cargo.lock"]);
    assert!(
        !classify_change_scope(CiEvent::PullRequest, &bump_paths, false, true).swift_node_required
    );
    let docs_paths = paths(&["apps/thin-talk/README.md"]);
    assert!(
        !classify_change_scope(CiEvent::PullRequest, &docs_paths, false, false).swift_node_required
    );
}

#[test]
fn should_run_the_swift_lane_on_a_full_push_with_no_swift_input() {
    let rust_paths = paths(&["crates/config/src/lib.rs"]);
    let push_verdict = classify_change_scope(CiEvent::Push, &rust_paths, false, false);
    assert!(push_verdict.macos_verification_required);
    assert!(
        push_verdict.swift_node_required,
        "only a full default-branch run saves the Swift caches"
    );
    let pull_request_verdict =
        classify_change_scope(CiEvent::PullRequest, &rust_paths, false, false);
    assert!(!pull_request_verdict.swift_node_required);
}

#[test]
fn should_keep_the_swift_lane_off_on_a_push_that_needs_no_macos_lanes() {
    let bump_paths = paths(&["Cargo.toml", "Cargo.lock"]);
    assert!(!classify_change_scope(CiEvent::Push, &bump_paths, false, true).swift_node_required);
    let docs_paths = paths(&["docs/guide.md"]);
    assert!(!classify_change_scope(CiEvent::Push, &docs_paths, false, false).swift_node_required);
    let rust_paths = paths(&["crates/config/src/lib.rs"]);
    let full_verdict = classify_change_scope(CiEvent::Push, &rust_paths, false, false);
    assert!(!verdict_for_verified_tree(full_verdict).swift_node_required);
}
