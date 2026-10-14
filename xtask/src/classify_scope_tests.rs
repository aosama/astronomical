use astronomical_test_harness::offline_test;

use crate::classify_scope::{CiEvent, classify_change_scope, is_static_only_path};

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

offline_test!(
    static_only_paths_span_site_docs_assets_and_the_pages_workflow,
    {
        assert!(is_static_only_path("site/index.html"));
        assert!(is_static_only_path("site/guides/deep/page.html"));
        assert!(is_static_only_path("docs/anything.md"));
        assert!(is_static_only_path(".github/assets/banner.png"));
        assert!(is_static_only_path(".github/workflows/pages.yml"));
        assert!(!is_static_only_path(".github/workflows/ci.yml"));
        assert!(!is_static_only_path(
            "scripts/ci/bootstrap-native-dependencies.sh"
        ));
        assert!(!is_static_only_path("docs/readme.txt"));
    }
);

offline_test!(
    static_only_changes_skip_macos_verification_and_cache_publication,
    {
        assert_eq!(
            verdict_for(CiEvent::PullRequest, &["README.md", "site/a.html"], false),
            (false, false, false, "static_only".to_owned(), 2)
        );
    }
);

offline_test!(code_changes_require_verification_on_both_pr_and_push, {
    for event in [CiEvent::PullRequest, CiEvent::Push] {
        assert_eq!(
            verdict_for(event, &["crates/config/src/lib.rs"], false),
            (true, false, true, "code_or_build".to_owned(), 1)
        );
    }
});

offline_test!(native_input_changes_are_labelled_separately, {
    assert_eq!(
        verdict_for(
            CiEvent::Push,
            &["third-party/pins/mlx.cmake", "README.md"],
            true
        ),
        (true, true, true, "native_inputs".to_owned(), 2)
    );
});

offline_test!(mixed_static_and_code_changes_count_as_code, {
    assert_eq!(
        verdict_for(
            CiEvent::PullRequest,
            &["README.md", "apps/daemon/src/main.rs"],
            false
        ),
        (true, false, true, "code_or_build".to_owned(), 2)
    );
});

offline_test!(
    manual_dispatch_verifies_everything_without_native_rebuild_assumptions,
    {
        assert_eq!(
            verdict_for(CiEvent::WorkflowDispatch, &[], false),
            (true, false, true, "manual_run".to_owned(), 0)
        );
    }
);

offline_test!(unknown_history_fails_safe_with_every_gate_open, {
    // Pinned by the CI coordination contract journey: an unresolvable range
    // must force code_changed true — deriving it from an empty path list
    // once silently closed the verification gate.
    let verdict = crate::classify_scope::fail_safe_verdict();
    assert!(verdict.code_changed);
    assert!(verdict.native_inputs_changed);
    assert!(verdict.macos_verification_required);
    assert_eq!(verdict.classification, "unknown_history");
    assert_eq!(verdict.changed_file_count, 0);
});
