use crate::github_actions_cache_prune::{
    Deletion, DeletionReason, GithubActionsCacheEntry, PruneOptions, completion_log_line,
    deletion_log_line, deletion_plan, family_of_key, parse_github_actions_cache_list_tsv,
    parse_prune_arguments,
};

fn github_actions_cache_entry(
    id: u64,
    key: &str,
    ref_name: &str,
    created_at: &str,
    size_bytes: u64,
) -> GithubActionsCacheEntry {
    GithubActionsCacheEntry {
        id,
        key: key.to_owned(),
        ref_name: ref_name.to_owned(),
        created_at: created_at.to_owned(),
        size_bytes,
    }
}

const DEFAULT_REF: &str = "refs/heads/main";

#[test]
fn should_keep_the_xtask_binary_keys_of_each_platform_in_their_own_family() {
    assert_eq!(
        family_of_key("astronomical-v2-xtask-binary-macOS-ARM64-digest"),
        "xtask-binary"
    );
    assert_eq!(
        family_of_key("astronomical-v2-xtask-binary-Linux-X64-digest"),
        "xtask"
    );
}

#[test]
fn family_extraction_stops_at_the_runner_platform_segment() {
    assert_eq!(
        family_of_key("astronomical-v2-sccache-macOS-ARM64-toolchain-lock"),
        "sccache"
    );
    assert_eq!(
        family_of_key("astronomical-v2-native-build-macOS-ARM64-identity"),
        "native-build"
    );
    assert_eq!(
        family_of_key("astronomical-v2-cargo-downloads-macOS"),
        "cargo-downloads"
    );
    // A key without the platform segment falls back to the first token, and
    // a foreign key is treated the same way (never mis-cut as another family).
    assert_eq!(family_of_key("astronomical-v2-swiftpm"), "swiftpm");
    assert_eq!(family_of_key("something-else"), "something");
}

#[test]
fn thin_talk_swiftpm_key_lands_in_its_own_prune_family() {
    // The menu and thin-talk SwiftPM entries of the `github_actions_cache`
    // share no bytes but once shared the `swiftpm` family, whose keep-one policy made the prune delete them
    // in alternation — every run rebuilt whichever package the previous run
    // had evicted (observed across the three main runs of 2026-10-31). The
    // thin-talk key therefore carries its own family segment, and this pin
    // keeps that segment distinct from the menu entry's family.
    assert_eq!(
        family_of_key("astronomical-v2-swiftpm-thin-talk-macOS-ARM64-toolchain-package-sources"),
        "swiftpm-thin-talk"
    );
    assert_eq!(
        family_of_key("astronomical-v2-swiftpm-macOS-ARM64-toolchain-package-sources"),
        "swiftpm"
    );
    assert_ne!(
        family_of_key("astronomical-v2-swiftpm-thin-talk-macOS-ARM64-toolchain-package-sources"),
        family_of_key("astronomical-v2-swiftpm-macOS-ARM64-toolchain-package-sources"),
        "menu and thin-talk SwiftPM entries of the github_actions_cache must prune in separate families"
    );
}

#[test]
fn plan_keeps_one_entry_per_swiftpm_family_not_one_entry_total() {
    // A menu entry and a thin-talk entry, same ref, thin-talk newest: the
    // keep-one-per-family policy must keep BOTH because they are different
    // families, which is the whole point of the family segment split.
    let entries = vec![
        github_actions_cache_entry(
            1,
            "astronomical-v2-swiftpm-macOS-ARM64-toolchain-menu-hash",
            "refs/heads/main",
            "2026-10-31T15:00:00Z",
            100,
        ),
        github_actions_cache_entry(
            2,
            "astronomical-v2-swiftpm-thin-talk-macOS-ARM64-toolchain-thin-talk-hash",
            "refs/heads/main",
            "2026-10-31T15:05:00Z",
            100,
        ),
    ];
    let deletions = deletion_plan(&entries, "refs/heads/main");
    assert!(
        deletions.is_empty(),
        "the prune deleted a swiftpm family entry that owns its own family slot: {deletions:?}"
    );
}

#[test]
fn plan_keeps_newest_per_ref_class_and_native_build_keeps_two() {
    let entries = vec![
        github_actions_cache_entry(
            1,
            "astronomical-v2-native-build-macOS-a",
            "refs/heads/main",
            "2026-10-01",
            100,
        ),
        github_actions_cache_entry(
            2,
            "astronomical-v2-native-build-macOS-b",
            "refs/heads/main",
            "2026-10-02",
            100,
        ),
        github_actions_cache_entry(
            3,
            "astronomical-v2-native-build-macOS-c",
            "refs/heads/main",
            "2026-10-03",
            100,
        ),
        github_actions_cache_entry(
            4,
            "astronomical-v2-native-build-macOS-d",
            "refs/heads/feature",
            "2026-10-04",
            100,
        ),
        github_actions_cache_entry(
            5,
            "astronomical-v2-native-build-macOS-e",
            "refs/heads/feature",
            "2026-10-05",
            100,
        ),
    ];
    let deleted_ids: Vec<u64> = deletion_plan(&entries, DEFAULT_REF)
        .into_iter()
        .map(|deletion| deletion.id)
        .collect();
    // Keeps the two newest on main and the single newest elsewhere; the plan
    // walks each family newest-first.
    assert_eq!(deleted_ids, vec![4, 1]);
}

#[test]
fn sccache_family_keeps_zero_non_default_ref_entries() {
    let entries = vec![
        github_actions_cache_entry(
            1,
            "astronomical-v2-sccache-macOS-a",
            "refs/heads/feature",
            "2026-10-05",
            100,
        ),
        github_actions_cache_entry(
            2,
            "astronomical-v2-sccache-macOS-b",
            "refs/heads/main",
            "2026-10-01",
            100,
        ),
        github_actions_cache_entry(
            3,
            "astronomical-v2-sccache-macOS-c",
            "refs/heads/main",
            "2026-10-02",
            100,
        ),
    ];
    let deleted_ids: Vec<u64> = deletion_plan(&entries, DEFAULT_REF)
        .into_iter()
        .map(|deletion| deletion.id)
        .collect();
    // The feature-ref entry is never superseded by a default-branch save, so
    // keeping it preserves a stale generation forever; and only the single
    // newest default-branch entry survives the family keep count.
    assert_eq!(deleted_ids, vec![1, 2]);
}

#[test]
fn oversize_sccache_entries_are_deleted_even_when_newest() {
    let entries = vec![
        github_actions_cache_entry(
            1,
            "astronomical-v2-sccache-macOS-newest",
            "refs/heads/main",
            "2026-10-09",
            700_000_000,
        ),
        github_actions_cache_entry(
            2,
            "astronomical-v2-sccache-macOS-older",
            "refs/heads/main",
            "2026-10-01",
            100_000_000,
        ),
    ];
    let plan = deletion_plan(&entries, DEFAULT_REF);
    assert_eq!(plan.len(), 1);
    assert_eq!(plan[0].id, 1);
    assert_eq!(plan[0].reason, DeletionReason::Oversize);
}

#[test]
fn non_sccache_families_have_no_size_ceiling() {
    let entries = vec![github_actions_cache_entry(
        1,
        "astronomical-v2-native-build-macOS-huge",
        "refs/heads/feature",
        "2026-10-09",
        u64::MAX / 4,
    )];
    let plan = deletion_plan(&entries, "refs/heads/main");
    assert!(
        plan.is_empty(),
        "native-build keeps its one feature-ref entry regardless of size"
    );
}

#[test]
fn tsv_parsing_round_trips_gh_output() {
    let tsv_text = "7\tastronomical-v2-sccache-macOS-a\trefs/heads/main\t2026-10-01T00:00:00Z\t123456\n\
                    9\tastronomical-v2-native-build-macOS-b\trefs/heads/feature\t2026-10-02T00:00:00Z\t42\n";
    let entries = parse_github_actions_cache_list_tsv(tsv_text);
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].id, 7);
    assert_eq!(entries[0].key, "astronomical-v2-sccache-macOS-a");
    assert_eq!(entries[1].size_bytes, 42);
    assert!(parse_github_actions_cache_list_tsv("not-enough-fields").is_empty());
}

fn argument_list(arguments: &[&str]) -> std::vec::IntoIter<String> {
    arguments
        .iter()
        .map(|argument| (*argument).to_owned())
        .collect::<Vec<_>>()
        .into_iter()
}

#[test]
fn should_read_the_repository_and_the_dry_run_flag_from_the_arguments() {
    let prune_options = parse_prune_arguments(
        argument_list(&["--dry-run", "--repo", "example-owner/example-repo"]),
        None,
    );

    assert_eq!(
        prune_options,
        Ok(PruneOptions {
            repository: "example-owner/example-repo".to_owned(),
            is_dry_run: true,
        })
    );
}

#[test]
fn should_fall_back_to_the_environment_repository_unless_repo_is_given() {
    let from_environment = parse_prune_arguments(
        argument_list(&[]),
        Some("environment-owner/environment-repo".to_owned()),
    );
    let from_argument = parse_prune_arguments(
        argument_list(&["--repo", "argument-owner/argument-repo"]),
        Some("environment-owner/environment-repo".to_owned()),
    );

    assert_eq!(
        from_environment,
        Ok(PruneOptions {
            repository: "environment-owner/environment-repo".to_owned(),
            is_dry_run: false,
        })
    );
    assert_eq!(
        from_argument.map(|prune_options| prune_options.repository),
        Ok("argument-owner/argument-repo".to_owned())
    );
}

#[test]
fn should_reject_an_unknown_prune_argument() {
    let prune_options = parse_prune_arguments(argument_list(&["--force"]), None);

    assert_eq!(
        prune_options,
        Err("unexpected prune-caches argument: --force".to_owned())
    );
}

#[test]
fn should_reject_a_repo_argument_without_a_value() {
    let prune_options = parse_prune_arguments(argument_list(&["--repo"]), None);

    assert_eq!(prune_options, Err("--repo requires owner/name".to_owned()));
}

#[test]
fn should_reject_a_run_that_names_no_repository() {
    let prune_options = parse_prune_arguments(argument_list(&["--dry-run"]), None);

    assert_eq!(
        prune_options,
        Err("no repository given; pass --repo owner/name or set GITHUB_REPOSITORY".to_owned())
    );
}

#[test]
fn should_log_an_oversize_deletion_with_its_action_reason_and_size() {
    let deletion = Deletion {
        id: 41,
        reason: DeletionReason::Oversize,
        size_bytes: 700_000_000,
    };

    assert_eq!(
        deletion_log_line("would-delete", &deletion),
        "[cache-prune] action=would-delete reason=oversize cache_id=41 size_bytes=700000000"
    );
}

#[test]
fn should_log_a_surplus_deletion_that_gh_removed() {
    let deletion = Deletion {
        id: 7,
        reason: DeletionReason::Surplus,
        size_bytes: 0,
    };

    assert_eq!(
        deletion_log_line("deleted", &deletion),
        "[cache-prune] action=deleted reason=surplus cache_id=7 size_bytes=0"
    );
}

#[test]
fn should_name_the_pruned_outcome_in_the_completion_line() {
    assert_eq!(
        completion_log_line(false, 3, 1_500, 12),
        "[cache-prune] status=complete outcome=pruned deleted=3 freed_bytes=1500 elapsed_seconds=12"
    );
}

#[test]
fn should_name_the_dry_run_outcome_with_zero_totals_in_the_completion_line() {
    assert_eq!(
        completion_log_line(true, 0, 0, 0),
        "[cache-prune] status=complete outcome=dry-run deleted=0 freed_bytes=0 elapsed_seconds=0"
    );
}
