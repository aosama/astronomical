use crate::cache_prune::{
    CacheEntry, DeletionReason, deletion_plan, family_of_key, parse_cache_list_tsv,
};

fn cache_entry(
    id: u64,
    key: &str,
    ref_name: &str,
    created_at: &str,
    size_bytes: u64,
) -> CacheEntry {
    CacheEntry {
        id,
        key: key.to_owned(),
        ref_name: ref_name.to_owned(),
        created_at: created_at.to_owned(),
        size_bytes,
    }
}

const DEFAULT_REF: &str = "refs/heads/main";

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
fn plan_keeps_newest_per_ref_class_and_native_build_keeps_two() {
    let entries = vec![
        cache_entry(
            1,
            "astronomical-v2-native-build-macOS-a",
            "refs/heads/main",
            "2026-10-01",
            100,
        ),
        cache_entry(
            2,
            "astronomical-v2-native-build-macOS-b",
            "refs/heads/main",
            "2026-10-02",
            100,
        ),
        cache_entry(
            3,
            "astronomical-v2-native-build-macOS-c",
            "refs/heads/main",
            "2026-10-03",
            100,
        ),
        cache_entry(
            4,
            "astronomical-v2-native-build-macOS-d",
            "refs/heads/feature",
            "2026-10-04",
            100,
        ),
        cache_entry(
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
        cache_entry(
            1,
            "astronomical-v2-sccache-macOS-a",
            "refs/heads/feature",
            "2026-10-05",
            100,
        ),
        cache_entry(
            2,
            "astronomical-v2-sccache-macOS-b",
            "refs/heads/main",
            "2026-10-01",
            100,
        ),
        cache_entry(
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
        cache_entry(
            1,
            "astronomical-v2-sccache-macOS-newest",
            "refs/heads/main",
            "2026-10-09",
            500_000_000,
        ),
        cache_entry(
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
    let entries = vec![cache_entry(
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
    let entries = parse_cache_list_tsv(tsv_text);
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].id, 7);
    assert_eq!(entries[0].key, "astronomical-v2-sccache-macOS-a");
    assert_eq!(entries[1].size_bytes, 42);
    assert!(parse_cache_list_tsv("not-enough-fields").is_empty());
}
