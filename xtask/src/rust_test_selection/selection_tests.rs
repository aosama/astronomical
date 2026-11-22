use std::collections::{BTreeMap, BTreeSet};

use super::selection::{RustTestSelection, select_rust_tests};
use super::workspace_graph_tests::fictional_graph;

fn names(values: &[&str]) -> BTreeSet<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

fn readers() -> BTreeMap<String, BTreeSet<String>> {
    BTreeMap::from([
        ("registry/catalog.json".to_owned(), names(&["engine"])),
        (
            "apps/thin-talk/Tests/Fixtures".to_owned(),
            names(&["daemon"]),
        ),
    ])
}

/// The lanes test every package but the tool.
fn lane_packages() -> BTreeSet<String> {
    names(&["core", "daemon", "e2e", "engine"])
}

fn select(changed: &[&str]) -> RustTestSelection {
    let changed_paths: Vec<String> = changed.iter().map(|path| (*path).to_owned()).collect();
    select_rust_tests(
        &changed_paths,
        &fictional_graph(),
        &readers(),
        &lane_packages(),
        false,
    )
}

#[test]
fn should_select_a_changed_package_and_everything_that_depends_on_it() {
    assert_eq!(
        select(&["crates/engine/src/lib.rs"]),
        RustTestSelection::Packages(names(&["daemon", "e2e", "engine"]))
    );
}

#[test]
fn should_select_the_readers_of_a_file_outside_every_package() {
    assert_eq!(
        select(&["registry/catalog.json"]),
        RustTestSelection::Packages(names(&["daemon", "e2e", "engine"]))
    );
    assert_eq!(
        select(&["apps/thin-talk/Tests/Fixtures/status.json"]),
        RustTestSelection::Packages(names(&["daemon", "e2e"]))
    );
}

#[test]
fn should_select_nothing_for_swift_web_and_static_changes_no_package_reads() {
    assert_eq!(
        select(&[
            "apps/thin-talk/Sources/View.swift",
            "apps/astronomical-menu/Sources/Menu.swift",
            "docs/guide.md",
            "site/index.html"
        ]),
        RustTestSelection::Packages(BTreeSet::new())
    );
}

#[test]
fn should_leave_out_a_package_the_lanes_do_not_test() {
    assert_eq!(
        select(&["tools/tool/src/main.rs"]),
        RustTestSelection::Packages(BTreeSet::new())
    );
}

#[test]
fn should_select_everything_for_a_manifest_a_workspace_setting_or_an_unplaced_path() {
    for changed_path in [
        "Cargo.lock",
        "crates/core/Cargo.toml",
        ".cargo/config.toml",
        "rust-toolchain.toml",
        "third-party/pins/mlx.cmake",
        ".github/workflows/ci.yml",
        "unknown/file.txt",
    ] {
        assert!(
            matches!(select(&[changed_path]), RustTestSelection::All { .. }),
            "{changed_path}"
        );
    }
}

#[test]
fn should_select_everything_when_native_inputs_changed() {
    let changed_paths = vec!["crates/core/src/lib.rs".to_owned()];
    assert!(matches!(
        select_rust_tests(
            &changed_paths,
            &fictional_graph(),
            &readers(),
            &lane_packages(),
            true
        ),
        RustTestSelection::All { .. }
    ));
}

#[test]
fn should_select_nothing_for_an_empty_change() {
    assert_eq!(select(&[]), RustTestSelection::Packages(BTreeSet::new()));
}
