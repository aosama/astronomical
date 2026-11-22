use std::collections::BTreeSet;
use std::path::Path;

use super::workspace_graph::WorkspaceGraph;

/// A small fictional workspace: core <- engine <- daemon <-(dev) e2e, and an
/// unrelated tool. The daemon has a REST test target; e2e an `e2e` target.
pub(crate) const FICTIONAL_METADATA: &str = r#"{"packages":[
 {"name":"core","manifest_path":"/repo/crates/core/Cargo.toml","targets":[{"name":"core","kind":["lib"]}],"dependencies":[{"name":"serde_json","kind":null}]},
 {"name":"engine","manifest_path":"/repo/crates/engine/Cargo.toml","targets":[{"name":"engine","kind":["lib"]},{"name":"integration_tests","kind":["test"]}],"dependencies":[{"name":"core","kind":null,"path":"/repo/crates/core"}]},
 {"name":"daemon","manifest_path":"/repo/apps/daemon/Cargo.toml","targets":[{"name":"daemon","kind":["lib"]},{"name":"rest_api_tests","kind":["test"]},{"name":"integration_tests","kind":["test"]}],"dependencies":[{"name":"engine","kind":null,"path":"/repo/crates/engine"}]},
 {"name":"e2e","manifest_path":"/repo/verification/e2e/Cargo.toml","targets":[{"name":"e2e","kind":["test"]}],"dependencies":[{"name":"daemon","kind":"dev","path":"/repo/apps/daemon"}]},
 {"name":"tool","manifest_path":"/repo/tools/tool/Cargo.toml","targets":[{"name":"tool","kind":["bin"]}],"dependencies":[]}
]}"#;

pub(crate) fn fictional_graph() -> WorkspaceGraph {
    WorkspaceGraph::from_metadata(FICTIONAL_METADATA, Path::new("/repo"))
        .expect("the fictional metadata parses")
}

fn package_names(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

#[test]
fn should_read_folders_test_targets_and_local_dependencies() {
    let workspace_graph = fictional_graph();
    let daemon = &workspace_graph.packages["daemon"];
    assert_eq!(daemon.relative_directory, "apps/daemon");
    assert_eq!(
        daemon.test_target_names,
        package_names(&["integration_tests", "rest_api_tests"])
    );
    assert_eq!(
        workspace_graph.packages["core"].local_dependency_names,
        BTreeSet::new(),
        "a registry dependency is not local"
    );
}

#[test]
fn should_count_a_dev_dependency_as_a_dependent_edge() {
    assert_eq!(
        fictional_graph().with_dependents(&package_names(&["daemon"])),
        package_names(&["daemon", "e2e"])
    );
}

#[test]
fn should_widen_a_change_to_every_package_that_depends_on_it() {
    assert_eq!(
        fictional_graph().with_dependents(&package_names(&["core"])),
        package_names(&["core", "daemon", "e2e", "engine"])
    );
    assert_eq!(
        fictional_graph().with_dependents(&BTreeSet::new()),
        BTreeSet::new()
    );
}

#[test]
fn should_place_a_path_in_its_package_and_nothing_outside_one() {
    let workspace_graph = fictional_graph();
    assert_eq!(
        workspace_graph.owning_package("apps/daemon/console/page.js"),
        Some("daemon")
    );
    assert_eq!(
        workspace_graph.owning_package("apps/daemon"),
        Some("daemon")
    );
    assert_eq!(
        workspace_graph.owning_package("apps/daemon-notes.md"),
        None,
        "a sibling with the same prefix is not inside"
    );
    assert_eq!(
        workspace_graph.owning_package("registry/catalog.json"),
        None
    );
}

#[test]
fn should_refuse_metadata_that_is_not_json_or_has_no_packages() {
    assert!(WorkspaceGraph::from_metadata("not json", Path::new("/repo")).is_err());
    assert!(WorkspaceGraph::from_metadata("{}", Path::new("/repo")).is_err());
}
