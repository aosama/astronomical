use serde_json::{Value, json};

use crate::host_platforms::packages_unbuildable_on_host;

/// A `cargo metadata --no-deps` package with the given path dependencies and
/// host-platforms declaration.
fn workspace_package(
    package_name: &str,
    path_dependencies: &[&str],
    host_platforms: Value,
) -> Value {
    let mut package_dependencies = Vec::new();
    for dependency_name in path_dependencies {
        package_dependencies.push(
            json!({"name": dependency_name, "path": format!("/workspace/{dependency_name}")}),
        );
    }
    let package_metadata = if host_platforms.is_null() {
        Value::Null
    } else {
        json!({"astronomical": {"host-platforms": host_platforms}})
    };
    json!({"name": package_name, "dependencies": package_dependencies, "metadata": package_metadata})
}

/// apple-only builds on macOS only; engine uses it, server uses engine,
/// and portable stands alone.
fn layered_workspace() -> Value {
    json!({"packages": [
        workspace_package("apple-only", &[], json!(["macos"])),
        workspace_package("engine", &["apple-only"], Value::Null),
        workspace_package("server", &["engine"], Value::Null),
        workspace_package("portable", &[], Value::Null),
    ]})
}

#[test]
fn should_leave_out_nothing_on_macos() {
    let packages_left_out = packages_unbuildable_on_host(&layered_workspace(), "macos").unwrap();

    assert!(packages_left_out.is_empty());
}

#[test]
fn should_leave_out_a_macos_only_package_and_everything_built_on_it_elsewhere() {
    let packages_left_out = packages_unbuildable_on_host(&layered_workspace(), "linux").unwrap();

    assert_eq!(packages_left_out, ["apple-only", "engine", "server"]);
}

#[test]
fn should_keep_a_package_that_lists_the_host_among_several_platforms() {
    let workspace_metadata = json!({"packages": [
        workspace_package("cross-platform", &[], json!(["macos", "linux"])),
    ]});

    assert!(
        packages_unbuildable_on_host(&workspace_metadata, "linux")
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        packages_unbuildable_on_host(&workspace_metadata, "windows").unwrap(),
        ["cross-platform"]
    );
}

#[test]
fn should_leave_out_a_package_that_lists_no_platform_at_all() {
    let workspace_metadata = json!({"packages": [workspace_package("nowhere", &[], json!([]))]});

    assert_eq!(
        packages_unbuildable_on_host(&workspace_metadata, "macos").unwrap(),
        ["nowhere"]
    );
}

#[test]
fn should_ignore_a_registry_dependency_that_shares_a_name_with_a_left_out_package() {
    let workspace_metadata = json!({"packages": [
        workspace_package("apple-only", &[], json!(["macos"])),
        {"name": "uses-registry-crate", "dependencies": [{"name": "apple-only", "path": null}], "metadata": null},
    ]});

    assert_eq!(
        packages_unbuildable_on_host(&workspace_metadata, "linux").unwrap(),
        ["apple-only"]
    );
}

#[test]
fn should_refuse_a_declaration_that_is_not_a_list() {
    let workspace_metadata =
        json!({"packages": [workspace_package("misdeclared", &[], json!("macos"))]});

    let refusal = packages_unbuildable_on_host(&workspace_metadata, "linux").unwrap_err();
    assert!(refusal.contains("misdeclared"), "{refusal}");
}

#[test]
fn should_refuse_metadata_without_packages() {
    assert!(packages_unbuildable_on_host(&json!({}), "linux").is_err());
}
