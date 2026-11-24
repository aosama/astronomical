use std::path::Path;

use crate::model_loading_changed::{
    ADDITIONAL_MODEL_LOADING_PATHS, model_loading_paths, parse_previous_tag,
    unchanged_flag_from_git_exit_code,
};

const REPOSITORY_ROOT: &str = "/fixture/repository";

/// One workspace package for a fixture document: its name, its folder under
/// the repository root, and its dependencies as (name, kind, is_local).
struct FixturePackage<'fixture> {
    name: &'fixture str,
    folder: String,
    dependencies: Vec<(&'fixture str, Option<&'fixture str>, bool)>,
}

/// A minimal `cargo metadata --no-deps` document for the given packages.
fn metadata_document_for(packages: &[FixturePackage]) -> String {
    let mut package_entries = Vec::new();
    for package in packages {
        let mut dependency_entries = Vec::new();
        for (dependency_name, dependency_kind, is_local) in &package.dependencies {
            let kind_json = match dependency_kind {
                Some(kind_text) => format!("\"{kind_text}\""),
                None => "null".to_owned(),
            };
            let path_json = if *is_local {
                format!(",\"path\":\"{REPOSITORY_ROOT}/crates/{dependency_name}\"")
            } else {
                String::new()
            };
            dependency_entries.push(format!(
                "{{\"name\":\"{dependency_name}\",\"kind\":{kind_json}{path_json}}}"
            ));
        }
        package_entries.push(format!(
            "{{\"name\":\"{}\",\"manifest_path\":\"{REPOSITORY_ROOT}/{}/Cargo.toml\",\"dependencies\":[{}]}}",
            package.name,
            package.folder,
            dependency_entries.join(",")
        ));
    }
    format!("{{\"packages\":[{}]}}", package_entries.join(","))
}

fn worker_package(
    dependencies: Vec<(&'static str, Option<&'static str>, bool)>,
) -> FixturePackage<'static> {
    FixturePackage {
        name: "astronomical-inference-worker",
        folder: "apps/inference-worker".to_owned(),
        dependencies,
    }
}

fn library_package(
    name: &'static str,
    dependencies: Vec<(&'static str, Option<&'static str>, bool)>,
) -> FixturePackage<'static> {
    FixturePackage {
        name,
        folder: format!("crates/{name}"),
        dependencies,
    }
}

fn expected_paths_with(closure_folders: &[&str]) -> Vec<String> {
    let mut expected_paths: Vec<String> = closure_folders
        .iter()
        .chain(ADDITIONAL_MODEL_LOADING_PATHS.iter())
        .map(|watched_path| (*watched_path).to_owned())
        .collect();
    expected_paths.sort();
    expected_paths.dedup();
    expected_paths
}

#[test]
fn should_watch_the_workers_transitive_local_closure_and_the_additional_paths() {
    let metadata_json_text = metadata_document_for(&[
        worker_package(vec![("model-serving", None, true)]),
        library_package("model-serving", vec![("qwen35", None, true)]),
        library_package("qwen35", vec![("native-identity", Some("build"), true)]),
        library_package("native-identity", Vec::new()),
    ]);
    let watched_paths =
        model_loading_paths(&metadata_json_text, Path::new(REPOSITORY_ROOT)).unwrap();
    assert_eq!(
        watched_paths,
        expected_paths_with(&[
            "apps/inference-worker",
            "crates/engine/model-serving",
            "crates/families/text/qwen35",
            "crates/foundation/native-identity",
        ])
    );
}

#[test]
fn should_not_watch_dev_only_or_registry_dependencies_or_unreached_packages() {
    let metadata_json_text = metadata_document_for(&[
        worker_package(vec![
            ("test-harness", Some("dev"), true),
            ("serde_json", None, false),
        ]),
        library_package("test-harness", Vec::new()),
        library_package("download-catalog", Vec::new()),
    ]);
    let watched_paths =
        model_loading_paths(&metadata_json_text, Path::new(REPOSITORY_ROOT)).unwrap();
    assert_eq!(
        watched_paths,
        expected_paths_with(&["apps/inference-worker"])
    );
}

#[test]
fn should_list_a_diamond_dependency_once_in_sorted_order() {
    let metadata_json_text = metadata_document_for(&[
        worker_package(vec![("speech", None, true), ("qwen35", None, true)]),
        library_package("speech", vec![("memory-policy", None, true)]),
        library_package("qwen35", vec![("memory-policy", None, true)]),
        library_package("memory-policy", Vec::new()),
    ]);
    let watched_paths =
        model_loading_paths(&metadata_json_text, Path::new(REPOSITORY_ROOT)).unwrap();
    let memory_policy_entries = watched_paths
        .iter()
        .filter(|watched_path| watched_path.as_str() == "crates/foundation/memory-policy")
        .count();
    assert_eq!(memory_policy_entries, 1);
    let mut sorted_paths = watched_paths.clone();
    sorted_paths.sort();
    assert_eq!(watched_paths, sorted_paths);
}

#[test]
fn should_fail_closed_when_the_worker_package_is_missing() {
    let metadata_json_text = metadata_document_for(&[library_package("qwen35", Vec::new())]);
    let failure = model_loading_paths(&metadata_json_text, Path::new(REPOSITORY_ROOT)).unwrap_err();
    assert!(
        failure.contains("reason=worker-package-missing"),
        "{failure}"
    );
}

#[test]
fn should_fail_closed_on_an_unknown_dependency_kind() {
    let metadata_json_text = metadata_document_for(&[worker_package(vec![(
        "qwen35",
        Some("optional-later"),
        true,
    )])]);
    let failure = model_loading_paths(&metadata_json_text, Path::new(REPOSITORY_ROOT)).unwrap_err();
    assert!(
        failure.contains("reason=unknown-dependency-kind"),
        "{failure}"
    );
}

#[test]
fn should_fail_closed_on_a_package_outside_the_repository() {
    let metadata_json_text = metadata_document_for(&[worker_package(Vec::new())]);
    let failure =
        model_loading_paths(&metadata_json_text, Path::new("/another/repository")).unwrap_err();
    assert!(
        failure.contains("reason=package-outside-repository"),
        "{failure}"
    );
}

#[test]
fn should_fail_closed_on_metadata_that_is_not_json() {
    let failure = model_loading_paths("not json", Path::new(REPOSITORY_ROOT)).unwrap_err();
    assert!(failure.contains("reason=metadata-not-json"), "{failure}");
}

#[test]
fn should_fail_closed_on_metadata_without_packages() {
    let failure = model_loading_paths("{}", Path::new(REPOSITORY_ROOT)).unwrap_err();
    assert!(
        failure.contains("reason=metadata-without-packages"),
        "{failure}"
    );
}

#[test]
fn should_fail_closed_on_a_package_without_a_name() {
    let metadata_json_text = format!(
        "{{\"packages\":[{{\"manifest_path\":\"{REPOSITORY_ROOT}/apps/inference-worker/Cargo.toml\",\"dependencies\":[]}}]}}"
    );
    let failure = model_loading_paths(&metadata_json_text, Path::new(REPOSITORY_ROOT)).unwrap_err();
    assert!(
        failure.contains("reason=metadata-missing-field field=name"),
        "{failure}"
    );
}

#[test]
fn should_fail_closed_on_a_package_without_a_dependencies_array() {
    let metadata_json_text = format!(
        "{{\"packages\":[{{\"name\":\"astronomical-inference-worker\",\"manifest_path\":\"{REPOSITORY_ROOT}/apps/inference-worker/Cargo.toml\"}}]}}"
    );
    let failure = model_loading_paths(&metadata_json_text, Path::new(REPOSITORY_ROOT)).unwrap_err();
    assert!(
        failure.contains("reason=package-without-dependencies"),
        "{failure}"
    );
}

#[test]
fn should_fail_closed_on_a_local_dependency_outside_the_workspace() {
    let metadata_json_text =
        metadata_document_for(&[worker_package(vec![("vendored-sibling", None, true)])]);
    let failure = model_loading_paths(&metadata_json_text, Path::new(REPOSITORY_ROOT)).unwrap_err();
    assert!(
        failure.contains("reason=local-dependency-not-in-workspace package=vendored-sibling"),
        "{failure}"
    );
}

#[test]
fn should_fail_closed_on_a_dependency_kind_that_is_not_text() {
    let metadata_json_text = format!(
        "{{\"packages\":[{{\"name\":\"astronomical-inference-worker\",\"manifest_path\":\"{REPOSITORY_ROOT}/apps/inference-worker/Cargo.toml\",\"dependencies\":[{{\"name\":\"qwen35\",\"kind\":5,\"path\":\"{REPOSITORY_ROOT}/crates/qwen35\"}}]}}]}}"
    );
    let failure = model_loading_paths(&metadata_json_text, Path::new(REPOSITORY_ROOT)).unwrap_err();
    assert!(
        failure.contains("reason=unknown-dependency-kind kind=unreadable"),
        "{failure}"
    );
}

#[test]
fn should_follow_a_dependency_whose_kind_is_spelled_normal() {
    let metadata_json_text = metadata_document_for(&[
        worker_package(vec![("qwen35", Some("normal"), true)]),
        library_package("qwen35", Vec::new()),
    ]);
    let watched_paths =
        model_loading_paths(&metadata_json_text, Path::new(REPOSITORY_ROOT)).unwrap();
    assert!(watched_paths.contains(&"crates/families/text/qwen35".to_owned()));
}

#[test]
fn should_accept_exactly_one_previous_tag() {
    assert_eq!(parse_previous_tag(&["v0.2.118".to_owned()]), Ok("v0.2.118"));
}

#[test]
fn should_refuse_a_missing_previous_tag() {
    assert!(parse_previous_tag(&[]).is_err());
}

#[test]
fn should_refuse_more_than_one_previous_tag() {
    assert!(parse_previous_tag(&["v0.2.117".to_owned(), "v0.2.118".to_owned()]).is_err());
}

#[test]
fn should_refuse_a_flag_or_an_empty_text_as_the_previous_tag() {
    assert!(parse_previous_tag(&["--since".to_owned()]).is_err());
    assert!(parse_previous_tag(&[String::new()]).is_err());
}

#[test]
fn should_map_git_diff_quiet_exits_to_the_skills_flag_and_refuse_other_outcomes() {
    assert_eq!(unchanged_flag_from_git_exit_code(Some(0)), Ok(0));
    assert_eq!(unchanged_flag_from_git_exit_code(Some(1)), Ok(1));
    assert!(unchanged_flag_from_git_exit_code(Some(128)).is_err());
    assert!(unchanged_flag_from_git_exit_code(None).is_err());
}
