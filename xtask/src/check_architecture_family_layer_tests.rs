use crate::check_architecture_family_layer::{
    FAMILY_WITHOUT_OTHER_FAMILY_RULE, FAMILY_WITHOUT_SERVING_RULE, family_layer_violations,
};
use crate::check_architecture_test_support::metadata_document_for;

/// A speech kind with its shell and two families, an image kind with its
/// shell and one family, and the serving crate above them all.
fn layered_workspace_packages() -> Vec<(&'static str, &'static str, Vec<&'static str>)> {
    vec![
        (
            "speech",
            "crates/families/speech/speech",
            vec!["mlx-bridge"],
        ),
        ("kokoro", "crates/families/speech/kokoro", vec!["speech"]),
        ("phonon", "crates/families/speech/phonon", vec!["speech"]),
        (
            "image-generation",
            "crates/families/image/image-generation",
            vec!["mlx-bridge"],
        ),
        (
            "flux2-klein",
            "crates/families/image/flux2-klein",
            vec!["image-generation"],
        ),
        ("mlx-bridge", "crates/foundation/mlx-bridge", Vec::new()),
        (
            "model-serving",
            "crates/engine/model-serving",
            vec!["speech", "kokoro", "phonon", "flux2-klein"],
        ),
    ]
}

fn with_dependencies(
    mut packages: Vec<(&'static str, &'static str, Vec<&'static str>)>,
    package_name: &str,
    dependency_names: Vec<&'static str>,
) -> Vec<(&'static str, &'static str, Vec<&'static str>)> {
    if let Some(package) = packages
        .iter_mut()
        .find(|package| package.0 == package_name)
    {
        package.2 = dependency_names;
    }
    packages
}

#[test]
fn should_pass_a_layout_where_families_reach_only_their_shell() {
    let metadata_json_text = metadata_document_for(&layered_workspace_packages());
    let violations = family_layer_violations(&metadata_json_text).unwrap();
    assert!(violations.is_empty(), "{violations:?}");
}

#[test]
fn should_detect_one_family_depending_on_another_of_its_kind() {
    let packages = with_dependencies(
        layered_workspace_packages(),
        "kokoro",
        vec!["speech", "phonon"],
    );
    let violations = family_layer_violations(&metadata_document_for(&packages)).unwrap();
    assert_eq!(violations.len(), 1, "{violations:?}");
    assert_eq!(violations[0].rule_name, FAMILY_WITHOUT_OTHER_FAMILY_RULE);
    assert_eq!(violations[0].dependency_chain, vec!["kokoro", "phonon"]);
}

#[test]
fn should_detect_a_family_crate_reaching_the_serving_crate() {
    let packages = with_dependencies(
        layered_workspace_packages(),
        "flux2-klein",
        vec!["image-generation", "model-serving"],
    );
    let violations = family_layer_violations(&metadata_document_for(&packages)).unwrap();
    assert_eq!(violations.len(), 1, "{violations:?}");
    assert_eq!(violations[0].rule_name, FAMILY_WITHOUT_SERVING_RULE);
    assert_eq!(
        violations[0].dependency_chain,
        vec!["flux2-klein", "model-serving"]
    );
}

#[test]
fn should_detect_a_shell_crate_reaching_the_serving_crate_through_another_crate() {
    let mut packages = with_dependencies(
        layered_workspace_packages(),
        "speech",
        vec!["mlx-bridge", "relay"],
    );
    packages.push(("relay", "crates/engine/relay", vec!["model-serving"]));
    let violations = family_layer_violations(&metadata_document_for(&packages)).unwrap();
    // The shell and both families above it reach the serving crate through
    // the relay, each reported with its own chain. Because the serving crate
    // depends on both families, that edge also closes a loop through which
    // each family reaches the other, so both rules fire.
    let serving_chains: Vec<&Vec<String>> = violations
        .iter()
        .filter(|violation| violation.rule_name == FAMILY_WITHOUT_SERVING_RULE)
        .map(|violation| &violation.dependency_chain)
        .collect();
    assert_eq!(
        serving_chains,
        vec![
            &vec!["kokoro", "speech", "relay", "model-serving"],
            &vec!["phonon", "speech", "relay", "model-serving"],
            &vec!["speech", "relay", "model-serving"],
        ],
        "{violations:?}"
    );
    assert_eq!(violations.len(), 5, "{violations:?}");
}

#[test]
fn should_allow_families_of_different_kinds_to_share_a_foundation_crate() {
    // Two kinds never form one family group, so a family of one kind next
    // to a family of another is no violation.
    let packages = with_dependencies(
        layered_workspace_packages(),
        "kokoro",
        vec!["speech", "mlx-bridge"],
    );
    let violations = family_layer_violations(&metadata_document_for(&packages)).unwrap();
    assert!(violations.is_empty(), "{violations:?}");
}

#[test]
fn should_ignore_a_manifest_nested_deeper_than_a_crate_folder() {
    let mut packages = layered_workspace_packages();
    packages.push((
        "speech-fixture",
        "crates/families/speech/speech/fixtures",
        vec!["model-serving"],
    ));
    let violations = family_layer_violations(&metadata_document_for(&packages)).unwrap();
    assert!(violations.is_empty(), "{violations:?}");
}

#[test]
fn should_fail_closed_when_no_crate_sits_under_the_families_folder() {
    let metadata_json_text = metadata_document_for(&[
        ("mlx-bridge", "crates/foundation/mlx-bridge", Vec::new()),
        ("model-serving", "crates/engine/model-serving", Vec::new()),
    ]);
    let error = family_layer_violations(&metadata_json_text).unwrap_err();
    assert!(error.contains("no crate sits under"), "{error}");
}

#[test]
fn should_fail_closed_when_the_serving_crate_is_absent() {
    let packages: Vec<_> = layered_workspace_packages()
        .into_iter()
        .filter(|package| package.0 != "model-serving")
        .collect();
    let error = family_layer_violations(&metadata_document_for(&packages)).unwrap_err();
    assert!(error.contains("unknown crate model-serving"), "{error}");
}

#[test]
fn should_fail_closed_when_a_package_has_no_manifest_path() {
    let metadata_json_text =
        "{\"packages\":[{\"name\":\"speech\",\"dependencies\":[]}]}".to_owned();
    let error = family_layer_violations(&metadata_json_text).unwrap_err();
    assert!(error.contains("has no manifest_path"), "{error}");
}

#[test]
fn should_treat_a_single_crate_kind_as_a_shell() {
    // The text kind holds its families in one crate today; with no sibling
    // to depend on, that crate is a shell and only the serving rule applies.
    let mut packages = layered_workspace_packages();
    packages.push(("qwen35", "crates/families/text/qwen35", vec!["mlx-bridge"]));
    let violations = family_layer_violations(&metadata_document_for(&packages)).unwrap();
    assert!(violations.is_empty(), "{violations:?}");
}
