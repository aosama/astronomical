use crate::check_architecture_recognition_crates::{
    RECOGNITION_WITHOUT_MLX_RULE, RECOGNITION_WITHOUT_SERVING_RULE, recognition_crate_violations,
};
use crate::check_architecture_test_support::metadata_document_for;

/// Two families' recognition crates on discovery, and the engine crates and
/// serving crate above them.
fn recognition_workspace_packages(
    kokoro_recognition_dependencies: Vec<&'static str>,
) -> Vec<(&'static str, &'static str, Vec<&'static str>)> {
    vec![
        (
            "model-discovery",
            "crates/foundation/model-discovery",
            vec!["serde_json"],
        ),
        (
            "kokoro-recognition",
            "crates/families/speech/kokoro/recognition",
            kokoro_recognition_dependencies,
        ),
        (
            "kev-recognition",
            "crates/families/decision/kev/recognition",
            vec!["model-discovery"],
        ),
        (
            "kokoro",
            "crates/families/speech/kokoro",
            vec!["mlx-bridge"],
        ),
        ("mlx-bridge", "crates/foundation/mlx-bridge", Vec::new()),
        (
            "model-serving",
            "crates/engine/model-serving",
            vec!["kokoro"],
        ),
    ]
}

#[test]
fn should_pass_recognition_crates_that_reach_only_discovery() {
    let packages = recognition_workspace_packages(vec!["model-discovery"]);
    let violations = recognition_crate_violations(&metadata_document_for(&packages)).unwrap();
    assert!(violations.is_empty(), "{violations:?}");
}

#[test]
fn should_detect_a_recognition_crate_reaching_the_mlx_bindings_through_its_engine() {
    let packages = recognition_workspace_packages(vec!["model-discovery", "kokoro"]);
    let violations = recognition_crate_violations(&metadata_document_for(&packages)).unwrap();
    assert_eq!(violations.len(), 1, "{violations:?}");
    assert_eq!(violations[0].rule_name, RECOGNITION_WITHOUT_MLX_RULE);
    assert_eq!(
        violations[0].dependency_chain,
        vec!["kokoro-recognition", "kokoro", "mlx-bridge"]
    );
}

#[test]
fn should_detect_a_recognition_crate_reaching_the_serving_crate() {
    let mut packages = recognition_workspace_packages(vec!["model-discovery", "model-serving"]);
    // The serving crate here depends on nothing, so only the serving rule fires.
    packages.retain(|package| package.0 != "model-serving");
    packages.push(("model-serving", "crates/engine/model-serving", Vec::new()));
    let violations = recognition_crate_violations(&metadata_document_for(&packages)).unwrap();
    assert_eq!(violations.len(), 1, "{violations:?}");
    assert_eq!(violations[0].rule_name, RECOGNITION_WITHOUT_SERVING_RULE);
}

#[test]
fn should_ignore_a_crate_named_recognition_outside_a_family_folder() {
    let mut packages = recognition_workspace_packages(vec!["model-discovery"]);
    packages.push((
        "stray-recognition",
        "crates/families/speech/recognition",
        vec!["mlx-bridge"],
    ));
    let violations = recognition_crate_violations(&metadata_document_for(&packages)).unwrap();
    assert!(violations.is_empty(), "{violations:?}");
}

#[test]
fn should_fail_closed_when_no_recognition_crate_exists() {
    let packages = vec![
        ("mlx-bridge", "crates/foundation/mlx-bridge", Vec::new()),
        ("model-serving", "crates/engine/model-serving", Vec::new()),
    ];
    let error = recognition_crate_violations(&metadata_document_for(&packages)).unwrap_err();
    assert!(error.contains("no crate sits at"), "{error}");
}

#[test]
fn should_fail_closed_when_the_mlx_bindings_are_absent() {
    let mut packages = recognition_workspace_packages(vec!["model-discovery"]);
    packages.retain(|package| package.0 != "mlx-bridge" && package.0 != "kokoro");
    let error = recognition_crate_violations(&metadata_document_for(&packages)).unwrap_err();
    assert!(error.contains("unknown crate mlx-bridge"), "{error}");
}
