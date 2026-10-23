use crate::check_architecture::architecture_violations;

/// Minimal `cargo metadata` document for a workspace whose packages are the
/// given `(name, dependency names, dependency kind)` tuples. All
/// dependencies of one package share the kind, which is enough for these
/// fixtures.
fn metadata_document_for(packages: &[(&str, Vec<&str>, &str)]) -> String {
    let package_json_entries: Vec<String> = packages
        .iter()
        .map(|(package_name, dependency_names, dependency_kind)| {
            let dependency_json_entries: Vec<String> = dependency_names
                .iter()
                .map(|dependency_name| {
                    format!("{{\"name\":\"{dependency_name}\",\"kind\":\"{dependency_kind}\"}}")
                })
                .collect();
            format!(
                "{{\"name\":\"{package_name}\",\"dependencies\":[{}]}}",
                dependency_json_entries.join(",")
            )
        })
        .collect();
    format!("{{\"packages\":[{}]}}", package_json_entries.join(","))
}

#[test]
fn should_pass_when_no_rule_crate_reaches_the_forbidden_crate() {
    let metadata_json_text = metadata_document_for(&[
        ("astronomical-memory-policy", vec!["serde_json"], "normal"),
        ("mlx-c-rust", Vec::new(), "normal"),
        ("astronomical-mlx-c-fork", Vec::new(), "normal"),
        ("astronomical-model-serving", vec!["mlx-c-rust"], "normal"),
        // The jangtq2 family crate clean of the serving crate: the one-way
        // edge the family-crate rule exists to hold.
        ("astronomical-jangtq2-qwen35", vec!["mlx-c-rust"], "normal"),
    ]);
    let violations = architecture_violations(&metadata_json_text).unwrap();
    assert!(violations.is_empty());
}

#[test]
fn should_detect_a_direct_family_to_serving_dependency() {
    // The family crate depending back on the serving crate would rebuild the
    // serving crate on every family edit, defeating the extraction.
    let metadata_json_text = metadata_document_for(&[
        ("astronomical-memory-policy", vec!["serde_json"], "normal"),
        ("mlx-c-rust", Vec::new(), "normal"),
        ("astronomical-mlx-c-fork", Vec::new(), "normal"),
        (
            "astronomical-model-serving",
            vec!["mlx-c-rust", "astronomical-jangtq2-qwen35"],
            "normal",
        ),
        (
            "astronomical-jangtq2-qwen35",
            vec!["astronomical-model-serving"],
            "normal",
        ),
    ]);
    let violations = architecture_violations(&metadata_json_text).unwrap();
    assert_eq!(violations.len(), 1);
    assert_eq!(violations[0].rule_name, "family-crate-without-serving");
    assert_eq!(
        violations[0].dependency_chain,
        vec!["astronomical-jangtq2-qwen35", "astronomical-model-serving"]
    );
}

#[test]
fn should_detect_a_direct_forbidden_dependency() {
    // Detector self-check carried over from the shell contract: a known MLX
    // declaration must be caught so the check can never pass vacuously.
    let metadata_json_text = metadata_document_for(&[
        ("astronomical-memory-policy", vec!["mlx-c-rust"], "normal"),
        ("mlx-c-rust", Vec::new(), "normal"),
        ("astronomical-mlx-c-fork", Vec::new(), "normal"),
        ("astronomical-model-serving", Vec::new(), "normal"),
        ("astronomical-jangtq2-qwen35", Vec::new(), "normal"),
    ]);
    let violations = architecture_violations(&metadata_json_text).unwrap();
    assert_eq!(violations.len(), 1);
    assert_eq!(
        violations[0].dependency_chain,
        vec!["astronomical-memory-policy", "mlx-c-rust"]
    );
}

#[test]
fn should_detect_a_transitive_forbidden_dependency() {
    // The capability the manifest grep this replaces could never have: the
    // breach arrives through an intermediate crate, not a direct line.
    let metadata_json_text = metadata_document_for(&[
        (
            "astronomical-memory-policy",
            vec!["astronomical-new-intermediate"],
            "normal",
        ),
        (
            "astronomical-new-intermediate",
            vec!["mlx-c-rust"],
            "normal",
        ),
        ("mlx-c-rust", Vec::new(), "normal"),
        ("astronomical-mlx-c-fork", Vec::new(), "normal"),
        ("astronomical-model-serving", Vec::new(), "normal"),
        ("astronomical-jangtq2-qwen35", Vec::new(), "normal"),
    ]);
    let violations = architecture_violations(&metadata_json_text).unwrap();
    assert_eq!(violations.len(), 1);
    assert_eq!(
        violations[0].dependency_chain,
        vec![
            "astronomical-memory-policy",
            "astronomical-new-intermediate",
            "mlx-c-rust"
        ]
    );
}

#[test]
fn should_count_dev_and_build_dependencies_as_reachability_edges() {
    for breach_kind in ["dev", "build"] {
        let metadata_json_text = metadata_document_for(&[
            (
                "astronomical-memory-policy",
                vec!["mlx-c-rust"],
                breach_kind,
            ),
            ("mlx-c-rust", Vec::new(), "normal"),
            ("astronomical-mlx-c-fork", Vec::new(), "normal"),
            ("astronomical-model-serving", Vec::new(), "normal"),
            ("astronomical-jangtq2-qwen35", Vec::new(), "normal"),
        ]);
        let violations = architecture_violations(&metadata_json_text).unwrap();
        assert_eq!(
            violations.len(),
            1,
            "a {breach_kind} dependency must count as a breach"
        );
    }
}

#[test]
fn should_fail_closed_when_a_rule_source_crate_is_absent() {
    let metadata_json_text =
        metadata_document_for(&[("astronomical-unrelated-crate", Vec::new(), "normal")]);
    let check_outcome = architecture_violations(&metadata_json_text);
    assert!(check_outcome.is_err());
    assert!(
        check_outcome
            .unwrap_err()
            .contains("unknown source crate astronomical-memory-policy")
    );
}

#[test]
fn should_fail_closed_when_the_forbidden_crate_is_absent() {
    // A renamed forbidden crate would make the rule pass vacuously forever;
    // rename drift must surface as a hard error instead.
    let metadata_json_text =
        metadata_document_for(&[("astronomical-memory-policy", Vec::new(), "normal")]);
    let check_outcome = architecture_violations(&metadata_json_text);
    assert!(check_outcome.is_err());
    assert!(
        check_outcome
            .unwrap_err()
            .contains("unknown forbidden crate mlx-c-rust")
    );
}

#[test]
fn should_fail_closed_when_the_metadata_is_malformed() {
    for malformed_json_text in ["", "not json", "{\"packages\": 4}"] {
        assert!(
            architecture_violations(malformed_json_text).is_err(),
            "malformed metadata must fail closed: {malformed_json_text}"
        );
    }
}
