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
        ("memory-policy", vec!["serde_json"], "normal"),
        ("model-discovery", vec!["serde_json"], "normal"),
        ("mlx-bridge", Vec::new(), "normal"),
        ("mlx-c-fork", Vec::new(), "normal"),
        ("image-generation", vec!["mlx-bridge"], "normal"),
        ("artifact-validation", Vec::new(), "normal"),
        ("engine-core", vec!["mlx-bridge"], "normal"),
        ("qwen35", vec!["mlx-bridge"], "normal"),
        ("decision-models", vec!["mlx-bridge"], "normal"),
        ("quantization", Vec::new(), "normal"),
        ("speech", vec!["mlx-bridge"], "normal"),
        ("model-primitives", vec!["mlx-bridge"], "normal"),
        ("model-serving", vec!["mlx-bridge"], "normal"),
        // The jangtq2 family crate clean of the serving crate: the one-way
        // edge the family-crate rule exists to hold.
        ("jangtq2-qwen35", vec!["mlx-bridge"], "normal"),
    ]);
    let violations = architecture_violations(&metadata_json_text).unwrap();
    assert!(violations.is_empty());
}

#[test]
fn should_detect_a_direct_family_to_serving_dependency() {
    // The family crate depending back on the serving crate would rebuild the
    // serving crate on every family edit, defeating the extraction.
    let metadata_json_text = metadata_document_for(&[
        ("memory-policy", vec!["serde_json"], "normal"),
        ("model-discovery", vec!["serde_json"], "normal"),
        ("mlx-bridge", Vec::new(), "normal"),
        ("mlx-c-fork", Vec::new(), "normal"),
        ("image-generation", vec!["mlx-bridge"], "normal"),
        ("artifact-validation", Vec::new(), "normal"),
        ("engine-core", vec!["mlx-bridge"], "normal"),
        ("qwen35", vec!["mlx-bridge"], "normal"),
        ("decision-models", vec!["mlx-bridge"], "normal"),
        ("quantization", Vec::new(), "normal"),
        ("speech", vec!["mlx-bridge"], "normal"),
        ("model-primitives", vec!["mlx-bridge"], "normal"),
        (
            "model-serving",
            vec!["mlx-bridge", "jangtq2-qwen35"],
            "normal",
        ),
        ("jangtq2-qwen35", vec!["model-serving"], "normal"),
    ]);
    let violations = architecture_violations(&metadata_json_text).unwrap();
    assert_eq!(violations.len(), 1);
    assert_eq!(violations[0].rule_name, "weight-format-without-serving");
    assert_eq!(
        violations[0].dependency_chain,
        vec!["jangtq2-qwen35", "model-serving"]
    );
}

#[test]
fn should_detect_a_model_primitives_to_serving_dependency() {
    // The model primitives sit under every family; an edge back to the
    // serving crate would let a shared kernel reach into one family.
    let metadata_json_text = metadata_document_for(&[
        ("memory-policy", vec!["serde_json"], "normal"),
        ("model-discovery", vec!["serde_json"], "normal"),
        ("mlx-bridge", Vec::new(), "normal"),
        ("mlx-c-fork", Vec::new(), "normal"),
        ("image-generation", vec!["mlx-bridge"], "normal"),
        ("artifact-validation", Vec::new(), "normal"),
        ("engine-core", vec!["mlx-bridge"], "normal"),
        ("qwen35", vec!["mlx-bridge"], "normal"),
        ("decision-models", vec!["mlx-bridge"], "normal"),
        ("quantization", Vec::new(), "normal"),
        ("speech", vec!["mlx-bridge"], "normal"),
        (
            "model-primitives",
            vec!["mlx-bridge", "model-serving"],
            "normal",
        ),
        ("model-serving", vec!["model-primitives"], "normal"),
        ("jangtq2-qwen35", vec!["mlx-bridge"], "normal"),
    ]);
    let violations = architecture_violations(&metadata_json_text).unwrap();
    assert_eq!(violations.len(), 1);
    assert_eq!(violations[0].rule_name, "model-primitives-without-serving");
    assert_eq!(
        violations[0].dependency_chain,
        vec!["model-primitives", "model-serving"]
    );
}

#[test]
fn should_detect_a_artifact_validation_to_serving_dependency() {
    // Every family validates through this crate; an edge to the serving
    // crate would make each family depend on the engine.
    let metadata_json_text = metadata_document_for(&[
        ("memory-policy", vec!["serde_json"], "normal"),
        ("model-discovery", vec!["serde_json"], "normal"),
        ("mlx-bridge", Vec::new(), "normal"),
        ("mlx-c-fork", Vec::new(), "normal"),
        ("model-primitives", vec!["mlx-bridge"], "normal"),
        ("speech", vec!["mlx-bridge"], "normal"),
        ("image-generation", vec!["mlx-bridge"], "normal"),
        ("artifact-validation", vec!["model-serving"], "normal"),
        ("jangtq2-qwen35", vec!["mlx-bridge"], "normal"),
        ("engine-core", vec!["mlx-bridge"], "normal"),
        ("qwen35", vec!["mlx-bridge"], "normal"),
        ("decision-models", vec!["mlx-bridge"], "normal"),
        ("quantization", Vec::new(), "normal"),
        ("model-serving", vec!["artifact-validation"], "normal"),
    ]);
    let violations = architecture_violations(&metadata_json_text).unwrap();
    assert_eq!(violations.len(), 1);
    assert_eq!(
        violations[0].rule_name,
        "artifact-validation-without-serving"
    );
    assert_eq!(
        violations[0].dependency_chain,
        vec!["artifact-validation", "model-serving"]
    );
}

/// Every rule's crates, sound except for the engine core's own dependencies.
fn workspace_metadata_with_engine_core_dependencies(engine_core_dependencies: Vec<&str>) -> String {
    metadata_document_for(&[
        ("memory-policy", vec!["serde_json"], "normal"),
        ("model-discovery", vec!["serde_json"], "normal"),
        ("mlx-bridge", Vec::new(), "normal"),
        ("mlx-c-fork", Vec::new(), "normal"),
        ("model-primitives", vec!["mlx-bridge"], "normal"),
        ("speech", vec!["mlx-bridge"], "normal"),
        ("image-generation", vec!["mlx-bridge"], "normal"),
        ("artifact-validation", Vec::new(), "normal"),
        ("jangtq2-qwen35", vec!["mlx-bridge"], "normal"),
        ("qwen35", vec!["mlx-bridge"], "normal"),
        ("decision-models", vec!["mlx-bridge"], "normal"),
        ("quantization", Vec::new(), "normal"),
        ("engine-core", engine_core_dependencies, "normal"),
        ("model-serving", vec!["engine-core"], "normal"),
    ])
}

#[test]
fn should_detect_an_engine_core_to_serving_dependency() {
    // Every family implements the engine core's contract; an edge back to
    // the serving crate would make the split circular.
    let metadata_json_text =
        workspace_metadata_with_engine_core_dependencies(vec!["model-serving"]);
    let violations = architecture_violations(&metadata_json_text).unwrap();
    assert_eq!(violations.len(), 1);
    assert_eq!(violations[0].rule_name, "engine-core-without-serving");
    assert_eq!(
        violations[0].dependency_chain,
        vec!["engine-core", "model-serving"]
    );
}

#[test]
fn should_detect_an_engine_core_to_qwen35_family_dependency() {
    // The engine core holds family-neutral machinery; an edge to the Qwen
    // 3.5 family would let it name a family type again.
    let metadata_json_text = workspace_metadata_with_engine_core_dependencies(vec!["qwen35"]);
    let violations = architecture_violations(&metadata_json_text).unwrap();
    assert_eq!(violations.len(), 1);
    assert_eq!(violations[0].rule_name, "engine-core-without-qwen35-family");
    assert_eq!(
        violations[0].dependency_chain,
        vec!["engine-core", "qwen35"]
    );
}

#[test]
fn should_detect_a_qwen35_to_quantizer_dependency() {
    // The worker and daemon build on the Qwen 3.5 family; the quantizer must
    // reach it only through the probe crate above it.
    let metadata_json_text = metadata_document_for(&[
        ("memory-policy", vec!["serde_json"], "normal"),
        ("model-discovery", vec!["serde_json"], "normal"),
        ("mlx-bridge", Vec::new(), "normal"),
        ("mlx-c-fork", Vec::new(), "normal"),
        ("model-primitives", vec!["mlx-bridge"], "normal"),
        ("speech", vec!["mlx-bridge"], "normal"),
        ("image-generation", vec!["mlx-bridge"], "normal"),
        ("artifact-validation", Vec::new(), "normal"),
        ("engine-core", vec!["mlx-bridge"], "normal"),
        ("decision-models", vec!["mlx-bridge"], "normal"),
        ("qwen35", vec!["mlx-bridge", "quantization"], "normal"),
        ("quantization", Vec::new(), "normal"),
        ("model-serving", Vec::new(), "normal"),
        ("jangtq2-qwen35", vec!["mlx-bridge"], "normal"),
    ]);
    let violations = architecture_violations(&metadata_json_text).unwrap();
    assert_eq!(violations.len(), 1);
    assert_eq!(violations[0].rule_name, "qwen35-without-quantizer");
    assert_eq!(
        violations[0].dependency_chain,
        vec!["qwen35", "quantization"]
    );
}

#[test]
fn should_detect_a_direct_forbidden_fork_dependency() {
    // The fork crate carries the same graphics-processor linkage as the
    // stock bridge, so the memory policy crate must stay free of it too.
    let metadata_json_text = metadata_document_for(&[
        ("memory-policy", vec!["mlx-c-fork"], "normal"),
        ("mlx-bridge", Vec::new(), "normal"),
        ("model-discovery", vec!["serde_json"], "normal"),
        ("mlx-c-fork", Vec::new(), "normal"),
        ("image-generation", vec!["mlx-bridge"], "normal"),
        ("artifact-validation", Vec::new(), "normal"),
        ("engine-core", vec!["mlx-bridge"], "normal"),
        ("qwen35", vec!["mlx-bridge"], "normal"),
        ("decision-models", vec!["mlx-bridge"], "normal"),
        ("quantization", Vec::new(), "normal"),
        ("speech", vec!["mlx-bridge"], "normal"),
        ("model-primitives", vec!["mlx-bridge"], "normal"),
        ("model-serving", Vec::new(), "normal"),
        ("jangtq2-qwen35", Vec::new(), "normal"),
    ]);
    let violations = architecture_violations(&metadata_json_text).unwrap();
    assert_eq!(violations.len(), 1);
    assert_eq!(violations[0].rule_name, "policy-without-mlx-fork-bindings");
    assert_eq!(
        violations[0].dependency_chain,
        vec!["memory-policy", "mlx-c-fork"]
    );
}

#[test]
fn should_detect_a_direct_forbidden_dependency() {
    // Detector self-check carried over from the shell contract: a known MLX
    // declaration must be caught so the check can never pass vacuously.
    let metadata_json_text = metadata_document_for(&[
        ("memory-policy", vec!["mlx-bridge"], "normal"),
        ("mlx-bridge", Vec::new(), "normal"),
        ("model-discovery", vec!["serde_json"], "normal"),
        ("mlx-c-fork", Vec::new(), "normal"),
        ("image-generation", vec!["mlx-bridge"], "normal"),
        ("artifact-validation", Vec::new(), "normal"),
        ("engine-core", vec!["mlx-bridge"], "normal"),
        ("qwen35", vec!["mlx-bridge"], "normal"),
        ("decision-models", vec!["mlx-bridge"], "normal"),
        ("quantization", Vec::new(), "normal"),
        ("speech", vec!["mlx-bridge"], "normal"),
        ("model-primitives", vec!["mlx-bridge"], "normal"),
        ("model-serving", Vec::new(), "normal"),
        ("jangtq2-qwen35", Vec::new(), "normal"),
    ]);
    let violations = architecture_violations(&metadata_json_text).unwrap();
    assert_eq!(violations.len(), 1);
    assert_eq!(
        violations[0].dependency_chain,
        vec!["memory-policy", "mlx-bridge"]
    );
}

#[test]
fn should_detect_a_transitive_forbidden_dependency() {
    // The capability the manifest grep this replaces could never have: the
    // breach arrives through an intermediate crate, not a direct line.
    let metadata_json_text = metadata_document_for(&[
        (
            "memory-policy",
            vec!["astronomical-new-intermediate"],
            "normal",
        ),
        (
            "astronomical-new-intermediate",
            vec!["mlx-bridge"],
            "normal",
        ),
        ("mlx-bridge", Vec::new(), "normal"),
        ("model-discovery", vec!["serde_json"], "normal"),
        ("mlx-c-fork", Vec::new(), "normal"),
        ("image-generation", vec!["mlx-bridge"], "normal"),
        ("artifact-validation", Vec::new(), "normal"),
        ("engine-core", vec!["mlx-bridge"], "normal"),
        ("qwen35", vec!["mlx-bridge"], "normal"),
        ("decision-models", vec!["mlx-bridge"], "normal"),
        ("quantization", Vec::new(), "normal"),
        ("speech", vec!["mlx-bridge"], "normal"),
        ("model-primitives", vec!["mlx-bridge"], "normal"),
        ("model-serving", Vec::new(), "normal"),
        ("jangtq2-qwen35", Vec::new(), "normal"),
    ]);
    let violations = architecture_violations(&metadata_json_text).unwrap();
    assert_eq!(violations.len(), 1);
    assert_eq!(
        violations[0].dependency_chain,
        vec![
            "memory-policy",
            "astronomical-new-intermediate",
            "mlx-bridge"
        ]
    );
}

#[test]
fn should_count_dev_and_build_dependencies_as_reachability_edges() {
    for breach_kind in ["dev", "build"] {
        let metadata_json_text = metadata_document_for(&[
            ("memory-policy", vec!["mlx-bridge"], breach_kind),
            ("mlx-bridge", Vec::new(), "normal"),
            ("model-discovery", vec!["serde_json"], "normal"),
            ("mlx-c-fork", Vec::new(), "normal"),
            ("image-generation", vec!["mlx-bridge"], "normal"),
            ("artifact-validation", Vec::new(), "normal"),
            ("engine-core", vec!["mlx-bridge"], "normal"),
            ("qwen35", vec!["mlx-bridge"], "normal"),
            ("decision-models", vec!["mlx-bridge"], "normal"),
            ("quantization", Vec::new(), "normal"),
            ("speech", vec!["mlx-bridge"], "normal"),
            ("model-primitives", vec!["mlx-bridge"], "normal"),
            ("model-serving", Vec::new(), "normal"),
            ("jangtq2-qwen35", Vec::new(), "normal"),
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
            .contains("unknown source crate memory-policy")
    );
}

#[test]
fn should_fail_closed_when_the_forbidden_crate_is_absent() {
    // A renamed forbidden crate would make the rule pass vacuously forever;
    // rename drift must surface as a hard error instead.
    let metadata_json_text = metadata_document_for(&[("memory-policy", Vec::new(), "normal")]);
    let check_outcome = architecture_violations(&metadata_json_text);
    assert!(check_outcome.is_err());
    assert!(
        check_outcome
            .unwrap_err()
            .contains("unknown forbidden crate mlx-bridge")
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

#[test]
fn should_detect_model_discovery_reaching_the_mlx_bindings_through_another_crate() {
    // The daemon discovers models and never links MLX, so discovery reaching
    // the bindings through any crate fails, with the whole chain named.
    let metadata_json_text = metadata_document_for(&[
        ("memory-policy", vec!["serde_json"], "normal"),
        ("mlx-bridge", Vec::new(), "normal"),
        ("model-discovery", vec!["expert-atlas"], "normal"),
        ("expert-atlas", vec!["mlx-bridge"], "normal"),
        ("mlx-c-fork", Vec::new(), "normal"),
        ("image-generation", vec!["mlx-bridge"], "normal"),
        ("artifact-validation", Vec::new(), "normal"),
        ("engine-core", vec!["mlx-bridge"], "normal"),
        ("qwen35", vec!["mlx-bridge"], "normal"),
        ("decision-models", vec!["mlx-bridge"], "normal"),
        ("quantization", Vec::new(), "normal"),
        ("speech", vec!["mlx-bridge"], "normal"),
        ("model-primitives", vec!["mlx-bridge"], "normal"),
        ("model-serving", Vec::new(), "normal"),
        ("jangtq2-qwen35", Vec::new(), "normal"),
    ]);
    let violations = architecture_violations(&metadata_json_text).unwrap();
    assert_eq!(violations.len(), 1);
    assert_eq!(
        violations[0].rule_name,
        "model-discovery-without-mlx-bindings"
    );
    assert_eq!(
        violations[0].dependency_chain,
        vec!["model-discovery", "expert-atlas", "mlx-bridge"]
    );
}
