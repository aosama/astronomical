//! Unit tests for the metadata document. The cross-crate contract (that
//! model-serving's strict reader accepts it) lives in model-serving's
//! integration tests.

use super::*;

fn written(entries: &[(&str, u32, u32)]) -> BTreeMap<String, (u32, u32)> {
    let mut written_quantization_by_module = BTreeMap::new();
    for (module_name, bits, group_size) in entries {
        written_quantization_by_module.insert((*module_name).to_owned(), (*bits, *group_size));
    }
    written_quantization_by_module
}

#[test]
fn should_name_methods_and_references_the_way_optiq_does() {
    assert_eq!(
        OptiQArtifactMetadata::method_name(SensitivityMethod::ExactBfloat16Reference),
        "optiq_mixed_precision"
    );
    assert_eq!(
        OptiQArtifactMetadata::method_name(SensitivityMethod::StaticStructuralRules),
        "static_mixed_precision"
    );
    assert_eq!(
        OptiQArtifactMetadata::reference_name(SensitivityMethod::ExactUniform4BitReference),
        "uniform_4bit"
    );
}

#[test]
fn should_list_only_the_written_modules_at_their_written_parameters() {
    let mut metadata = OptiQArtifactMetadata::for_method(
        SensitivityMethod::ExactBfloat16Reference,
        "example/base",
        64,
    );
    metadata.describe_written_artifact(&written(&[("model.layers.0.mlp.gate", 4, 64)]));

    // The router was passed through and the up projection dropped to group 32.
    metadata.describe_written_artifact(&written(&[("model.layers.0.mlp.up_proj", 4, 32)]));

    assert_eq!(metadata.per_layer().len(), 1);
    assert_eq!(
        metadata.per_layer()["model.layers.0.mlp.up_proj"],
        PerLayerEntry {
            bits: 4,
            group_size: 32
        }
    );
}

#[test]
fn should_summarize_the_mixed_layout_the_way_published_quants_do() {
    let mut metadata = OptiQArtifactMetadata::for_method(
        SensitivityMethod::ExactBfloat16Reference,
        "example/base",
        64,
    );

    metadata.describe_written_artifact(&written(&[
        ("model.embed_tokens", 8, 64),
        ("model.layers.0.mlp.up_proj", 4, 64),
        ("model.layers.1.mlp.up_proj", 4, 64),
    ]));

    let serialized = serde_json::to_value(&metadata).expect("the document should serialize");
    assert_eq!(serialized["method"], "optiq_mixed_precision");
    assert_eq!(serialized["default_bits"], 4);
    assert_eq!(serialized["bit_widths_used"], serde_json::json!([4, 8]));
    assert_eq!(serialized["layers_per_bit_width"]["4"], 2);
    assert_eq!(serialized["n_quantized_layers"], 3);
    assert!(serialized.get("reference").is_none());
    assert!(serialized.get("recipe_note").is_none());
}

#[test]
fn should_write_the_static_layout_for_the_structural_method() {
    let metadata = OptiQArtifactMetadata::for_method(
        SensitivityMethod::StaticStructuralRules,
        "example/base",
        64,
    );

    let serialized = serde_json::to_value(&metadata).expect("the document should serialize");

    assert_eq!(serialized["method"], "static_mixed_precision");
    assert_eq!(serialized["reference"], "structural_rules");
    assert_eq!(serialized["threshold"], 0.0);
    assert!(serialized.get("default_bits").is_none());
}

#[test]
fn should_read_back_a_published_mixed_document_and_refuse_unknown_keys() {
    let published = r#"{"method": "optiq_mixed_precision", "base_model": "b", "default_bits": 4,
        "group_size": 64, "bit_widths_used": [4], "layers_per_bit_width": {"4": 1},
        "n_quantized_layers": 1, "recipe_note": "reused",
        "per_layer": {"m": {"bits": 4, "group_size": 64}}}"#;
    let with_unknown_key = r#"{"method": "static_mixed_precision", "base_model": "b",
        "reference": "bf16", "target_bpw": 4.0, "achieved_bpw": 4.0, "n_high_bits": 0,
        "n_low_bits": 0, "threshold": 0.0, "per_layer": {}, "producer": "x"}"#;

    let parsed: OptiQArtifactMetadata =
        serde_json::from_str(published).expect("a published document should parse");

    assert_eq!(parsed.per_layer().len(), 1);
    assert!(serde_json::from_str::<OptiQArtifactMetadata>(with_unknown_key).is_err());
}

#[test]
fn should_leave_keep_bfloat16_decisions_out_of_the_planned_document() {
    let allocation = AllocationResult {
        decisions: vec![
            crate::allocation::ModuleQuantizationDecision {
                module_name: "lm_head".to_owned(),
                bit_width: crate::bit_width::BitWidth::new(16).expect("valid"),
                group_size: crate::bit_width::GroupSize::new(64).expect("valid"),
                logical_parameter_count: 10,
            },
            crate::allocation::ModuleQuantizationDecision {
                module_name: "model.layers.0.mlp.up_proj".to_owned(),
                bit_width: crate::bit_width::BitWidth::new(4).expect("valid"),
                group_size: crate::bit_width::GroupSize::new(64).expect("valid"),
                logical_parameter_count: 10,
            },
        ],
        target_bits_per_weight: 4.0,
        achieved_bits_per_weight: 10.0,
        module_count_at_highest_width: 1,
        module_count_at_lowest_width: 1,
        total_logical_parameter_count: 20,
        estimated_weight_bytes: 25,
        floor_upgrade_count: 0,
        run_guard_upgrade_count: 0,
        non_finite_module_names: Vec::new(),
    };

    let metadata = OptiQArtifactMetadata::from_allocation(
        &allocation,
        "example/base",
        SensitivityMethod::StaticStructuralRules,
    );

    assert_eq!(metadata.per_layer().len(), 1);
    let OptiQArtifactMetadata::StaticAllocation(static_metadata) = metadata else {
        panic!("the structural method writes the static layout");
    };
    assert_eq!(static_metadata.achieved_bpw, 10.0);
    assert_eq!(static_metadata.n_high_bits, 1);
}
