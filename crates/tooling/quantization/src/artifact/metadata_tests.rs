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
fn should_list_only_the_written_modules_when_there_is_no_plan() {
    let mut metadata = OptiQArtifactMetadata::for_method(
        SensitivityMethod::ExactBfloat16Reference,
        "example/base",
    );

    // The up projection dropped to group 32 on the way out.
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
fn should_write_one_layout_for_measured_and_structural_methods() {
    let measured = OptiQArtifactMetadata::for_method(
        SensitivityMethod::ExactBfloat16Reference,
        "example/base",
    );
    let structural =
        OptiQArtifactMetadata::for_method(SensitivityMethod::StaticStructuralRules, "example/base");

    let measured_document = serde_json::to_value(&measured).expect("the document serializes");
    let structural_document = serde_json::to_value(&structural).expect("the document serializes");

    assert_eq!(measured_document["method"], "optiq_mixed_precision");
    assert_eq!(measured_document["reference"], "bf16");
    assert_eq!(measured_document["threshold"], 0.0);
    assert!(measured_document.get("default_bits").is_none());
    assert_eq!(structural_document["method"], "static_mixed_precision");
    assert_eq!(structural_document["reference"], "structural_rules");
}

#[test]
fn should_resummarize_a_published_summary_layout_document() {
    let published = r#"{"method": "optiq_mixed_precision", "base_model": "b", "default_bits": 4,
        "group_size": 64, "bit_widths_used": [4], "layers_per_bit_width": {"4": 1},
        "n_quantized_layers": 1, "per_layer": {"m": {"bits": 4, "group_size": 64}}}"#;
    let mut metadata: OptiQArtifactMetadata =
        serde_json::from_str(published).expect("a published document should parse");

    metadata.describe_written_artifact(&written(&[
        ("model.embed_tokens", 8, 64),
        ("model.layers.0.mlp.up_proj", 4, 64),
        ("model.layers.1.mlp.up_proj", 4, 64),
    ]));

    let serialized = serde_json::to_value(&metadata).expect("the document should serialize");
    assert_eq!(serialized["default_bits"], 4);
    assert_eq!(serialized["bit_widths_used"], serde_json::json!([4, 8]));
    assert_eq!(serialized["layers_per_bit_width"]["4"], 2);
    assert_eq!(serialized["n_quantized_layers"], 3);
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
fn should_record_the_plan_then_describe_what_was_written() {
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

    let mut metadata = OptiQArtifactMetadata::from_allocation(
        &allocation,
        "example/base",
        SensitivityMethod::StaticStructuralRules,
    );

    // The plan records the keep-bfloat16 tier at 16 bits, as OptiQ does.
    assert_eq!(metadata.per_layer().len(), 2);
    assert_eq!(metadata.per_layer()["lm_head"].bits, 16);

    // The up projection was written at group 32; the embedding table was
    // written too but was never planned, so it stays out of the map.
    metadata.describe_written_artifact(&written(&[
        ("model.layers.0.mlp.up_proj", 4, 32),
        ("model.embed_tokens", 8, 64),
    ]));

    assert_eq!(metadata.per_layer().len(), 2);
    assert_eq!(metadata.per_layer()["lm_head"].bits, 16);
    assert_eq!(
        metadata.per_layer()["model.layers.0.mlp.up_proj"],
        PerLayerEntry {
            bits: 4,
            group_size: 32
        }
    );
    let OptiQArtifactMetadata::StaticAllocation(static_metadata) = metadata else {
        panic!("the pipeline writes the static layout");
    };
    assert_eq!(static_metadata.achieved_bpw, 10.0);
    assert_eq!(static_metadata.n_high_bits, 1);
}
