//! Unit tests for the metadata document. The cross-crate contract (that
//! model-serving's strict reader accepts it) lives in model-serving's
//! integration tests.

use super::*;

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
#[ignore = "pending implementation of describe_written_artifact"]
fn should_record_unwritten_planned_modules_as_bfloat16() {
    let mut metadata = OptiQArtifactMetadata {
        method: "optiq_mixed_precision".to_owned(),
        base_model: "example/base".to_owned(),
        reference: "bf16".to_owned(),
        target_bits_per_weight: 4.5,
        achieved_bits_per_weight: 4.6,
        module_count_at_highest_width: 1,
        module_count_at_lowest_width: 1,
        threshold: 0.0,
        per_layer: BTreeMap::from([
            (
                "model.layers.0.mlp.gate".to_owned(),
                PerLayerEntry {
                    bits: 4,
                    group_size: Some(64),
                },
            ),
            (
                "model.layers.0.mlp.up_proj".to_owned(),
                PerLayerEntry {
                    bits: 4,
                    group_size: Some(64),
                },
            ),
        ]),
        producer: "astronomical-quantization".to_owned(),
    };
    let written = BTreeMap::from([("model.layers.0.mlp.up_proj".to_owned(), (4_u32, 32_u32))]);

    metadata.describe_written_artifact(&written);

    // The router was passed through, the up projection dropped to group 32.
    assert_eq!(
        metadata.per_layer["model.layers.0.mlp.gate"],
        PerLayerEntry {
            bits: 16,
            group_size: None
        }
    );
    assert_eq!(
        metadata.per_layer["model.layers.0.mlp.up_proj"],
        PerLayerEntry {
            bits: 4,
            group_size: Some(32)
        }
    );
}
