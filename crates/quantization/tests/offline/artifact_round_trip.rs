//! The artifact writers produce documents this crate's readers parse back
//! from disk, in the exact file locations OptiQ and Astronomical expect.

use std::collections::BTreeMap;

use astronomical_quantization::artifact::{
    METADATA_FILE_NAME, OptiQArtifactMetadata, OptiQSensitivityDocument,
    QuantizationConfigDocument, SENSITIVITY_FILE_NAME, artifact_path, resolve_metadata_path,
};
use astronomical_quantization::bit_width::{BitWidth, GroupSize};
use astronomical_quantization::sensitivity::{
    SensitivityMethod, SensitivityResult, SensitivityTable,
};

#[test]
#[ignore = "pending implementation of the artifact writers"]
fn should_write_config_metadata_and_sensitivity_where_optiq_and_astronomical_read_them() {
    let source = tempfile::tempdir().expect("a temporary directory should be creatable");
    let output = tempfile::tempdir().expect("a temporary directory should be creatable");
    std::fs::write(
        source.path().join("config.json"),
        r#"{"model_type": "qwen3_5", "architectures": ["Qwen3_5ForCausalLM"]}"#,
    )
    .expect("the fixture config should be writable");
    let written = BTreeMap::from([("model.layers.0.mlp.up_proj".to_owned(), (8_u32, 64_u32))]);
    let table = SensitivityTable {
        method: SensitivityMethod::ExactBfloat16Reference,
        results: vec![SensitivityResult {
            module_name: "model.layers.0.mlp.up_proj".to_owned(),
            kl_divergence_by_bit_width: BTreeMap::from([
                (BitWidth::new(4).expect("valid"), 0.01),
                (BitWidth::new(8).expect("valid"), 0.001),
            ]),
            logical_parameter_count: 4096,
        }],
    };

    QuantizationConfigDocument::from_written_modules(64, 4, &written)
        .write_into_config(
            &source.path().join("config.json"),
            &output.path().join("config.json"),
        )
        .expect("the config should be written");
    let mut metadata = OptiQArtifactMetadata {
        method: "optiq_mixed_precision".to_owned(),
        base_model: "example/base".to_owned(),
        reference: "bf16".to_owned(),
        target_bits_per_weight: 4.5,
        achieved_bits_per_weight: 8.0,
        module_count_at_highest_width: 1,
        module_count_at_lowest_width: 0,
        threshold: 0.0,
        per_layer: BTreeMap::new(),
        producer: "astronomical-quantization".to_owned(),
    };
    metadata.describe_written_artifact(&written);
    metadata
        .write(output.path())
        .expect("the metadata should be written");
    OptiQSensitivityDocument::from_table(
        &table,
        "example/base",
        &[4, 8],
        GroupSize::new(64).expect("valid"),
    )
    .write(output.path())
    .expect("the sensitivity document should be written");

    assert_eq!(
        resolve_metadata_path(output.path()),
        Some(artifact_path(output.path(), METADATA_FILE_NAME))
    );
    let reread_sensitivity =
        OptiQSensitivityDocument::read(&artifact_path(output.path(), SENSITIVITY_FILE_NAME))
            .expect("the sensitivity document should read back");
    assert_eq!(
        reread_sensitivity.to_table().expect("translates").results,
        table.results
    );
    let reread_config =
        QuantizationConfigDocument::read_from_config(&output.path().join("config.json"))
            .expect("the config should read back")
            .expect("the quantization document is present");
    assert_eq!(
        reread_config.source_parameters_for("model.layers.0.mlp.up_proj"),
        (8, 64)
    );
}
