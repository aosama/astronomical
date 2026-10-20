//! Unit tests for the config.json quantization document.

use super::*;

#[test]
#[ignore = "pending implementation of from_written_modules"]
fn should_serialize_defaults_and_flattened_module_entries() {
    let written = BTreeMap::from([
        ("model.layers.0.mlp.up_proj".to_owned(), (8_u32, 64_u32)),
        ("model.layers.1.mlp.up_proj".to_owned(), (4_u32, 64_u32)),
    ]);

    let document = QuantizationConfigDocument::from_written_modules(64, 4, &written);
    let serialized = serde_json::to_value(&document).expect("the document should serialize");

    assert_eq!(serialized["group_size"], 64);
    assert_eq!(serialized["bits"], 4);
    assert_eq!(serialized["mode"], "affine");
    assert_eq!(serialized["model.layers.0.mlp.up_proj"]["bits"], 8);
    assert_eq!(serialized["model.layers.0.mlp.up_proj"]["mode"], "affine");
}

#[test]
#[ignore = "pending implementation of write_into_config and read_from_config"]
fn should_preserve_every_other_config_key_when_writing() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    let source_config_path = directory.path().join("source-config.json");
    let output_config_path = directory.path().join("config.json");
    std::fs::write(
        &source_config_path,
        r#"{"model_type": "qwen3_5", "hidden_size": 2048, "architectures": ["Qwen3_5ForCausalLM"]}"#,
    )
    .expect("the fixture config should be writable");
    let document = QuantizationConfigDocument::from_written_modules(64, 4, &BTreeMap::new());

    document
        .write_into_config(&source_config_path, &output_config_path)
        .expect("writing the config should succeed");

    let written: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&output_config_path).expect("the output config should exist"),
    )
    .expect("the output config should be JSON");
    assert_eq!(written["hidden_size"], 2048);
    assert_eq!(written["quantization"]["mode"], "affine");
    assert_eq!(written["quantization_config"]["mode"], "affine");
    let reread = QuantizationConfigDocument::read_from_config(&output_config_path)
        .expect("reading should succeed")
        .expect("the document should be present");
    assert_eq!(reread, document);
}

#[test]
#[ignore = "pending implementation of source_parameters_for"]
fn should_resolve_per_module_source_parameters_over_the_defaults() {
    let document = QuantizationConfigDocument {
        group_size: 64,
        bits: 4,
        mode: "affine".to_owned(),
        modules: BTreeMap::from([(
            "model.layers.0.mlp.up_proj".to_owned(),
            ModuleQuantizationEntry {
                group_size: 128,
                bits: 8,
                mode: "affine".to_owned(),
            },
        )]),
    };

    assert_eq!(
        document.source_parameters_for("model.layers.0.mlp.up_proj"),
        (8, 128)
    );
    assert_eq!(
        document.source_parameters_for("model.layers.1.mlp.up_proj"),
        (4, 64)
    );
}
