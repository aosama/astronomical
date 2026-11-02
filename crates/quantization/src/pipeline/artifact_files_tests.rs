//! Unit tests for the artifact documents and the verbatim copy.

use std::collections::BTreeMap;

use super::*;
use crate::allocation::ModuleQuantizationDecision;
use crate::bit_width::BitWidth;
use crate::requantize::DEFAULT_COPY_PATTERNS;
use crate::sensitivity::SensitivityResult;

const UP_PROJECTION: &str = "model.layers.0.mlp.up_proj";
const DOWN_PROJECTION: &str = "model.layers.0.mlp.down_proj";

fn two_module_allocation() -> AllocationResult {
    let mut decisions = Vec::new();
    for module_name in [UP_PROJECTION, DOWN_PROJECTION] {
        decisions.push(ModuleQuantizationDecision {
            module_name: module_name.to_owned(),
            bit_width: BitWidth::new(4).expect("valid"),
            group_size: GroupSize::new(64).expect("valid"),
            logical_parameter_count: 4096,
        });
    }
    AllocationResult {
        decisions,
        target_bits_per_weight: 4.0,
        achieved_bits_per_weight: 4.0,
        module_count_at_highest_width: 0,
        module_count_at_lowest_width: 2,
        total_logical_parameter_count: 8192,
        estimated_weight_bytes: 4096,
        floor_upgrade_count: 0,
        run_guard_upgrade_count: 0,
        non_finite_module_names: Vec::new(),
    }
}

fn one_module_table() -> SensitivityTable {
    SensitivityTable {
        method: SensitivityMethod::StaticStructuralRules,
        results: vec![SensitivityResult {
            module_name: UP_PROJECTION.to_owned(),
            kl_divergence_by_bit_width: BTreeMap::from([
                (BitWidth::new(4).expect("valid"), 0.5),
                (BitWidth::new(8).expect("valid"), 0.0),
            ]),
            logical_parameter_count: 4096,
        }],
    }
}

#[test]
fn should_describe_what_was_written_rather_than_the_plan() {
    // down_proj was planned at 4 bits but written at bfloat16 (absent from
    // the written map); neither document may claim it is quantized.
    let source = tempfile::tempdir().expect("a temporary directory should be creatable");
    let artifact = tempfile::tempdir().expect("a temporary directory should be creatable");
    std::fs::write(
        source.path().join("config.json"),
        r#"{"model_type": "qwen3_5", "hidden_size": 256}"#,
    )
    .expect("the source config should be writable");
    let written = BTreeMap::from([(UP_PROJECTION.to_owned(), (4_u32, 64_u32))]);
    let allocation = two_module_allocation();
    let table = one_module_table();

    let sensitivity_written = write_artifact_documents(&ArtifactDocumentsRequest {
        source_directory: source.path(),
        artifact_directory: artifact.path(),
        base_model_name: "fixture/base",
        method: SensitivityMethod::StaticStructuralRules,
        allocation: &allocation,
        sensitivity_table: &table,
        candidate_bits: &[4, 8],
        group_size: GroupSize::new(64).expect("valid"),
        default_bits: 4,
        written_quantization_by_module: &written,
    })
    .expect("the documents should be written");

    assert!(sensitivity_written);
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(artifact.path().join("config.json")).expect("config"),
    )
    .expect("config parses");
    assert_eq!(config["hidden_size"], 256);
    assert_eq!(config["quantization"][UP_PROJECTION]["bits"], 4);
    assert!(config["quantization"].get(DOWN_PROJECTION).is_none());
    assert_eq!(config["quantization"], config["quantization_config"]);
    let metadata: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(artifact.path().join("optiq/metadata.json")).expect("metadata"),
    )
    .expect("metadata parses");
    assert!(metadata["per_layer"].get(DOWN_PROJECTION).is_none());
    assert!(artifact.path().join("optiq/sensitivity.json").is_file());
}

#[test]
fn should_refuse_when_the_source_config_is_missing() {
    let source = tempfile::tempdir().expect("a temporary directory should be creatable");
    let artifact = tempfile::tempdir().expect("a temporary directory should be creatable");
    let allocation = two_module_allocation();
    let table = one_module_table();

    let refusal = write_artifact_documents(&ArtifactDocumentsRequest {
        source_directory: source.path(),
        artifact_directory: artifact.path(),
        base_model_name: "fixture/base",
        method: SensitivityMethod::StaticStructuralRules,
        allocation: &allocation,
        sensitivity_table: &table,
        candidate_bits: &[4, 8],
        group_size: GroupSize::new(64).expect("valid"),
        default_bits: 4,
        written_quantization_by_module: &BTreeMap::new(),
    });

    assert!(matches!(
        refusal,
        Err(QuantizationError::ModelConfigUnreadable { .. })
    ));
}

#[test]
fn should_copy_tokenizer_and_template_files_but_not_weights_or_directories() {
    let source = tempfile::tempdir().expect("a temporary directory should be creatable");
    let artifact = tempfile::tempdir().expect("a temporary directory should be creatable");
    for file_name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "model.safetensors",
        "config.json",
    ] {
        std::fs::write(source.path().join(file_name), b"{}").expect("writable");
    }
    std::fs::create_dir(source.path().join("tokenizer_assets")).expect("creatable");

    let copied_file_count =
        copy_source_files(source.path(), artifact.path(), DEFAULT_COPY_PATTERNS)
            .expect("the copy should succeed");

    assert_eq!(copied_file_count, 4);
    assert!(artifact.path().join("chat_template.jinja").is_file());
    assert!(!artifact.path().join("model.safetensors").exists());
    assert!(!artifact.path().join("config.json").exists());
    assert!(!artifact.path().join("tokenizer_assets").exists());
}

#[test]
fn should_match_prefix_suffix_and_exact_patterns_only() {
    assert!(matches_any_copy_pattern("tokenizer.model", &["tokenizer*"]));
    assert!(matches_any_copy_pattern("template.jinja", &["*.jinja"]));
    assert!(matches_any_copy_pattern(
        "generation_config.json",
        &["generation_config.json"]
    ));
    assert!(!matches_any_copy_pattern(
        "my_tokenizer.json",
        &["tokenizer*"]
    ));
    assert!(!matches_any_copy_pattern(
        "generation_config.json.bak",
        &["generation_config.json"]
    ));
    assert!(!matches_any_copy_pattern("anything", &[]));
}

#[test]
fn should_fail_with_a_typed_error_when_the_source_directory_is_missing() {
    let artifact = tempfile::tempdir().expect("a temporary directory should be creatable");

    let failure = copy_source_files(
        &artifact.path().join("absent"),
        artifact.path(),
        DEFAULT_COPY_PATTERNS,
    );

    assert!(matches!(
        failure,
        Err(QuantizationError::ArtifactWriteFailure { .. })
    ));
}
