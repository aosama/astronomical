//! Refusal contracts: every unsupported declaration is a typed refusal, so a
//! future pack revision is accepted deliberately rather than mis-served.

use super::{parse_config, shipped_shape_config_json};

fn refusal(
    mutate: impl FnOnce(&mut serde_json::Value),
) -> crate::prism_hadamard_qwen35::configuration::FoldedPackConfigError {
    let mut document = shipped_shape_config_json();
    mutate(&mut document);
    let error = parse_config(&document).expect_err("every mutated fixture must be refused");
    error
}

#[test]
fn should_refuse_unsupported_identity_fields() {
    use super::FoldedPackConfigError;
    assert!(matches!(
        refusal(|d| d["schema_version"] = serde_json::json!(3)),
        FoldedPackConfigError::UnsupportedSchemaVersion {
            actual_schema_version: 3
        }
    ));
    assert!(matches!(
        refusal(|d| d["model_type"] = serde_json::json!("something_else")),
        FoldedPackConfigError::UnsupportedModelType { .. }
    ));
    assert!(matches!(
        refusal(|d| d["base_model_type"] = serde_json::json!("qwen3_8")),
        FoldedPackConfigError::UnsupportedBaseModelType { .. }
    ));
    assert!(matches!(
        refusal(|d| d["hadamard_config"] = serde_json::json!("other.json")),
        FoldedPackConfigError::UnsupportedHadamardConfigReference { .. }
    ));
    assert!(matches!(
        refusal(|d| d["tensor_namespace"] = serde_json::json!("unknown-namespace")),
        FoldedPackConfigError::UnsupportedTensorNamespace { .. }
    ));
    assert!(matches!(
        refusal(|d| d["gdn_activation_layout"] = serde_json::json!("flat")),
        FoldedPackConfigError::UnsupportedGdnLayout { .. }
    ));
}

#[test]
fn should_refuse_unsupported_component_declarations() {
    use super::FoldedPackConfigError;
    assert!(matches!(
        refusal(
            |d| d["components"] = serde_json::json!({"text": true, "vision": true, "mtp": true})
        ),
        FoldedPackConfigError::MultiTokenPredictionUnsupported
    ));
    assert!(matches!(
        refusal(
            |d| d["components"] = serde_json::json!({"text": false, "vision": true, "mtp": false})
        ),
        FoldedPackConfigError::TextComponentMissing
    ));
    assert!(matches!(
        refusal(|d| {
            d["components"] = serde_json::json!({"text": true, "vision": true, "mtp": false});
            d["vision_config"] = serde_json::Value::Null;
        }),
        FoldedPackConfigError::VisionConfigMissing
    ));
    assert!(matches!(
        refusal(|d| {
            d["components"] = serde_json::json!({"text": true, "vision": true, "mtp": false});
            d["image_token_id"] = serde_json::Value::Null;
        }),
        FoldedPackConfigError::VisionTokenIdsMissing
    ));
    assert!(matches!(
        refusal(|d| d["tie_word_embeddings"] = serde_json::json!(true)),
        FoldedPackConfigError::TiedEmbeddingsUnsupported
    ));
}

#[test]
fn should_refuse_unsupported_quantization() {
    use super::FoldedPackConfigError;
    assert!(matches!(
        refusal(|d| d["quantization"] =
            serde_json::json!({"bits": 4, "group_size": 128, "mode": "affine"})),
        FoldedPackConfigError::UnsupportedQuantization { actual_bits: 4, .. }
    ));
    assert!(matches!(
        refusal(|d| d["quantization"] =
            serde_json::json!({"bits": 2, "group_size": 64, "mode": "affine"})),
        FoldedPackConfigError::UnsupportedQuantization {
            actual_group_size: 64,
            ..
        }
    ));
}

#[test]
fn should_refuse_unsupported_text_semantics() {
    use super::FoldedPackConfigError;
    assert!(matches!(
        refusal(|d| d["text_config"]["model_type"] = serde_json::json!("qwen3_5_moe_text")),
        FoldedPackConfigError::UnsupportedTextModelType { .. }
    ));
    assert!(matches!(
        refusal(|d| d["text_config"]["hidden_act"] = serde_json::json!("gelu")),
        FoldedPackConfigError::UnsupportedHiddenActivation { .. }
    ));
    assert!(matches!(
        refusal(|d| d["text_config"]["output_gate_type"] = serde_json::json!("sigmoid")),
        FoldedPackConfigError::UnsupportedOutputGateType { .. }
    ));
    assert!(matches!(
        refusal(|d| d["text_config"]["attn_output_gate"] = serde_json::json!(false)),
        FoldedPackConfigError::OutputGateMissing
    ));
    assert!(matches!(
        refusal(|d| d["text_config"]["mamba_ssm_dtype"] = serde_json::json!("bfloat16")),
        FoldedPackConfigError::UnsupportedSsmDtype { .. }
    ));
    assert!(matches!(
        refusal(|d| d["text_config"]["attention_bias"] = serde_json::json!(true)),
        FoldedPackConfigError::AttentionBiasUnsupported
    ));
    assert!(matches!(
        refusal(|d| d["text_config"]["tie_word_embeddings"] = serde_json::json!(true)),
        FoldedPackConfigError::TiedEmbeddingsUnsupported
    ));
    assert!(matches!(
        refusal(|d| d["text_config"]["rope_parameters"]["type"] = serde_json::json!("yarn")),
        FoldedPackConfigError::UnsupportedRopeType { .. }
    ));
    assert!(matches!(
        refusal(|d| d["text_config"]["rope_parameters"]["mrope_section"] = serde_json::json!([])),
        FoldedPackConfigError::EmptyMropeSection
    ));
}

#[test]
fn should_refuse_invalid_dimensionality() {
    use super::FoldedPackConfigError;
    assert!(matches!(
        refusal(|d| d["text_config"]["hidden_size"] = serde_json::json!(0)),
        FoldedPackConfigError::NonPositiveDimension
    ));
    assert!(matches!(
        refusal(|d| d["text_config"]["num_key_value_heads"] = serde_json::json!(8)),
        FoldedPackConfigError::InvalidKeyValueHeadCount { .. }
    ));
    assert!(matches!(
        refusal(|d| d["text_config"]["linear_num_value_heads"] = serde_json::json!(6)),
        FoldedPackConfigError::InvalidLinearHeadCounts { .. }
    ));
    assert!(matches!(
        refusal(|d| d["text_config"]["layer_types"] =
            serde_json::json!(["linear_attention", "linear_attention", "linear_attention"])),
        FoldedPackConfigError::LayerTypeCountMismatch { .. }
    ));
    assert!(matches!(
        refusal(|d| d["text_config"]["layer_types"] = serde_json::json!([
            "linear_attention",
            "linear_attention",
            "sliding_attention",
            "full_attention"
        ])),
        FoldedPackConfigError::UnsupportedLayerType { .. }
    ));
}

#[test]
fn should_refuse_invalid_packed_modules() {
    use super::FoldedPackConfigError;
    assert!(matches!(
        refusal(|d| d["modules"] = serde_json::json!([])),
        FoldedPackConfigError::NoPackedModules
    ));
    assert!(matches!(
        refusal(|d| d["modules"][0]["dtype"] = serde_json::json!("bfloat16")),
        FoldedPackConfigError::UnsupportedModuleDtype { .. }
    ));
    assert!(matches!(
        refusal(|d| d["modules"][0]["block"] = serde_json::json!(768)),
        FoldedPackConfigError::UnsupportedModuleBlock { .. }
    ));
    assert!(matches!(
        refusal(|d| d["modules"][2]["path"] = serde_json::json!("lm_head")),
        FoldedPackConfigError::DuplicateModulePath { .. }
    ));
}

#[test]
fn should_refuse_an_absurdly_long_requires_runtime_marker() {
    use super::FoldedPackConfigError;
    // Absence is legitimate, but a marker longer than the bound is refused
    // because it rides inside every load error report.
    assert!(matches!(
        refusal(|d| {
            d["requires_runtime"] =
                serde_json::json!["x".repeat(super::super::MAXIMUM_REQUIRES_RUNTIME_BYTES + 1)];
        }),
        FoldedPackConfigError::InvalidRequiresRuntime
    ));
}
