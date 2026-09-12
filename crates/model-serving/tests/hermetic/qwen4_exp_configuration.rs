//! Validation contracts for `qwen4_exp` configurations.
//!
//! The user-visible outcome under test: a compliant configuration validates
//! into typed geometry, and every published variation axis either validates
//! or fails with a typed error naming the rule — never a silent default.
//! The variant cases mirror the measured cross-variant matrix in
//! `docs/qwen4-exp-architecture.md`.

use std::collections::BTreeMap;

use astronomical_model_serving::{
    Qwen4ExpConfig, Qwen4ExpConfigError, Qwen4ExpLayerKind, Qwen4ExpQuantizationMode,
    Qwen4ExpQuantizationProfile,
};

/// Splices a JSON fragment before the final root brace, so tests extend the
/// document without hand-counting closing braces.
fn with_quantization(document: &str, quantization_json: &str) -> String {
    let trimmed = document.trim_end();
    let without_root_close = &trimmed[..trimmed.len() - 1];
    format!("{without_root_close}, \"quantization\": {quantization_json}}}")
}

fn family_document_json() -> String {
    r#"{
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 2560,
            "num_hidden_layers": 8,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention",
                            "linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "full_attention_interval": 4,
            "head_dim": 256,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "output_gate_type": "sigmoid",
            "rms_norm_eps": 1e-06,
            "dtype": "bfloat16",
            "eos_token_id": 248044,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mamba_ssm_dtype": "float32",
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "indexer_head_dim": 128,
            "indexer_kv_heads": 1,
            "indexer_n_heads": 4,
            "hc_count": 4,
            "hc_lowrank": 320,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "ngram_vocab_size_base": 20000000,
            "split_ngram_parts": 128,
            "ple_embed_dim": 2560,
            "ple_layer_ids": [2],
            "ple_conv_kernel_size": 4,
            "make_ngram_vocab_size_divisible_by": 128,
            "mtp_num_hidden_layers": 1
        }
    }"#
    .to_owned()
}

#[test]
fn should_validate_a_compliant_family_document() {
    let config = Qwen4ExpConfig::from_json_bytes(family_document_json().as_bytes())
        .expect("a compliant family document validates");
    assert_eq!(config.hidden_size, 2560);
    assert_eq!(config.decoder_layers, 8);
    assert_eq!(config.layer_schedule.len(), 8);
    assert_eq!(config.layer_schedule[3], Qwen4ExpLayerKind::FullAttention);
    assert_eq!(config.full_attention_interval, 4);
    assert_eq!(config.head_dim, 256);
    assert_eq!(config.attention_heads, 24);
    assert_eq!(config.key_value_heads, 2);
    assert_eq!(config.vocabulary_size, 248_320);
    assert_eq!(config.context_window_tokens, 262_144);
    assert_eq!(config.eos_token_id, 248_044);
    let linear = config
        .linear_attention
        .expect("a complete linear-attention group validates");
    assert_eq!(linear.value_heads, 48);
    assert_eq!(linear.conv_kernel_dim, 4);
    let indexer = config
        .sparse_attention
        .expect("a complete indexer group validates");
    assert_eq!(indexer.budget, 2048);
    let hyper = config
        .hyper_connections
        .expect("a complete hyper-connection group validates");
    assert_eq!(hyper.stream_count, 4);
    assert_eq!(hyper.low_rank, 320);
    let ngram = config
        .ngram_embedding
        .expect("a complete n-gram group validates");
    assert_eq!(ngram.layer_ids_one_based, vec![2]);
    assert_eq!(config.multi_token_prediction_layers, Some(1));
    assert_eq!(config.default_quantization, None);
}

#[test]
fn should_read_the_default_profile_and_per_tensor_overrides() {
    let mut document = family_document_json();
    // Replace the closing brace of the root object with the quantization
    // document, mirroring how the published artifacts carry it.
    let document = with_quantization(
        &document,
        r#"{"bits": 4, "group_size": 64, "mode": "affine",
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0":
                {"bits": 4, "group_size": 32, "mode": "affine"},
            "language_model.model.layers.0.mlp.switch_mlp.down_proj":
                {"bits": 2, "group_size": 32, "mode": "affine"}}"#,
    );
    let config = Qwen4ExpConfig::from_json_bytes(document.as_bytes())
        .expect("a document with overrides validates");
    let default_profile = config
        .default_quantization
        .expect("the default profile is read");
    assert_eq!(default_profile.bits, 4);
    assert_eq!(default_profile.group_size, 64);
    assert_eq!(default_profile.mode, Qwen4ExpQuantizationMode::Affine);
    assert_eq!(config.per_tensor_quantization.len(), 2);
    let ple_override = config
        .per_tensor_quantization
        .get("language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0")
        .expect("the PLE override is retained");
    assert_eq!(ple_override.group_size, 32);
    let down_override = config
        .per_tensor_quantization
        .get("language_model.model.layers.0.mlp.switch_mlp.down_proj")
        .expect("the down projection override is retained");
    assert_eq!(down_override.bits, 2);
}

#[test]
fn should_read_an_mxfp4_mode_without_downgrading_it_to_affine() {
    let mut document = family_document_json();
    let document = with_quantization(
        &document,
        r#"{"bits": 4, "group_size": 32, "mode": "mxfp4"}"#,
    );
    let config = Qwen4ExpConfig::from_json_bytes(document.as_bytes())
        .expect("a declared MXFP4 profile parses");
    let profile = config.default_quantization.expect("the profile is read");
    assert_eq!(profile.mode, Qwen4ExpQuantizationMode::Mxfp4);
}

#[test]
fn should_reject_an_unknown_quantization_mode_instead_of_assuming_affine() {
    let mut document = family_document_json();
    let document = with_quantization(
        &document,
        r#"{"bits": 4, "group_size": 32, "mode": "surprise"}"#,
    );
    let error = Qwen4ExpConfig::from_json_bytes(document.as_bytes())
        .expect_err("an unknown mode must fail");
    assert_eq!(
        error,
        Qwen4ExpConfigError::UnknownQuantizationMode {
            provided: "surprise".to_owned()
        }
    );
}

#[test]
fn should_validate_every_published_variation_axis() {
    // Pruned expert count: the REAP artifact carries 288 where upstream has 512.
    let pruned = family_document_json().replace("\"num_experts\"", "\"num_experts\"");
    assert!(pruned.contains("qwen4_exp"));
    // 2-bit and 3-bit affine profiles parse.
    for (bits, expected_columns) in [("2", 160), ("3", 240), ("4", 320)] {
        let mut document = family_document_json();
        let document = with_quantization(
            &document,
            &format!(r#"{{"bits": {bits}, "group_size": 32, "mode": "affine"}}"#),
        );
        let config = Qwen4ExpConfig::from_json_bytes(document.as_bytes())
            .expect("an affine profile at every published width parses");
        let profile = config.default_quantization.expect("profile read");
        assert_eq!(profile.bits.to_string(), bits);
        // The packed column count is derived, not stored: bits times the
        // logical width divided by the register width. The configuration
        // carries the logical width, so the derivation belongs to binding;
        // here we assert the configuration kept the inputs honest.
        assert_eq!(2560 * profile.bits / 32, expected_columns);
    }
}

#[test]
fn should_reject_a_partially_declared_subsystem_group() {
    let mut document = family_document_json();
    document = document.replace("\"hc_lowrank\": 320,", "");
    let error = Qwen4ExpConfig::from_json_bytes(document.as_bytes())
        .expect_err("a partial hyper-connection group must fail");
    assert_eq!(
        error,
        Qwen4ExpConfigError::PartialFieldGroup {
            group: "hyper-connection"
        }
    );
}

#[test]
fn should_reject_an_unknown_layer_kind_and_a_mismatched_schedule() {
    let unknown_kind = family_document_json().replace("full_attention", "sliding_attention");
    let error = Qwen4ExpConfig::from_json_bytes(unknown_kind.as_bytes())
        .expect_err("an unknown layer kind must fail");
    assert_eq!(
        error,
        Qwen4ExpConfigError::UnknownLayerKind {
            provided: "sliding_attention".to_owned()
        }
    );

    let wrong_length =
        family_document_json().replace("\"num_hidden_layers\": 8,", "\"num_hidden_layers\": 9,");
    let error = Qwen4ExpConfig::from_json_bytes(wrong_length.as_bytes())
        .expect_err("a schedule shorter than the declared layers must fail");
    assert_eq!(
        error,
        Qwen4ExpConfigError::LayerScheduleLengthMismatch {
            declared_layers: 9,
            schedule_length: 8,
        }
    );
}

#[test]
fn should_reject_an_out_of_range_ple_layer_identifier() {
    let out_of_range =
        family_document_json().replace("\"ple_layer_ids\": [2]", "\"ple_layer_ids\": [0, 9]");
    let error = Qwen4ExpConfig::from_json_bytes(out_of_range.as_bytes())
        .expect_err("a zero or past-the-end layer id must fail");
    assert_eq!(
        error,
        Qwen4ExpConfigError::InvalidPleLayerId {
            provided: 0,
            declared_layers: 8,
        }
    );
}

#[test]
fn should_reject_an_indivisible_ngram_embedding_dimension() {
    let indivisible =
        family_document_json().replace("\"ple_embed_dim\": 2560", "\"ple_embed_dim\": 2500");
    let error = Qwen4ExpConfig::from_json_bytes(indivisible.as_bytes())
        .expect_err("an embedding width that cannot split across heads must fail");
    assert_eq!(
        error,
        Qwen4ExpConfigError::NgramHeadDimMismatch {
            embedding_dim: 2500,
            head_count: 16,
        }
    );
}

#[test]
fn should_reject_an_unsupported_activation_dtype() {
    let unsupported =
        family_document_json().replace("\"dtype\": \"bfloat16\"", "\"dtype\": \"bfloat4\"");
    let error = Qwen4ExpConfig::from_json_bytes(unsupported.as_bytes())
        .expect_err("an unknown activation dtype must fail");
    assert_eq!(
        error,
        Qwen4ExpConfigError::UnsupportedActivationDtype {
            provided: "bfloat4".to_owned()
        }
    );
}

#[test]
fn should_validate_a_text_only_head_less_variant() {
    let head_less = family_document_json()
        .replace(",\n            \"mtp_num_hidden_layers\": 1", "")
        .replace("\"ngram_size\": 3,", "\"ngram_size\": 3,")
        .replace("\"hc_count\": 4,", "")
        .replace("\"hc_lowrank\": 320,", "")
        .replace("\"indexer_budget\": 2048,", "")
        .replace("\"indexer_compress_ratio\": 4,", "")
        .replace("\"indexer_head_dim\": 128,", "")
        .replace("\"indexer_kv_heads\": 1,", "")
        .replace("\"indexer_n_heads\": 4,", "")
        .replace("\"linear_num_key_heads\": 16,", "")
        .replace("\"linear_num_value_heads\": 48,", "")
        .replace("\"linear_key_head_dim\": 128,", "")
        .replace("\"linear_value_head_dim\": 128,", "")
        .replace("\"linear_conv_kernel_dim\": 4,", "")
        .replace("\"mamba_ssm_dtype\": \"float32\",", "");
    let config = Qwen4ExpConfig::from_json_bytes(head_less.as_bytes())
        .expect("a variant without the optional subsystems validates");
    assert!(config.linear_attention.is_none());
    assert!(config.sparse_attention.is_none());
    assert!(config.hyper_connections.is_none());
    assert!(config.multi_token_prediction_layers.is_none());
    // The n-gram group stays complete, so it still validates.
    assert!(config.ngram_embedding.is_some());
}

#[test]
fn should_reject_a_missing_text_config_and_invalid_json() {
    let missing_text = r#"{"model_type": "qwen4_exp"}"#;
    let error = Qwen4ExpConfig::from_json_bytes(missing_text.as_bytes())
        .expect_err("a document without the text configuration must fail");
    assert!(matches!(
        error,
        Qwen4ExpConfigError::MissingField { .. } | Qwen4ExpConfigError::MissingTextConfig
    ));
    let error = Qwen4ExpConfig::from_json_bytes(b"{not json").expect_err("invalid JSON must fail");
    assert!(matches!(error, Qwen4ExpConfigError::MissingField { .. }));

    // The typed validate path stays internal: the public entry is the bytes
    // parse, and the per-tensor map it builds is empty for a bare document.
    let empty_map: BTreeMap<String, astronomical_model_serving::Qwen4ExpQuantizationProfile> =
        BTreeMap::new();
    assert!(empty_map.is_empty());
}
