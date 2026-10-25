//! Unit tests for the DFlash draft configuration parser.
//!
//! The positive fixture embeds the exact field set of the real Ornith DFlash
//! draft `config.json` so the parser is locked to the shipped artifact shape.

use super::draft_configuration::{
    DFlashDraftConfiguration, DFlashDraftConfigurationError, DFlashDraftLayerAttentionKind,
};

fn ornith_dflash_draft_config_document() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["DFlashDraftModel"],
        "dflash_config": {
            "block_size": 16,
            "mask_token_id": 248077,
            "target_layer_ids": [1, 6, 11, 16, 22, 27, 32, 37]
        },
        "dtype": "bfloat16",
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "layer_types": [
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention", "full_attention"
        ],
        "max_position_embeddings": 262144,
        "model_type": "qwen3",
        "num_attention_heads": 32,
        "num_hidden_layers": 6,
        "num_key_value_heads": 8,
        "num_target_layers": 40,
        "rms_norm_eps": 1e-06,
        "rope_parameters": { "rope_theta": 10000000, "rope_type": "default" },
        "sliding_window": 4096,
        "vocab_size": 248320
    })
}

#[test]
fn should_parse_the_shipped_ornith_dflash_draft_configuration() {
    let configuration =
        DFlashDraftConfiguration::from_config_value(&ornith_dflash_draft_config_document())
            .expect("the shipped Ornith DFlash draft config must parse");

    assert_eq!(configuration.hidden_size(), 2048);
    assert_eq!(configuration.layer_count(), 6);
    assert_eq!(configuration.attention_head_count(), 32);
    assert_eq!(configuration.key_value_head_count(), 8);
    assert_eq!(configuration.head_dimension(), 128);
    assert_eq!(configuration.vocabulary_size(), 248320);
    assert_eq!(configuration.mask_token_id(), 248077);
    assert_eq!(configuration.draft_block_size(), 16);
    assert_eq!(
        configuration.target_layer_ids(),
        &[1, 6, 11, 16, 22, 27, 32, 37]
    );
    assert_eq!(configuration.target_layer_count(), 40);
    assert_eq!(configuration.target_conditioning_input_width(2048), 16384);
    assert_eq!(configuration.layer_attention_kinds().len(), 6);
}

#[test]
fn should_resolve_sliding_layers_as_causal_and_full_layers_as_bidirectional() {
    let configuration =
        DFlashDraftConfiguration::from_config_value(&ornith_dflash_draft_config_document())
            .expect("the shipped config must parse");

    let layer_attention_kinds = configuration.layer_attention_kinds();
    for attention_kind in &layer_attention_kinds[..5] {
        assert_eq!(
            *attention_kind,
            DFlashDraftLayerAttentionKind::SlidingWindowCausal { window_size: 4096 },
            "the first five layers ship as causal sliding-window layers"
        );
    }
    assert_eq!(
        layer_attention_kinds[5],
        DFlashDraftLayerAttentionKind::FullBidirectional,
        "the last layer ships as the bidirectional full-attention layer"
    );
}

#[test]
fn should_reject_a_document_without_the_draft_architecture() {
    let mut config_document = ornith_dflash_draft_config_document();
    config_document["architectures"] = serde_json::json!(["Qwen3ForCausalLM"]);

    let parse_error = DFlashDraftConfiguration::from_config_value(&config_document)
        .expect_err("a non-DFlash architecture must be rejected");

    assert!(matches!(
        parse_error,
        DFlashDraftConfigurationError::NotADFlashDraftModel
    ));
}

#[test]
fn should_reject_a_missing_dflash_config_block() {
    let mut config_document = ornith_dflash_draft_config_document();
    config_document
        .as_object_mut()
        .expect("the fixture is an object")
        .remove("dflash_config");

    let parse_error = DFlashDraftConfiguration::from_config_value(&config_document)
        .expect_err("a missing dflash_config block must be rejected");

    assert!(matches!(
        parse_error,
        DFlashDraftConfigurationError::MissingField {
            field: "dflash_config"
        }
    ));
}

#[test]
fn should_reject_target_taps_outside_the_target_layer_range() {
    let mut config_document = ornith_dflash_draft_config_document();
    config_document["dflash_config"]["target_layer_ids"] = serde_json::json!([1, 40]);

    let parse_error = DFlashDraftConfiguration::from_config_value(&config_document)
        .expect_err("a tap at the target layer count is out of range");

    assert!(matches!(
        parse_error,
        DFlashDraftConfigurationError::InvalidGeometry { description }
            if description.contains("target layer range")
    ));
}

#[test]
fn should_reject_an_attention_head_count_that_breaks_grouped_query_attention() {
    let mut config_document = ornith_dflash_draft_config_document();
    config_document["num_key_value_heads"] = serde_json::json!(7);

    let parse_error = DFlashDraftConfiguration::from_config_value(&config_document)
        .expect_err("32 query heads cannot split across 7 key-value heads");

    assert!(matches!(
        parse_error,
        DFlashDraftConfigurationError::InvalidGeometry { description }
            if description.contains("grouped-query attention")
    ));
}

#[test]
fn should_reject_a_mask_token_outside_the_vocabulary() {
    let mut config_document = ornith_dflash_draft_config_document();
    config_document["dflash_config"]["mask_token_id"] = serde_json::json!(300000);

    let parse_error = DFlashDraftConfiguration::from_config_value(&config_document)
        .expect_err("a mask token beyond the vocabulary must be rejected");

    assert!(matches!(
        parse_error,
        DFlashDraftConfigurationError::InvalidGeometry { description }
            if description.contains("inside the vocabulary")
    ));
}

#[test]
fn should_reject_layer_types_that_do_not_cover_every_layer() {
    let mut config_document = ornith_dflash_draft_config_document();
    config_document["layer_types"] = serde_json::json!(["sliding_attention", "sliding_attention"]);

    let parse_error = DFlashDraftConfiguration::from_config_value(&config_document)
        .expect_err("two layer types cannot describe six layers");

    assert!(matches!(
        parse_error,
        DFlashDraftConfigurationError::InvalidGeometry { description }
            if description.contains("exactly num_hidden_layers layers")
    ));
}

#[test]
fn should_reject_a_non_positive_block_size() {
    let mut config_document = ornith_dflash_draft_config_document();
    config_document["dflash_config"]["block_size"] = serde_json::json!(0);

    let parse_error = DFlashDraftConfiguration::from_config_value(&config_document)
        .expect_err("a zero training block size cannot be valid");

    assert!(matches!(
        parse_error,
        DFlashDraftConfigurationError::InvalidGeometry { description }
            if description.contains("block_size must be positive")
    ));
}

#[test]
fn should_reject_non_positive_normalization_and_rope_parameters() {
    let mut non_positive_epsilon = ornith_dflash_draft_config_document();
    non_positive_epsilon["rms_norm_eps"] = serde_json::json!(0.0);
    assert!(matches!(
        DFlashDraftConfiguration::from_config_value(&non_positive_epsilon),
        Err(DFlashDraftConfigurationError::InvalidGeometry { description })
            if description.contains("rms_norm_eps must be positive")
    ));

    let mut non_positive_rope_theta = ornith_dflash_draft_config_document();
    non_positive_rope_theta["rope_parameters"]["rope_theta"] = serde_json::json!(0.0);
    assert!(matches!(
        DFlashDraftConfiguration::from_config_value(&non_positive_rope_theta),
        Err(DFlashDraftConfigurationError::InvalidGeometry { description })
            if description.contains("rope_theta must be positive")
    ));
}

#[test]
fn should_derive_head_dimension_from_the_trunk_when_absent() {
    // A hidden size of 2,048 over 16 heads derives a head dimension of 128
    // without an explicit head_dim field.
    let mut config_document = ornith_dflash_draft_config_document();
    config_document
        .as_object_mut()
        .expect("the fixture is an object")
        .remove("head_dim");
    config_document["num_attention_heads"] = serde_json::json!(16);
    let configuration = DFlashDraftConfiguration::from_config_value(&config_document)
        .expect("the derived head dimension should parse");

    assert_eq!(configuration.head_dimension(), 128);
}
