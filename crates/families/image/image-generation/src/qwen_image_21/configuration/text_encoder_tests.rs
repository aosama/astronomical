//! Beside-source coverage for the reviewed text-encoder configuration: a
//! full reviewed document parses into the geometry, and each section
//! validator rejects the deviation its name names.

use super::QwenImage21TextEncoderConfig;
use crate::qwen_image_21::configuration::QwenImage21ConfigError;

const REVIEWED_DOCUMENT_JSON: &str = r#"{
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "dtype": "bfloat16",
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "tie_word_embeddings": false,
    "transformers_version": "4.57.1",
    "video_token_id": 151656,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
    "mlx_format": true,
    "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    "text_config": {
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 12288,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {"mrope_interleaved": true, "mrope_section": [24, 20, 20], "rope_type": "default"},
        "rope_theta": 5000000,
        "use_cache": true,
        "vocab_size": 151936
    },
    "vision_config": {
        "deepstack_visual_indexes": [8, 16, 24],
        "depth": 27,
        "dtype": "bfloat16",
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "qwen3_vl",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 4096,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2
    }
}"#;

#[test]
fn should_parse_the_reviewed_text_encoder_document() {
    let config = QwenImage21TextEncoderConfig::parse(REVIEWED_DOCUMENT_JSON.as_bytes())
        .expect("the reviewed document parses");

    assert_eq!(config.vocab_size, 151_936);
    assert_eq!(config.mrope_section, [24, 20, 20]);
    assert_eq!(config.vision_depth, 27);
    assert_eq!(config.quantization_bits, 4);
    assert_eq!(config.image_token_id, 151_655);
}

#[test]
fn should_reject_an_unreviewed_text_geometry_fact() {
    let unreviewed_document =
        REVIEWED_DOCUMENT_JSON.replace("\"vocab_size\": 151936", "\"vocab_size\": 151937");

    let outcome = QwenImage21TextEncoderConfig::parse(unreviewed_document.as_bytes());

    assert!(
        matches!(
            &outcome,
            Err(QwenImage21ConfigError::UnsupportedProfile { field, .. })
                if *field == "text_config.vocab_size"
        ),
        "the geometry section names the offending field: {outcome:?}"
    );
}

#[test]
fn should_reject_an_unreviewed_document_header_fact() {
    let unreviewed_document = REVIEWED_DOCUMENT_JSON.replace(
        "\"transformers_version\": \"4.57.1\"",
        "\"transformers_version\": \"4.58.0\"",
    );

    let outcome = QwenImage21TextEncoderConfig::parse(unreviewed_document.as_bytes());

    assert!(
        matches!(
            &outcome,
            Err(QwenImage21ConfigError::UnsupportedProfile { field, .. }) if *field == "transformers_version"
        ),
        "the header section names the offending field: {outcome:?}"
    );
}

#[test]
fn should_reject_a_malformed_document() {
    let outcome = QwenImage21TextEncoderConfig::parse(b"{ not json");

    assert!(
        matches!(outcome, Err(QwenImage21ConfigError::Malformed { .. })),
        "malformed bytes report the Malformed variant: {outcome:?}"
    );
}
