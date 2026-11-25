//! Flash-Next configuration documents for unit tests: the published 4-bit
//! checkpoint's text configuration as it ships, and a miniature with the
//! same structure small enough to build weights for on the graphics
//! processor. The quantization map lists only the overrides each test needs.

use serde_json::{Value, json};

use crate::qwen3_5_core::configuration::Qwen3_5Config;

/// The published checkpoint's configuration document, with the quantization
/// overrides reduced to one of each kind.
pub fn frozen_flash_next_config_document() -> Value {
    let layer_types = (0..48)
        .map(|layer_index| {
            if layer_index % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect::<Vec<_>>();
    json!({
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "eos_token_id": [248046, 248044],
        "tie_word_embeddings": false,
        "quantization": flash_next_quantization(),
        "quantization_config": flash_next_quantization(),
        "text_config": flash_next_text_config(&layer_types)
    })
}

fn flash_next_text_config(layer_types: &[&str]) -> Value {
    let mut text_config: Value = serde_json::from_str(FLASH_NEXT_TEXT_CONFIG)
        .expect("the frozen Flash-Next text config is valid JSON");
    text_config["layer_types"] = json!(layer_types);
    text_config
}

/// The published `text_config`, without `layer_types`, which the builder
/// adds from the full-attention interval.
const FLASH_NEXT_TEXT_CONFIG: &str = r#"{
    "attention_bias": false, "dtype": "bfloat16", "eos_token_id": 248044,
    "full_attention_interval": 4, "hc_count": 4, "hc_lowrank": 320, "head_dim": 256,
    "heads_per_ngram": 8, "hidden_act": "silu", "hidden_size": 2560,
    "indexer_budget": 2048, "indexer_compress_ratio": 4, "indexer_head_dim": 128,
    "indexer_kv_heads": 1, "indexer_n_heads": 4,
    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
    "linear_num_value_heads": 48, "linear_value_head_dim": 128,
    "make_ngram_vocab_size_divisible_by": 128, "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 262144, "model_type": "qwen4_exp_text",
    "moe_intermediate_size": 640, "mtp_num_hidden_layers": 1, "ngram_size": 3,
    "ngram_vocab_size_base": 20000000, "num_attention_heads": 24, "num_experts": 512,
    "num_experts_per_tok": 10, "num_hidden_layers": 48, "num_key_value_heads": 2,
    "output_gate_type": "sigmoid", "partial_rotary_factor": 0.25,
    "ple_conv_kernel_size": 4, "ple_embed_dim": 2560, "ple_layer_ids": [2],
    "rms_norm_eps": 0.000001,
    "rope_parameters": {
        "mrope_interleaved": true, "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"
    },
    "shared_expert_intermediate_size": 640, "split_ngram_parts": 128,
    "tie_word_embeddings": false, "vocab_size": 248320
}"#;

fn flash_next_quantization() -> Value {
    json!({
        "group_size": 32, "bits": 4, "mode": "affine",
        "language_model.model.layers.0.mlp.gate": {"group_size": 64, "bits": 8, "mode": "affine"},
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0":
            {"group_size": 32, "bits": 4, "mode": "affine"}
    })
}

pub fn frozen_flash_next_config() -> Qwen3_5Config {
    Qwen3_5Config::from_json_bytes(&frozen_flash_next_config_document().to_string().into_bytes())
        .expect("the frozen Flash-Next config should parse")
}
