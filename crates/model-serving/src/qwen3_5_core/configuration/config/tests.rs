//! Offline contracts for Qwen3.5 configuration parsing and validation.
//!
//! Grouped by concern under one `tests` child of the config owner; every file
//! in this folder carries the `_tests.rs` suffix. Fixtures come from the
//! frozen Ornith 1.0 configuration support module.

use serde_json::{Value, json};

use crate::qwen3_5_core::artifacts::qwen3_5_language_tensor_profiles;
use crate::qwen3_5_core::frozen_config_test_support::{
    frozen_optiq_metadata_bytes, frozen_ornith_1_0_config_bytes,
    frozen_ornith_1_0_optiq_config_bytes,
};
use crate::qwen3_5_core::quantizations::optiq::{OptiQMetadata, OptiQMetadataError};

use super::{ModelWeightStorage, Qwen3_5Config};
use crate::qwen3_5_core::configuration::Qwen3_5ConfigError;

mod compatibility_fallbacks_tests;
mod dtype_profile_tests;
mod model_contract_tests;
mod optiq_metadata_tests;
mod quantization_tests;

pub(crate) fn minimal_valid_config_json() -> Value {
    let config_bytes = br#"
    {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "dtype": "bfloat16",
        "eos_token_id": [248046, 248044],
        "tie_word_embeddings": false,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_act": "silu",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 0.000001,
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000.0,
                "type": "default"
            },
            "partial_rotary_factor": 0.25,
            "attention_bias": false,
            "mlp_bias": false,
            "norm_topk_prob": true,
            "output_router_logits": false,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "full_attention_interval": 4,
            "linear_conv_kernel_dim": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512
        },
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "mode": "affine"
        },
        "quantization_config": {
            "group_size": 64,
            "bits": 4,
            "mode": "affine"
        }
    }
    "#;
    let mut config_value = serde_json::from_slice::<Value>(config_bytes)
        .expect("the minimal test config should decode as JSON");
    // Add the decoder attention schedule.
    config_value["text_config"]["layer_types"] = json!(
        (0..40)
            .map(|i| if i % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            })
            .collect::<Vec<_>>()
    );
    // Add quantization per-layer overrides (required by the parser).
    let mut quantization = json!({"group_size": 64, "bits": 4, "mode": "affine"});
    quantization["language_model.model.embed_tokens"] = json!({"group_size": 64, "bits": 8});
    quantization["language_model.lm_head"] = json!({"group_size": 64, "bits": 8});
    config_value["quantization"] = quantization.clone();
    config_value["quantization_config"] = quantization;
    config_value
}
