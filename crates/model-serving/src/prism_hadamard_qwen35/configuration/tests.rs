//! Offline contracts for folded-pack configuration parsing.
//!
//! Fixtures are small synthetic documents shaped exactly like the shipped
//! 27B pack (same keys, same wire conventions, tiny dimensions).

use serde_json::{Value, json};

pub(crate) use super::FoldedPackConfigError;

mod acceptance_tests;
mod refusal_tests;

pub(crate) fn shipped_shape_config_json() -> Value {
    json!({
        "schema_version": 2,
        "model_type": "prism_hadamard_qwen35",
        "base_model_type": "qwen3_5",
        "requires_runtime": "runtime/artifact.py",
        "hadamard_config": "hadamard.json",
        "tensor_namespace": "mlx-vlm-qwen3_5",
        "gdn_activation_layout": "grouped",
        "components": {"text": true, "vision": true, "mtp": false},
        "text_config": shipped_shape_text_config_json(),
        "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
        "modules": [
            {"path": "lm_head", "block": 1024, "embedding": false, "dtype": "float16"},
            {"path": "model.embed_tokens", "block": 1024, "embedding": true, "dtype": "float16"},
            {"path": "model.layers.0.mlp.down_proj", "block": 1024, "embedding": false, "dtype": "float16"},
        ],
        "tie_word_embeddings": false,
        "vision_config": {"model_type": "qwen3_5", "hidden_size": 64},
        "image_token_id": 151164,
        "video_token_id": 151165,
        "vision_start_token_id": 151161,
        "vision_end_token_id": 151162,
    })
}

pub(crate) fn shipped_shape_text_config_json() -> Value {
    json!({
        "model_type": "qwen3_5_text",
        "hidden_act": "silu",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 256,
        "max_position_embeddings": 4096,
        "full_attention_interval": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "linear_num_value_heads": 8,
        "linear_num_key_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "rms_norm_eps": 0.000001,
        "rope_parameters": {
            "type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": true,
            "mrope_section": [5, 5, 6],
        },
        "eos_token_id": 248044,
        "attention_bias": false,
        "tie_word_embeddings": false,
        "attn_output_gate": true,
        "output_gate_type": "swish",
        "mamba_ssm_dtype": "float32",
        // Inert upstream fields flow through untouched (tolerant text schema).
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "use_cache": true,
        "mtp_num_hidden_layers": 0,
        "mtp_use_dedicated_embeddings": false,
    })
}

pub(crate) fn parse_config(
    config_json: &Value,
) -> Result<super::FoldedPackConfig, super::FoldedPackConfigError> {
    super::FoldedPackConfig::from_json_bytes(config_json.to_string().as_bytes())
}
