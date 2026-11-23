//! Frozen Ornith 1.0 configuration fixtures shared by beside-source unit tests.
//!
//! These byte-exact configs pin the Qwen3.5 config parser and tensor-profile
//! generators to one frozen model contract. They compile only under
//! `cfg(test)`, so production builds never carry them. The configuration
//! documents come from `test_harness::fixtures::frozen_ornith_1_0_fixture`, the one
//! copy the serving integration tests and the acceptance GPU contracts also
//! read; this file keeps only the wrappers that need Qwen 3.5 types.

use serde_json::{Value, json};

use crate::astro_ssd_prompt_cache::PersistentVisualEmbeddingModelContract;
use crate::qwen3_5_core::artifacts::qwen3_5_language_tensor_profiles;
use crate::qwen3_5_core::vision::{Qwen3_5ImageProcessor, Qwen3_5VisionConfig};

use super::configuration::Qwen3_5Config;
pub use test_harness::fixtures::frozen_ornith_1_0_fixture::{
    frozen_optiq_metadata_bytes, frozen_ornith_1_0_config_bytes,
    frozen_ornith_1_0_optiq_config_bytes, frozen_ornith_1_0_optiq_vision_config_bytes,
    frozen_sparse_mixed_precision_config_bytes,
};

pub fn frozen_ornith_1_0_config() -> Qwen3_5Config {
    Qwen3_5Config::from_json_bytes(&frozen_ornith_1_0_optiq_config_bytes())
        .expect("the frozen Ornith 1.0 config should parse")
}

pub fn frozen_ornith_1_0_vision_config() -> Qwen3_5VisionConfig {
    Qwen3_5VisionConfig::from_json_bytes(&frozen_ornith_1_0_optiq_vision_config_bytes())
        .expect("the frozen Ornith 1.0 vision config should parse")
}

pub fn frozen_ornith_1_0_image_processor() -> Qwen3_5ImageProcessor {
    Qwen3_5ImageProcessor::from_vision_config(&frozen_ornith_1_0_vision_config())
}

pub fn persistent_visual_embedding_model_contract() -> PersistentVisualEmbeddingModelContract {
    let frozen_ornith_1_0_vision_config = frozen_ornith_1_0_vision_config();
    let qwen3_5_image_processor = frozen_ornith_1_0_image_processor();
    PersistentVisualEmbeddingModelContract::new(
        super::ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID.to_owned(),
        super::ORNITH_1_0_35B_OPTIQ_4BIT_REVISION.to_owned(),
        frozen_ornith_1_0_vision_config.out_hidden_size() as usize,
        qwen3_5_image_processor.maximum_image_token_count_after_spatial_merge(),
    )
}

pub fn frozen_dense_qwen3_6_config() -> Qwen3_5Config {
    let mut dense_config_document =
        serde_json::from_slice::<Value>(&frozen_ornith_1_0_optiq_config_bytes())
            .expect("the frozen OptiQ config should decode as JSON");
    dense_config_document["architectures"] = json!(["Qwen3_5ForConditionalGeneration"]);
    dense_config_document["model_type"] = json!("qwen3_5");
    dense_config_document["text_config"]["model_type"] = json!("qwen3_5_text");
    dense_config_document["text_config"]["num_experts"] = json!(0);
    dense_config_document["text_config"]["num_experts_per_tok"] = json!(0);
    dense_config_document["text_config"]["moe_intermediate_size"] = json!(0);
    dense_config_document["text_config"]["shared_expert_intermediate_size"] = json!(0);
    dense_config_document["text_config"]["intermediate_size"] = json!(512);
    let dense_quantization_config = json!({"bits": 4, "group_size": 64, "mode": "affine"});
    dense_config_document["quantization"] = dense_quantization_config.clone();
    dense_config_document["quantization_config"] = dense_quantization_config;
    let dense_config_bytes = serde_json::to_vec(&dense_config_document)
        .expect("the frozen dense config should serialize as JSON");
    Qwen3_5Config::from_json_bytes(&dense_config_bytes)
        .expect("the frozen dense config should parse")
}

pub fn expected_qwen3_5_language_tensor_profiles() -> Vec<crate::TensorProfile> {
    qwen3_5_language_tensor_profiles(&frozen_ornith_1_0_config())
        .expect("the frozen ornith 1.0 language tensor profiles should build")
}

/// A one-pixel red PNG used as the synthetic vision fixture.
pub const SYNTHETIC_RED_PNG_BYTES: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0,
    0, 0, 31, 21, 196, 137, 0, 0, 0, 13, 73, 68, 65, 84, 120, 156, 99, 248, 207, 192, 240, 31, 0,
    5, 0, 1, 255, 137, 153, 61, 29, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];

/// The frozen Ornith 1.0 configuration with its vision stack attached, used by
/// vision-config parsing contracts.
pub const FROZEN_VISION_CONFIG_JSON: &str = r#"{
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "dtype": "bfloat16",
    "eos_token_id": [248046, 248044],
    "tie_word_embeddings": false,
    "quantization": {"config_file": "optiq/optiq.safetensors", "n_tensors": 333, "base_model": "deepreinforce-ai/Ornith-1.0-35B"},
    "quantization_config": {"config_file": "optiq/optiq.safetensors", "n_tensors": 333, "base_model": "deepreinforce-ai/Ornith-1.0-35B"},
    "text_config": {
        "model_type": "qwen3_5_moe_text",
        "hidden_act": "silu",
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
        "attention_bias": false,
        "mlp_bias": false,
        "norm_topk_prob": true,
        "output_router_logits": false,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
        "full_attention_interval": 4,
        "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention"],
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
    "vision_config": {
        "deepstack_visual_indexes": [],
        "depth": 27,
        "dtype": "bfloat16",
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "qwen3_5_moe_vision",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2048,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2
    }
}"#;
