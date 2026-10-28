//! The frozen Ornith 1.0 fixture web for the persisted-storage contracts, built
//! from the crate's public API. The beside-source unit tests build the same
//! fixtures in-crate; the acceptance tier keeps its own typed copy beside the
//! GPU contracts. All three express the identical frozen model contract.

use crate::PersistentPromptCacheModelContract;
use crate::{
    DecoderCacheTensorDtype, ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
    ORNITH_1_0_35B_OPTIQ_4BIT_REVISION, Qwen3_5Config, Qwen3_5DecoderLayerCacheDtypes,
};
use serde_json::{Value, json};

pub fn frozen_ornith_1_0_config() -> Qwen3_5Config {
    Qwen3_5Config::from_json_bytes(&frozen_ornith_1_0_optiq_config_bytes())
        .expect("the frozen Ornith 1.0 config should parse")
}

pub fn persistent_prompt_cache_model_contract() -> PersistentPromptCacheModelContract {
    let frozen_ornith_1_0_config = frozen_ornith_1_0_config();
    // This fixture has no bound MLX arrays. Supply its frozen BF16
    // state contract explicitly; real engine acceptance uses load-derived
    // dtypes and read the resulting block geometry back from the engine.
    let decoder_layer_cache_dtypes = bfloat16_decoder_layer_cache_dtypes(&frozen_ornith_1_0_config);
    PersistentPromptCacheModelContract::resolve(
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID.to_owned(),
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION.to_owned(),
        crate::qwen3_5_decoder_cache_layout(
            &frozen_ornith_1_0_config,
            256,
            &decoder_layer_cache_dtypes,
        )
        .expect("the frozen Ornith 1.0 configuration should build a decoder-cache layout"),
        frozen_ornith_1_0_config.maximum_position_count() as usize,
        20_000_000_000,
        50_000_000_000,
        None,
        4,
    )
    .expect("the frozen Ornith 1.0 configuration should resolve a storage contract")
}

pub fn bfloat16_decoder_layer_cache_dtypes(
    qwen3_5_config: &Qwen3_5Config,
) -> Vec<Qwen3_5DecoderLayerCacheDtypes> {
    decoder_layer_cache_dtypes(qwen3_5_config, DecoderCacheTensorDtype::BFloat16)
}

fn decoder_layer_cache_dtypes(
    qwen3_5_config: &Qwen3_5Config,
    activation_state_dtype: DecoderCacheTensorDtype,
) -> Vec<Qwen3_5DecoderLayerCacheDtypes> {
    (0..qwen3_5_config.layer_count() as usize)
        .map(|decoder_layer_index| {
            if qwen3_5_config.decoder_layer_is_full_attention(decoder_layer_index) {
                Qwen3_5DecoderLayerCacheDtypes::FullAttention {
                    keys: activation_state_dtype,
                    values: activation_state_dtype,
                }
            } else {
                Qwen3_5DecoderLayerCacheDtypes::LinearAttention {
                    convolution: activation_state_dtype,
                }
            }
        })
        .collect()
}

pub fn frozen_ornith_1_0_optiq_config_bytes() -> Vec<u8> {
    let config_bytes = br#"
    {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "dtype": "bfloat16",
        "eos_token_id": [248046, 248044],
        "pad_token_id": 248044,
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
        }
    }
    "#;
    let mut config_value = serde_json::from_slice::<Value>(config_bytes)
        .expect("the frozen OptiQ test config should decode as JSON");
    let mut quantization = json!({
        "group_size": 64,
        "bits": 4,
        "mode": "affine",
    });
    let quantized_module_names = expected_quantized_module_names();
    for (quantized_module_index, quantized_module_name) in quantized_module_names.iter().enumerate()
    {
        let quantization_bits = if quantized_module_index < 113 { 4 } else { 8 };
        quantization[quantized_module_name] = json!({"group_size": 64, "bits": quantization_bits});
    }
    config_value["quantization"] = quantization.clone();
    config_value["quantization_config"] = quantization;
    config_value["text_config"]["layer_types"] = json!(expected_layer_types());
    serde_json::to_vec(&config_value)
        .expect("the frozen OptiQ test config should serialize as JSON")
}

fn expected_quantized_module_names() -> Vec<String> {
    let mut quantized_module_names = Vec::with_capacity(512);
    for decoder_layer_index in 0..40 {
        let layer_prefix = format!("language_model.model.layers.{decoder_layer_index}");
        if decoder_layer_index % 4 == 3 {
            for projection_name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                quantized_module_names.push(format!("{layer_prefix}.self_attn.{projection_name}"));
            }
        } else {
            for projection_name in [
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_b",
                "in_proj_a",
                "out_proj",
            ] {
                quantized_module_names
                    .push(format!("{layer_prefix}.linear_attn.{projection_name}"));
            }
        }
        for module_suffix in [
            "mlp.gate",
            "mlp.switch_mlp.gate_proj",
            "mlp.switch_mlp.up_proj",
            "mlp.switch_mlp.down_proj",
            "mlp.shared_expert.gate_proj",
            "mlp.shared_expert.up_proj",
            "mlp.shared_expert.down_proj",
            "mlp.shared_expert_gate",
        ] {
            quantized_module_names.push(format!("{layer_prefix}.{module_suffix}"));
        }
    }
    quantized_module_names.push("language_model.model.embed_tokens".to_owned());
    quantized_module_names.push("language_model.lm_head".to_owned());
    quantized_module_names
}

fn expected_layer_types() -> Vec<&'static str> {
    (0..40)
        .map(|decoder_layer_index| {
            if decoder_layer_index % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect()
}
