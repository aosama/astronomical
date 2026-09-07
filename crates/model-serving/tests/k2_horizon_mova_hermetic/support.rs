use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use astronomical_model_serving::K2HorizonMoVAConfig;

pub(super) fn family_member_config_json(
    hidden_layers: usize,
    mlp_only_layers: &[usize],
    num_experts: usize,
    mova_num_experts: usize,
) -> String {
    serde_json::json!({
        "model_type": "k2_horizon_mova",
        "architectures": ["K2HorizonForCausalLM"],
        "hidden_size": 16,
        "num_hidden_layers": hidden_layers,
        "intermediate_size": 32,
        "moe_intermediate_size": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts.min(2),
        "mova_num_experts": mova_num_experts,
        "mova_num_experts_per_tok": mova_num_experts.min(1),
        "num_shared_experts": 1,
        "decoder_sparse_step": 1,
        "mlp_only_layers": mlp_only_layers,
        "rms_norm_eps": 1e-6,
        "layernorm_num_groups": 2,
        "rope_parameters": { "rope_theta": 10_000.0, "rope_type": "default" },
        "attention_bias": false,
        "moe_gate_bias": true,
        "attention_gate_func": "softplus",
        "norm_topk_prob": true,
        "router_score_func": "sigmoid",
        "router_scaling_factor": 2.5,
        "tie_word_embeddings": false,
        "eos_token_id": [1, 2],
        "bos_token_id": 0,
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "mode": "affine"
        }
    })
    .to_string()
}

pub(super) fn write_stacked_affine_fixture(root: &Path) -> PathBuf {
    let model_directory = root.join("K2-Horizon-MoVA-Tiny-Fixture");
    fs::create_dir_all(&model_directory).expect("fixture directory should be created");
    let config_json = family_member_config_json(2, &[0], 4, 2);
    fs::write(model_directory.join("config.json"), &config_json)
        .expect("config.json should be written");
    let config = K2HorizonMoVAConfig::from_json_bytes(config_json.as_bytes())
        .expect("tiny family config should parse");
    let mut weight_map = BTreeMap::new();
    for tensor_name in astronomical_model_serving::expected_stacked_affine_tensor_names(&config) {
        weight_map.insert(tensor_name, "model-00001-of-00001.safetensors".to_owned());
    }
    let index = serde_json::json!({
        "metadata": { "total_size": 8 },
        "weight_map": weight_map
    });
    fs::write(
        model_directory.join("model.safetensors.index.json"),
        index.to_string(),
    )
    .expect("index should be written");
    fs::write(
        model_directory.join("model-00001-of-00001.safetensors"),
        b"fixture",
    )
    .expect("shard should be written");
    fs::write(
        model_directory.join("tokenizer.json"),
        b"{\"model\":{\"type\":\"BPE\"}}",
    )
    .expect("tokenizer should be written");
    fs::write(
        model_directory.join("chat_template.jinja"),
        b"{{ bos_token }}",
    )
    .expect("chat template should be written");
    model_directory
}
