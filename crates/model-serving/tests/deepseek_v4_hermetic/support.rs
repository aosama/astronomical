use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

const PRIMARY_SHARD_FILE_NAME: &str = "model-00001-of-00002.safetensors";
const SECONDARY_SHARD_FILE_NAME: &str = "model-00002-of-00002.safetensors";

pub(crate) fn selected_config_value(dspark_enabled: bool) -> Value {
    let mut config_value = json!({
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "vocab_size": 129280,
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "num_hash_layers": 3,
        "quantization": {"bits": 8, "group_size": 64, "mode": "affine"}
    });
    if dspark_enabled {
        config_value["dspark_block_size"] = json!(5);
        config_value["dspark_target_layer_ids"] = json!([40, 41, 42]);
        config_value["dspark_noise_token_id"] = json!(128799);
        config_value["dspark_markov_rank"] = json!(256);
    }
    config_value
}

pub(crate) fn write_artifact(
    temporary_directory: &tempfile::TempDir,
    dspark_enabled: bool,
    extra_header_tensor_name: Option<&str>,
    omitted_secondary_header_tensor_name: Option<&str>,
) -> PathBuf {
    let artifact_directory = temporary_directory
        .path()
        .join("DeepSeek-V4-Flash-0731-oQ8e-mtp");
    std::fs::create_dir_all(&artifact_directory)
        .expect("the test should create the DeepSeek artifact directory");
    std::fs::write(
        artifact_directory.join("config.json"),
        serde_json::to_vec(&selected_config_value(dspark_enabled))
            .expect("the test config should serialize"),
    )
    .expect("the test should write config.json");
    std::fs::write(
        artifact_directory.join("tokenizer.json"),
        br#"{"version":"1.0"}"#,
    )
    .expect("the test should write tokenizer.json");

    let mut tensor_name_to_shard_file_name = BTreeMap::from([
        (
            "model.embed_tokens.weight".to_owned(),
            PRIMARY_SHARD_FILE_NAME.to_owned(),
        ),
        (
            "model.norm.weight".to_owned(),
            PRIMARY_SHARD_FILE_NAME.to_owned(),
        ),
        (
            "lm_head.weight".to_owned(),
            PRIMARY_SHARD_FILE_NAME.to_owned(),
        ),
    ]);
    for layer_index in 0..43 {
        let shard_file_name = if layer_index % 2 == 0 {
            PRIMARY_SHARD_FILE_NAME
        } else {
            SECONDARY_SHARD_FILE_NAME
        };
        tensor_name_to_shard_file_name.insert(
            format!("model.layers.{layer_index}.attn.wq_a.weight"),
            shard_file_name.to_owned(),
        );
    }
    if dspark_enabled {
        for tensor_name in dspark_tensor_names() {
            tensor_name_to_shard_file_name
                .insert(tensor_name.to_owned(), SECONDARY_SHARD_FILE_NAME.to_owned());
        }
    }
    std::fs::write(
        artifact_directory.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({
            "metadata": {"total_size": 0},
            "weight_map": tensor_name_to_shard_file_name,
        }))
        .expect("the test shard index should serialize"),
    )
    .expect("the test should write the shard index");

    let mut primary_tensor_names = vec![
        "model.embed_tokens.weight".to_owned(),
        "model.norm.weight".to_owned(),
        "lm_head.weight".to_owned(),
    ];
    let mut secondary_tensor_names = Vec::new();
    for layer_index in 0..43 {
        let tensor_name = format!("model.layers.{layer_index}.attn.wq_a.weight");
        if layer_index % 2 == 0 {
            primary_tensor_names.push(tensor_name);
        } else {
            secondary_tensor_names.push(tensor_name);
        }
    }
    if dspark_enabled {
        secondary_tensor_names.extend(dspark_tensor_names().into_iter().map(str::to_owned));
    }
    if let Some(extra_header_tensor_name) = extra_header_tensor_name {
        secondary_tensor_names.push(extra_header_tensor_name.to_owned());
    }
    if let Some(omitted_secondary_header_tensor_name) = omitted_secondary_header_tensor_name {
        secondary_tensor_names
            .retain(|tensor_name| tensor_name != omitted_secondary_header_tensor_name);
    }
    write_safetensors_file(
        &artifact_directory.join(PRIMARY_SHARD_FILE_NAME),
        &primary_tensor_names,
    );
    write_safetensors_file(
        &artifact_directory.join(SECONDARY_SHARD_FILE_NAME),
        &secondary_tensor_names,
    );
    artifact_directory
}

pub(crate) fn remove_index_tensor(artifact_directory: &Path, tensor_name: &str) {
    let index_path = artifact_directory.join("model.safetensors.index.json");
    let mut index_document: Value = serde_json::from_slice(
        &std::fs::read(&index_path).expect("the test should read the shard index"),
    )
    .expect("the test shard index should parse");
    index_document["weight_map"]
        .as_object_mut()
        .expect("the test shard index should have a weight map")
        .remove(tensor_name);
    std::fs::write(
        index_path,
        serde_json::to_vec(&index_document).expect("the modified shard index should serialize"),
    )
    .expect("the test should rewrite the shard index");
}

pub(crate) fn add_unknown_index_tensor(artifact_directory: &Path) {
    let index_path = artifact_directory.join("model.safetensors.index.json");
    let mut index_document: Value = serde_json::from_slice(
        &std::fs::read(&index_path).expect("the test should read the shard index"),
    )
    .expect("the test shard index should parse");
    index_document["weight_map"]
        .as_object_mut()
        .expect("the test shard index should have a weight map")
        .insert(
            "unexpected.tensor".to_owned(),
            json!(PRIMARY_SHARD_FILE_NAME),
        );
    std::fs::write(
        index_path,
        serde_json::to_vec(&index_document).expect("the modified shard index should serialize"),
    )
    .expect("the test should rewrite the shard index");
}

pub(crate) fn add_unsupported_dspark_stage_tensor(artifact_directory: &Path) {
    let index_path = artifact_directory.join("model.safetensors.index.json");
    let mut index_document: Value = serde_json::from_slice(
        &std::fs::read(&index_path).expect("the test should read the shard index"),
    )
    .expect("the test shard index should parse");
    index_document["weight_map"]
        .as_object_mut()
        .expect("the test shard index should have a weight map")
        .insert(
            "mtp.3.attn.wq_a.weight".to_owned(),
            json!(SECONDARY_SHARD_FILE_NAME),
        );
    std::fs::write(
        index_path,
        serde_json::to_vec(&index_document).expect("the modified shard index should serialize"),
    )
    .expect("the test should rewrite the shard index");
}

pub(crate) fn secondary_shard_path(artifact_directory: &Path) -> PathBuf {
    artifact_directory.join(SECONDARY_SHARD_FILE_NAME)
}

fn dspark_tensor_names() -> Vec<&'static str> {
    vec![
        "mtp.0.attn.wq_a.weight",
        "mtp.0.ffn.gate.weight",
        "mtp.0.main_proj.weight",
        "mtp.0.main_norm.weight",
        "mtp.1.attn.wq_a.weight",
        "mtp.1.ffn.gate.weight",
        "mtp.2.attn.wq_a.weight",
        "mtp.2.ffn.gate.weight",
        "mtp.2.norm.weight",
        "mtp.2.markov_head.markov_w1.weight",
        "mtp.2.markov_head.markov_w2.weight",
        "mtp.2.confidence_head.proj.weight",
    ]
}

fn write_safetensors_file(file_path: &Path, tensor_names: &[String]) {
    let mut tensor_headers = BTreeMap::new();
    let mut next_data_offset_bytes = 0_u64;
    for tensor_name in tensor_names {
        let tensor_end_offset_bytes = next_data_offset_bytes + 2;
        tensor_headers.insert(
            tensor_name,
            json!({
                "dtype": "BF16",
                "shape": [1],
                "data_offsets": [next_data_offset_bytes, tensor_end_offset_bytes],
            }),
        );
        next_data_offset_bytes = tensor_end_offset_bytes;
    }
    let header_bytes = serde_json::to_vec(&tensor_headers)
        .expect("the synthetic safetensors header should serialize");
    let mut safetensors_file = std::fs::File::create(file_path)
        .expect("the test should create a synthetic safetensors shard");
    safetensors_file
        .write_all(&(header_bytes.len() as u64).to_le_bytes())
        .expect("the test should write the safetensors header length");
    safetensors_file
        .write_all(&header_bytes)
        .expect("the test should write the safetensors header");
    safetensors_file
        .write_all(&vec![0_u8; next_data_offset_bytes as usize])
        .expect("the test should write the safetensors payload");
}
