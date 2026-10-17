use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use astronomical_model_serving::K2HorizonMoVAConfig;

pub(super) use astronomical_test_harness::fixtures::family_member_config_json;

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
