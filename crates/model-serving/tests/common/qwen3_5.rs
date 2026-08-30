use astronomical_model_serving::Qwen3_5Config;
use serde_json::{Value, json};

use super::qwen3_5_moe::frozen_ornith_1_0_optiq_config_bytes;

pub(crate) fn frozen_dense_qwen3_6_config() -> Qwen3_5Config {
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
