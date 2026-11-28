//! A dense multi-token-prediction head configuration for tests that parse
//! or locate one.

use serde_json::{Value, json};

use super::Qwen3_5Config;

/// A head configuration shaped like Ternary Bonsai 2 9B's `mtp/config.json`,
/// sized to the model it sits beside.
pub fn dense_mtp_head_config_document(model_config: &Qwen3_5Config) -> Value {
    json!({
        "block_size": 2,
        "model_type": "qwen3_5_mtp",
        "text_config": {
            "hidden_size": model_config.hidden_size(),
            "vocab_size": model_config.vocabulary_size(),
            "num_attention_heads": model_config.query_head_count(),
            "num_key_value_heads": model_config.key_value_head_count(),
            "head_dim": model_config.head_dimension(),
            "intermediate_size": model_config.dense_intermediate_size(),
            "mtp_num_hidden_layers": 0,
            "mtp_use_dedicated_embeddings": false,
        },
        "quantization": {"group_size": 64, "bits": 4},
    })
}
