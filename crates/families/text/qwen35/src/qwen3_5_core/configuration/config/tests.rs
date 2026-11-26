//! Unit contracts for Qwen3.5 configuration parsing and validation.
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
use crate::qwen3_5_core::quantizations::producers::optiq::{OptiQMetadata, OptiQMetadataError};

use super::{ModelWeightStorage, Qwen3_5Config};
use crate::qwen3_5_core::configuration::Qwen3_5ConfigError;

mod compatibility_fallbacks_tests;
mod dtype_profile_tests;
mod model_contract_tests;
mod optiq_metadata_tests;
mod quantization_tests;

/// The frozen Ornith 1.0 config written the way newer conversions write it:
/// `dtype` rather than `torch_dtype`, the rope settings under
/// `rope_parameters` rather than a bare `rope_theta`, and no `pad_token_id`.
/// Deriving it from the one frozen document keeps every other field from
/// drifting, while these contracts still parse the newer spellings.
pub fn minimal_valid_config_json() -> Value {
    let mut config_value = serde_json::from_slice::<Value>(&frozen_ornith_1_0_config_bytes())
        .expect("the frozen Ornith 1.0 config should decode as JSON");
    let config_fields = config_value
        .as_object_mut()
        .expect("the frozen Ornith 1.0 config should be a JSON object");
    let torch_dtype = config_fields
        .remove("torch_dtype")
        .expect("the frozen Ornith 1.0 config names its torch_dtype");
    config_fields.insert("dtype".to_owned(), torch_dtype);
    config_fields.remove("pad_token_id");
    let text_config_fields = config_value["text_config"]
        .as_object_mut()
        .expect("the frozen Ornith 1.0 text_config should be a JSON object");
    let rope_theta = text_config_fields
        .remove("rope_theta")
        .expect("the frozen Ornith 1.0 text_config names its rope_theta");
    text_config_fields.insert(
        "rope_parameters".to_owned(),
        json!({
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": rope_theta,
            "type": "default"
        }),
    );
    config_value["text_config"]["layer_types"] = json!(decoder_attention_schedule(40));
    let quantization = uniform_four_bit_quantization_with_eight_bit_embeddings();
    config_value["quantization"] = quantization.clone();
    config_value["quantization_config"] = quantization;
    config_value
}

/// Every fourth decoder layer attends over the full context; the rest are
/// linear attention, as in the Qwen 3.5 MoE layout.
fn decoder_attention_schedule(layer_count: usize) -> Vec<&'static str> {
    let mut layer_types = Vec::with_capacity(layer_count);
    for layer_position in 0..layer_count {
        if layer_position % 4 == 3 {
            layer_types.push("full_attention");
        } else {
            layer_types.push("linear_attention");
        }
    }
    layer_types
}

/// The parser requires per-layer overrides, so the embedding and output
/// projections carry their own 8-bit entries beside the 4-bit default.
fn uniform_four_bit_quantization_with_eight_bit_embeddings() -> Value {
    let mut quantization = json!({"group_size": 64, "bits": 4, "mode": "affine"});
    quantization["language_model.model.embed_tokens"] = json!({"group_size": 64, "bits": 8});
    quantization["language_model.lm_head"] = json!({"group_size": 64, "bits": 8});
    quantization
}
