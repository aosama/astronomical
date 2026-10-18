//! Private serde wire types for the folded-pack `config.json` document.
//!
//! Kept separate from the validated domain types so the strict validation
//! rules live in the configuration owner while the wire shape stays here.

use serde::Deserialize;

/// Decodes a JSON float into exact IEEE-754 bits without precision loss,
/// mirroring the bounded float convention used across the repository.
pub(super) fn deserialize_f32_bits<'de, DeserializerType>(
    deserializer: DeserializerType,
) -> Result<u32, DeserializerType::Error>
where
    DeserializerType: serde::Deserializer<'de>,
{
    let float_value = f32::deserialize(deserializer)?;
    Ok(float_value.to_bits())
}

/// Normalizes `eos_token_id` from either a single integer or an array.
pub(super) fn deserialize_optional_eos_token_ids<'de, DeserializerType>(
    deserializer: DeserializerType,
) -> Result<Option<Vec<u32>>, DeserializerType::Error>
where
    DeserializerType: serde::Deserializer<'de>,
{
    Option::<EosTokenId>::deserialize(deserializer).map(|optional_eos| {
        optional_eos.map(|eos_token_id_variant| match eos_token_id_variant {
            EosTokenId::Single(token_id) => vec![token_id],
            EosTokenId::Array(token_ids) => token_ids,
        })
    })
}

/// Intermediate enum for `eos_token_id` in either wire shape.
#[derive(Deserialize)]
#[serde(untagged)]
enum EosTokenId {
    Single(u32),
    Array(Vec<u32>),
}

/// The raw folded-pack document. Unknown fields are refused: a future pack
/// revision must be accepted deliberately, not silently tolerated.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FoldedPackDocument {
    pub(super) schema_version: u32,
    pub(super) model_type: String,
    pub(super) base_model_type: String,
    pub(super) requires_runtime: String,
    pub(super) hadamard_config: String,
    pub(super) tensor_namespace: String,
    pub(super) gdn_activation_layout: String,
    pub(super) components: FoldedComponentsDocument,
    pub(super) text_config: FoldedTextConfigDocument,
    pub(super) quantization: FoldedQuantizationDocument,
    pub(super) modules: Vec<PackedModuleDocument>,
    pub(super) tie_word_embeddings: bool,
    pub(super) vision_config: Option<serde_json::Value>,
    pub(super) image_token_id: Option<u32>,
    pub(super) video_token_id: Option<u32>,
    pub(super) vision_start_token_id: Option<u32>,
    pub(super) vision_end_token_id: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FoldedComponentsDocument {
    pub(super) text: bool,
    pub(super) vision: bool,
    pub(super) mtp: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FoldedQuantizationDocument {
    pub(super) bits: u32,
    pub(super) group_size: u32,
    pub(super) mode: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PackedModuleDocument {
    pub(super) path: String,
    pub(super) block: u32,
    pub(super) embedding: bool,
    pub(super) dtype: String,
}

/// The raw text-side configuration. Tolerant of unknown fields (mirroring the
/// core qwen3_5 text wire convention); every field listed here is validated
/// or consumed by the family.
#[derive(Debug, Deserialize)]
pub(super) struct FoldedTextConfigDocument {
    pub(super) model_type: String,
    pub(super) hidden_act: String,
    pub(super) hidden_size: u32,
    pub(super) intermediate_size: u32,
    pub(super) num_hidden_layers: u32,
    pub(super) num_attention_heads: u32,
    pub(super) num_key_value_heads: u32,
    pub(super) head_dim: u32,
    pub(super) vocab_size: u32,
    pub(super) max_position_embeddings: u32,
    pub(super) full_attention_interval: u32,
    pub(super) layer_types: Vec<String>,
    pub(super) linear_num_value_heads: u32,
    pub(super) linear_num_key_heads: u32,
    pub(super) linear_key_head_dim: u32,
    pub(super) linear_value_head_dim: u32,
    pub(super) linear_conv_kernel_dim: u32,
    #[serde(rename = "rms_norm_eps", deserialize_with = "deserialize_f32_bits")]
    pub(super) rms_norm_epsilon_bits: u32,
    pub(super) rope_parameters: RopeParametersDocument,
    #[serde(deserialize_with = "deserialize_optional_eos_token_ids", default)]
    pub(super) eos_token_id: Option<Vec<u32>>,
    pub(super) attention_bias: bool,
    pub(super) tie_word_embeddings: bool,
    pub(super) attn_output_gate: bool,
    pub(super) output_gate_type: String,
    pub(super) mamba_ssm_dtype: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RopeParametersDocument {
    #[serde(rename = "type", alias = "rope_type")]
    pub(super) rope_type: String,
    #[serde(rename = "rope_theta", deserialize_with = "deserialize_f32_bits")]
    pub(super) rope_theta_bits: u32,
    #[serde(
        rename = "partial_rotary_factor",
        deserialize_with = "deserialize_f32_bits"
    )]
    pub(super) partial_rotary_factor_bits: u32,
    pub(super) mrope_interleaved: bool,
    pub(super) mrope_section: Vec<u32>,
}
