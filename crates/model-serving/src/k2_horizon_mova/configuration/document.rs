//! Loose JSON projection for K2 Horizon MoVA `config.json`.
//!
//! Extra keys are ignored so one packaging variant cannot block another family
//! member. Known knobs stay typed.

use std::collections::BTreeMap;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(super) struct K2HorizonMoVAConfigDocument {
    pub(super) model_type: String,
    #[serde(default)]
    pub(super) architectures: Vec<String>,
    pub(super) hidden_size: usize,
    pub(super) num_hidden_layers: usize,
    pub(super) intermediate_size: usize,
    pub(super) moe_intermediate_size: usize,
    pub(super) num_attention_heads: usize,
    pub(super) num_key_value_heads: usize,
    pub(super) head_dim: usize,
    pub(super) vocab_size: u32,
    pub(super) max_position_embeddings: u32,
    pub(super) num_experts: usize,
    pub(super) num_experts_per_tok: usize,
    pub(super) mova_num_experts: usize,
    pub(super) mova_num_experts_per_tok: usize,
    #[serde(default = "default_shared_expert_count")]
    pub(super) num_shared_experts: usize,
    #[serde(default = "default_decoder_sparse_step")]
    pub(super) decoder_sparse_step: usize,
    #[serde(default)]
    pub(super) mlp_only_layers: Vec<usize>,
    pub(super) rms_norm_eps: f32,
    #[serde(default = "default_layernorm_group_count")]
    pub(super) layernorm_num_groups: usize,
    #[serde(default)]
    pub(super) rope_theta: Option<f32>,
    #[serde(default)]
    pub(super) rope_parameters: Option<K2HorizonMoVARopeParametersDocument>,
    #[serde(default)]
    pub(super) rope_head_dim: Option<usize>,
    #[serde(default)]
    pub(super) attention_bias: bool,
    #[serde(default = "default_true")]
    pub(super) moe_gate_bias: bool,
    #[serde(default)]
    pub(super) attention_gate_func: Option<String>,
    #[serde(default)]
    pub(super) query_key_norm: bool,
    #[serde(default = "default_true")]
    pub(super) norm_topk_prob: bool,
    #[serde(default = "default_sigmoid")]
    pub(super) router_score_func: String,
    #[serde(default = "default_router_scaling_factor")]
    pub(super) router_scaling_factor: f32,
    #[serde(default)]
    pub(super) tie_word_embeddings: bool,
    #[serde(default)]
    pub(super) eos_token_id: Option<K2HorizonMoVATokenIdDocument>,
    #[serde(default)]
    pub(super) bos_token_id: Option<u32>,
    #[serde(default)]
    pub(super) quantization: Option<K2HorizonMoVAQuantizationDocument>,
    #[serde(default)]
    pub(super) quantization_config: Option<K2HorizonMoVAQuantizationDocument>,
}

#[derive(Debug, Deserialize)]
pub(super) struct K2HorizonMoVARopeParametersDocument {
    #[serde(default)]
    pub(super) rope_theta: Option<f32>,
    #[serde(default)]
    pub(super) rope_type: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(super) enum K2HorizonMoVATokenIdDocument {
    One(u32),
    Many(Vec<u32>),
}

#[derive(Debug, Deserialize)]
pub(super) struct K2HorizonMoVAQuantizationDocument {
    pub(super) group_size: u32,
    pub(super) bits: u32,
    pub(super) mode: String,
    #[serde(flatten)]
    pub(super) module_overrides: BTreeMap<String, serde_json::Value>,
}

fn default_shared_expert_count() -> usize {
    1
}

fn default_decoder_sparse_step() -> usize {
    1
}

fn default_layernorm_group_count() -> usize {
    2
}

fn default_true() -> bool {
    true
}

fn default_sigmoid() -> String {
    "sigmoid".to_owned()
}

fn default_router_scaling_factor() -> f32 {
    2.5
}
