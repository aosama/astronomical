//! Typed K2 Horizon MoVA configuration owned by serving.
//!
//! Counts, widths, and router knobs come from the artifact. This type must not
//! bake in one Hugging Face repository's 36B geometry.

use super::affine_profile::K2HorizonMoVAQuantizationContract;
use super::document::{K2HorizonMoVAConfigDocument, K2HorizonMoVATokenIdDocument};
use super::error::K2HorizonMoVAConfigError;
use super::{K2HorizonMoVAAffineProfile, K2HorizonMoVALayerKind};

const EXPECTED_MODEL_TYPE: &str = "k2_horizon_mova";
const EXPECTED_ARCHITECTURE: &str = "K2HorizonForCausalLM";
const EXPECTED_ROUTER_SCORE: &str = "sigmoid";

/// Validated family configuration used to derive layer schedule and tensor names.
#[derive(Clone, Debug, PartialEq)]
pub struct K2HorizonMoVAConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    moe_intermediate_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rope_head_dim: usize,
    vocab_size: u32,
    max_position_embeddings: u32,
    num_experts: usize,
    num_experts_per_tok: usize,
    mova_num_experts: usize,
    mova_num_experts_per_tok: usize,
    num_shared_experts: usize,
    decoder_sparse_step: usize,
    mlp_only_layers: Vec<usize>,
    rms_norm_eps: f32,
    layernorm_num_groups: usize,
    rope_theta: f32,
    attention_bias: bool,
    moe_gate_bias: bool,
    attention_gate_func: Option<K2HorizonMoVAAttentionGateFunc>,
    query_key_norm: bool,
    norm_topk_prob: bool,
    router_scaling_factor: f32,
    tie_word_embeddings: bool,
    eos_token_ids: Vec<u32>,
    bos_token_id: u32,
    quantization: K2HorizonMoVAQuantizationContract,
}

/// Attention-gate nonlinearity declared by the artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum K2HorizonMoVAAttentionGateFunc {
    Silu,
    Softplus,
}

impl K2HorizonMoVAConfig {
    /// Parses retained `config.json` bytes into a family contract.
    pub fn from_json_bytes(config_bytes: &[u8]) -> Result<Self, K2HorizonMoVAConfigError> {
        let document: K2HorizonMoVAConfigDocument = serde_json::from_slice(config_bytes)
            .map_err(|source| K2HorizonMoVAConfigError::DeserializeConfig { source })?;
        if document.model_type != EXPECTED_MODEL_TYPE {
            return Err(K2HorizonMoVAConfigError::InvalidConfigValue {
                description: format!("model_type must be '{EXPECTED_MODEL_TYPE}'"),
            });
        }
        if !document
            .architectures
            .iter()
            .any(|architecture| architecture == EXPECTED_ARCHITECTURE)
        {
            return Err(K2HorizonMoVAConfigError::InvalidConfigValue {
                description: format!("architectures must include '{EXPECTED_ARCHITECTURE}'"),
            });
        }
        if document.router_score_func != EXPECTED_ROUTER_SCORE {
            return Err(K2HorizonMoVAConfigError::InvalidConfigValue {
                description: "router_score_func must be sigmoid".to_owned(),
            });
        }
        validate_positive("hidden_size", document.hidden_size)?;
        validate_positive("num_hidden_layers", document.num_hidden_layers)?;
        validate_positive("num_attention_heads", document.num_attention_heads)?;
        validate_positive("num_key_value_heads", document.num_key_value_heads)?;
        validate_positive("head_dim", document.head_dim)?;
        if document.num_attention_heads % document.num_key_value_heads != 0 {
            return Err(K2HorizonMoVAConfigError::InvalidConfigValue {
                description: "num_attention_heads must be a multiple of num_key_value_heads"
                    .to_owned(),
            });
        }
        if document.hidden_size % document.layernorm_num_groups != 0 {
            return Err(K2HorizonMoVAConfigError::InvalidConfigValue {
                description: "hidden_size must be divisible by layernorm_num_groups".to_owned(),
            });
        }
        if document.max_position_embeddings < 2 {
            return Err(K2HorizonMoVAConfigError::InvalidConfigValue {
                description: "max_position_embeddings must be at least 2".to_owned(),
            });
        }
        if document.num_experts_per_tok > document.num_experts && document.num_experts > 0 {
            return Err(K2HorizonMoVAConfigError::InvalidConfigValue {
                description: "num_experts_per_tok cannot exceed num_experts".to_owned(),
            });
        }
        if document.mova_num_experts_per_tok > document.mova_num_experts
            && document.mova_num_experts > 0
        {
            return Err(K2HorizonMoVAConfigError::InvalidConfigValue {
                description: "mova_num_experts_per_tok cannot exceed mova_num_experts".to_owned(),
            });
        }
        let attention_gate_func = match document.attention_gate_func.as_deref() {
            None => None,
            Some("silu") => Some(K2HorizonMoVAAttentionGateFunc::Silu),
            Some("softplus") => Some(K2HorizonMoVAAttentionGateFunc::Softplus),
            Some(other) => {
                return Err(K2HorizonMoVAConfigError::InvalidConfigValue {
                    description: format!("unsupported attention_gate_func '{other}'"),
                });
            }
        };
        if let Some(rope_type) = document
            .rope_parameters
            .as_ref()
            .and_then(|parameters| parameters.rope_type.as_deref())
            && rope_type != "default"
        {
            return Err(K2HorizonMoVAConfigError::InvalidConfigValue {
                description: format!("unsupported rope_type '{rope_type}'"),
            });
        }
        let rope_theta = document
            .rope_parameters
            .as_ref()
            .and_then(|parameters| parameters.rope_theta)
            .or(document.rope_theta)
            .unwrap_or(10_000_000.0);
        let quantization_document = document
            .quantization
            .as_ref()
            .or(document.quantization_config.as_ref())
            .ok_or(K2HorizonMoVAConfigError::InvalidConfigValue {
                description: "stacked affine serving requires a quantization document".to_owned(),
            })?;
        let quantization = K2HorizonMoVAQuantizationContract::from_document(quantization_document)?;
        let eos_token_ids = match document.eos_token_id {
            Some(K2HorizonMoVATokenIdDocument::One(token_id)) => vec![token_id],
            Some(K2HorizonMoVATokenIdDocument::Many(token_ids)) if !token_ids.is_empty() => {
                token_ids
            }
            _ => {
                return Err(K2HorizonMoVAConfigError::InvalidConfigValue {
                    description: "eos_token_id must contain at least one token".to_owned(),
                });
            }
        };
        Ok(Self {
            hidden_size: document.hidden_size,
            num_hidden_layers: document.num_hidden_layers,
            intermediate_size: document.intermediate_size,
            moe_intermediate_size: document.moe_intermediate_size,
            num_attention_heads: document.num_attention_heads,
            num_key_value_heads: document.num_key_value_heads,
            head_dim: document.head_dim,
            rope_head_dim: document.rope_head_dim.unwrap_or(document.head_dim),
            vocab_size: document.vocab_size,
            max_position_embeddings: document.max_position_embeddings,
            num_experts: document.num_experts,
            num_experts_per_tok: document.num_experts_per_tok,
            mova_num_experts: document.mova_num_experts,
            mova_num_experts_per_tok: document.mova_num_experts_per_tok,
            num_shared_experts: document.num_shared_experts,
            decoder_sparse_step: document.decoder_sparse_step,
            mlp_only_layers: document.mlp_only_layers,
            rms_norm_eps: document.rms_norm_eps,
            layernorm_num_groups: document.layernorm_num_groups,
            rope_theta,
            attention_bias: document.attention_bias,
            moe_gate_bias: document.moe_gate_bias,
            attention_gate_func,
            query_key_norm: document.query_key_norm,
            norm_topk_prob: document.norm_topk_prob,
            router_scaling_factor: document.router_scaling_factor,
            tie_word_embeddings: document.tie_word_embeddings,
            eos_token_ids,
            bos_token_id: document.bos_token_id.unwrap_or(0),
            quantization,
        })
    }

    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    #[must_use]
    pub const fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }
    #[must_use]
    pub const fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
    #[must_use]
    pub const fn moe_intermediate_size(&self) -> usize {
        self.moe_intermediate_size
    }
    #[must_use]
    pub const fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }
    #[must_use]
    pub const fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }
    #[must_use]
    pub const fn rope_head_dim(&self) -> usize {
        self.rope_head_dim
    }
    #[must_use]
    pub const fn vocab_size(&self) -> u32 {
        self.vocab_size
    }
    #[must_use]
    pub const fn max_position_embeddings(&self) -> u32 {
        self.max_position_embeddings
    }
    #[must_use]
    pub const fn num_experts(&self) -> usize {
        self.num_experts
    }
    #[must_use]
    pub const fn num_experts_per_tok(&self) -> usize {
        self.num_experts_per_tok
    }
    #[must_use]
    pub const fn mova_num_experts(&self) -> usize {
        self.mova_num_experts
    }
    #[must_use]
    pub const fn mova_num_experts_per_tok(&self) -> usize {
        self.mova_num_experts_per_tok
    }
    #[must_use]
    pub const fn num_shared_experts(&self) -> usize {
        self.num_shared_experts
    }
    #[must_use]
    pub const fn decoder_sparse_step(&self) -> usize {
        self.decoder_sparse_step
    }
    #[must_use]
    pub fn mlp_only_layers(&self) -> &[usize] {
        &self.mlp_only_layers
    }
    #[must_use]
    pub const fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }
    #[must_use]
    pub const fn layernorm_num_groups(&self) -> usize {
        self.layernorm_num_groups
    }
    #[must_use]
    pub const fn rope_theta(&self) -> f32 {
        self.rope_theta
    }
    #[must_use]
    pub const fn attention_bias(&self) -> bool {
        self.attention_bias
    }
    #[must_use]
    pub const fn moe_gate_bias(&self) -> bool {
        self.moe_gate_bias
    }
    #[must_use]
    pub const fn attention_gate_func(&self) -> Option<K2HorizonMoVAAttentionGateFunc> {
        self.attention_gate_func
    }
    #[must_use]
    pub const fn query_key_norm(&self) -> bool {
        self.query_key_norm
    }
    #[must_use]
    pub const fn norm_topk_prob(&self) -> bool {
        self.norm_topk_prob
    }
    #[must_use]
    pub const fn router_scaling_factor(&self) -> f32 {
        self.router_scaling_factor
    }
    #[must_use]
    pub const fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }
    #[must_use]
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }
    #[must_use]
    pub const fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }
    #[must_use]
    pub fn quantization(&self) -> &K2HorizonMoVAQuantizationContract {
        &self.quantization
    }
    #[must_use]
    pub fn affine_profile_for_module(&self, module_path: &str) -> K2HorizonMoVAAffineProfile {
        self.quantization.profile_for_module(module_path)
    }
    #[must_use]
    pub fn layer_kinds(&self) -> Vec<K2HorizonMoVALayerKind> {
        (0..self.num_hidden_layers)
            .map(|decoder_layer_index| self.layer_kind(decoder_layer_index))
            .collect()
    }
}

fn validate_positive(field_name: &str, value: usize) -> Result<(), K2HorizonMoVAConfigError> {
    if value == 0 {
        return Err(K2HorizonMoVAConfigError::InvalidConfigValue {
            description: format!("{field_name} must be positive"),
        });
    }
    Ok(())
}
