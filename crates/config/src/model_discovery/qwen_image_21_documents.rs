//! Wire documents for the reviewed Qwen-Image-2.1 MLX package.
//!
//! Field names intentionally mirror the Hugging Face serialization, including its upstream
//! `temperal_downsample` spelling. Discovery owns profile policy; this file owns only JSON shape.

use std::collections::BTreeMap;

use serde::Deserialize;

#[derive(Deserialize)]
pub(super) struct PipelineClass {
    #[serde(default, rename = "_class_name")]
    pub(super) class_name: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct PipelineIndex {
    #[serde(rename = "_class_name")]
    pub(super) class_name: String,
    pub(super) processor: [String; 2],
    pub(super) scheduler: [String; 2],
    pub(super) text_encoder: [String; 2],
    pub(super) transformer: [String; 2],
    pub(super) vae: [String; 2],
}

#[derive(Deserialize)]
pub(super) struct QuantizationGeometry {
    pub(super) bits: u8,
    pub(super) group_size: u16,
    pub(super) mode: String,
}

#[derive(Deserialize)]
pub(super) struct QwenImage21TransformerGeometry {
    #[serde(rename = "_class_name")]
    pub(super) class_name: String,
    pub(super) attention_head_dim: u32,
    pub(super) axes_dims_rope: [u32; 3],
    pub(super) context_in_dim: u32,
    pub(super) in_channels: u32,
    pub(super) num_attention_heads: u32,
    pub(super) num_layers: u32,
    pub(super) out_channels: u32,
    pub(super) patch_size: u32,
    pub(super) mlp_ratio: u32,
    pub(super) eps: f64,
    pub(super) causal_condition: bool,
    pub(super) quantization: QuantizationGeometry,
    pub(super) mlx_format: bool,
}

#[derive(Deserialize)]
pub(super) struct QwenImage21TextEncoderGeometry {
    pub(super) architectures: [String; 1],
    pub(super) dtype: String,
    pub(super) model_type: String,
    pub(super) tie_word_embeddings: bool,
    pub(super) text_config: Qwen3VlTextGeometry,
    pub(super) quantization: QuantizationGeometry,
    pub(super) mlx_format: bool,
}

#[derive(Deserialize)]
pub(super) struct Qwen3VlTextGeometry {
    pub(super) attention_bias: bool,
    pub(super) attention_dropout: f64,
    pub(super) dtype: String,
    pub(super) head_dim: u32,
    pub(super) hidden_act: String,
    pub(super) hidden_size: u32,
    pub(super) intermediate_size: u32,
    pub(super) max_position_embeddings: u32,
    pub(super) model_type: String,
    pub(super) num_attention_heads: u32,
    pub(super) num_hidden_layers: u32,
    pub(super) num_key_value_heads: u32,
    pub(super) rms_norm_eps: f64,
    pub(super) rope_scaling: RopeScalingGeometry,
    pub(super) rope_theta: u64,
    pub(super) use_cache: bool,
    pub(super) vocab_size: u32,
}

#[derive(Deserialize)]
pub(super) struct RopeScalingGeometry {
    pub(super) mrope_interleaved: bool,
    pub(super) mrope_section: [u32; 3],
    pub(super) rope_type: String,
}

#[derive(Deserialize)]
pub(super) struct QwenImage21VaeGeometry {
    #[serde(rename = "_class_name")]
    pub(super) class_name: String,
    pub(super) attn_scales: Vec<u32>,
    pub(super) base_dim: u32,
    pub(super) decoder_base_dim: u32,
    pub(super) dim_mult: [u32; 5],
    pub(super) dropout: f64,
    pub(super) in_channels: u32,
    pub(super) is_residual: bool,
    pub(super) latents_mean: Vec<f64>,
    pub(super) latents_std: Vec<f64>,
    pub(super) num_res_blocks: u32,
    pub(super) out_channels: u32,
    pub(super) patch_size: Option<u32>,
    pub(super) scale_factor_spatial: u32,
    pub(super) scale_factor_temporal: u32,
    /// The upstream serialization uses this non-standard spelling. The reviewed MLX export
    /// carries one flag per down block boundary (four), not one per `dim_mult` stage.
    #[serde(rename = "temperal_downsample")]
    pub(super) temporal_downsample: [bool; 4],
    pub(super) z_dim: u32,
    pub(super) mlx_format: bool,
}

#[derive(Deserialize)]
pub(super) struct QwenImage21SchedulerGeometry {
    #[serde(rename = "_class_name")]
    pub(super) class_name: String,
    pub(super) base_image_seq_len: u32,
    pub(super) base_shift: f64,
    pub(super) invert_sigmas: bool,
    pub(super) max_image_seq_len: u32,
    pub(super) max_shift: f64,
    pub(super) num_train_timesteps: u32,
    pub(super) shift: f64,
    pub(super) shift_terminal: Option<f64>,
    pub(super) stochastic_sampling: bool,
    pub(super) time_shift_type: String,
    pub(super) use_beta_sigmas: bool,
    pub(super) use_dynamic_shifting: bool,
    pub(super) use_exponential_sigmas: bool,
    pub(super) use_karras_sigmas: bool,
}

#[derive(Deserialize)]
pub(super) struct ComponentSafetensorsIndex {
    pub(super) metadata: ComponentSafetensorsIndexMetadata,
    pub(super) weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
pub(super) struct ComponentSafetensorsIndexMetadata {
    pub(super) total_size: u64,
}
