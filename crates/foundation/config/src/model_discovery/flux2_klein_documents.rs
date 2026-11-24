//! Strict deserialization documents for reviewed FLUX.2 Klein package evidence.
//!
//! Keeping wire-shaped JSON documents separate leaves discovery focused on validation policy and
//! filesystem evidence rather than serialization mechanics.

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
    pub(super) is_distilled: bool,
    pub(super) scheduler: [String; 2],
    pub(super) text_encoder: [String; 2],
    pub(super) tokenizer: [String; 2],
    pub(super) transformer: [String; 2],
    pub(super) vae: [String; 2],
}

#[derive(Deserialize)]
pub(super) struct TransformerGeometry {
    #[serde(rename = "_class_name")]
    pub(super) class_name: String,
    pub(super) attention_head_dim: u32,
    pub(super) axes_dims_rope: [u32; 4],
    pub(super) eps: f64,
    pub(super) guidance_embeds: bool,
    pub(super) in_channels: u32,
    pub(super) joint_attention_dim: u32,
    pub(super) mlp_ratio: f64,
    pub(super) num_attention_heads: u32,
    pub(super) num_layers: u32,
    pub(super) num_single_layers: u32,
    pub(super) out_channels: Option<u32>,
    pub(super) patch_size: u32,
    pub(super) rope_theta: f64,
    pub(super) timestep_guidance_channels: u32,
}

#[derive(Deserialize)]
pub(super) struct TextEncoderGeometry {
    pub(super) architectures: [String; 1],
    pub(super) attention_bias: bool,
    pub(super) attention_dropout: f64,
    pub(super) dtype: String,
    pub(super) head_dim: u32,
    pub(super) hidden_act: String,
    pub(super) hidden_size: u32,
    pub(super) intermediate_size: u32,
    pub(super) layer_types: Vec<String>,
    pub(super) max_position_embeddings: u32,
    pub(super) max_window_layers: u32,
    pub(super) model_type: String,
    pub(super) num_attention_heads: u32,
    pub(super) num_hidden_layers: u32,
    pub(super) num_key_value_heads: u32,
    pub(super) rms_norm_eps: f64,
    pub(super) rope_scaling: Option<ValueMarker>,
    pub(super) rope_theta: f64,
    pub(super) sliding_window: Option<u32>,
    pub(super) tie_word_embeddings: bool,
    pub(super) use_cache: bool,
    pub(super) use_sliding_window: bool,
    pub(super) vocab_size: u32,
}

#[derive(Deserialize)]
pub(super) struct VaeGeometry {
    #[serde(rename = "_class_name")]
    pub(super) class_name: String,
    pub(super) act_fn: String,
    pub(super) batch_norm_eps: f64,
    pub(super) batch_norm_momentum: f64,
    pub(super) block_out_channels: [u32; 4],
    pub(super) down_block_types: [String; 4],
    pub(super) force_upcast: bool,
    pub(super) in_channels: u32,
    pub(super) latent_channels: u32,
    pub(super) layers_per_block: u32,
    pub(super) mid_block_add_attention: bool,
    pub(super) norm_num_groups: u32,
    pub(super) out_channels: u32,
    pub(super) patch_size: [u32; 2],
    pub(super) sample_size: u32,
    pub(super) up_block_types: [String; 4],
    pub(super) use_post_quant_conv: bool,
    pub(super) use_quant_conv: bool,
}

#[derive(Deserialize)]
pub(super) struct SchedulerGeometry {
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
pub(super) struct TextEncoderIndex {
    pub(super) metadata: TextEncoderIndexMetadata,
    pub(super) weight_map: std::collections::BTreeMap<String, String>,
}

#[derive(Deserialize)]
pub(super) struct TextEncoderIndexMetadata {
    pub(super) total_size: u64,
}

#[derive(Deserialize)]
pub(super) struct ValueMarker;
