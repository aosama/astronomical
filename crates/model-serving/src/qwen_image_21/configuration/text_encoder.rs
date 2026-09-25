//! `text_encoder/config.json`: the Qwen3-VL text model's reviewed geometry, including the
//! vision tower's declared shape (the text-only path never loads the tower, but the package
//! still has to be the reviewed one).

use serde::Deserialize;

use super::{
    QuantizationDocument, QwenImage21ConfigError, parse_document, require, reviewed_quantization,
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TextConfigDocument {
    attention_bias: bool,
    attention_dropout: f64,
    bos_token_id: u32,
    dtype: String,
    eos_token_id: u32,
    head_dim: usize,
    hidden_act: String,
    hidden_size: usize,
    initializer_range: f64,
    intermediate_size: usize,
    max_position_embeddings: usize,
    model_type: String,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f64,
    rope_scaling: RopeScalingDocument,
    rope_theta: u64,
    use_cache: bool,
    vocab_size: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RopeScalingDocument {
    mrope_interleaved: bool,
    mrope_section: [usize; 3],
    rope_type: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VisionConfigDocument {
    deepstack_visual_indexes: [u32; 3],
    depth: usize,
    dtype: String,
    hidden_act: String,
    hidden_size: usize,
    in_channels: u32,
    initializer_range: f64,
    intermediate_size: usize,
    model_type: String,
    num_heads: usize,
    num_position_embeddings: usize,
    out_hidden_size: usize,
    patch_size: u32,
    spatial_merge_size: u32,
    temporal_patch_size: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TextEncoderDocument {
    architectures: [String; 1],
    dtype: String,
    image_token_id: u32,
    model_type: String,
    text_config: TextConfigDocument,
    tie_word_embeddings: bool,
    transformers_version: String,
    video_token_id: u32,
    vision_config: VisionConfigDocument,
    vision_end_token_id: u32,
    vision_start_token_id: u32,
    quantization: QuantizationDocument,
    mlx_format: bool,
}

/// Qwen3-VL text-encoder geometry consumed by the native language-model owner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QwenImage21TextEncoderConfig {
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: u64,
    pub mrope_section: [usize; 3],
    pub vocab_size: usize,
    pub vision_depth: usize,
    pub vision_hidden_size: usize,
    pub vision_intermediate_size: usize,
    pub vision_num_heads: usize,
    pub vision_num_position_embeddings: usize,
    pub vision_out_hidden_size: usize,
    pub vision_patch_size: u32,
    pub vision_spatial_merge_size: u32,
    pub vision_temporal_patch_size: u32,
    pub quantization_bits: u32,
    pub quantization_group_size: u32,
}

impl QwenImage21TextEncoderConfig {
    pub fn parse(json_bytes: &[u8]) -> Result<Self, QwenImage21ConfigError> {
        const DOCUMENT: &str = "text_encoder/config.json";
        let document: TextEncoderDocument = parse_document(json_bytes, DOCUMENT)?;
        let text = &document.text_config;
        let vision = &document.vision_config;
        let config = Self {
            image_token_id: document.image_token_id,
            video_token_id: document.video_token_id,
            vision_start_token_id: document.vision_start_token_id,
            vision_end_token_id: document.vision_end_token_id,
            head_dim: text.head_dim,
            hidden_size: text.hidden_size,
            intermediate_size: text.intermediate_size,
            num_attention_heads: text.num_attention_heads,
            num_hidden_layers: text.num_hidden_layers,
            num_key_value_heads: text.num_key_value_heads,
            rms_norm_eps: text.rms_norm_eps,
            rope_theta: text.rope_theta,
            mrope_section: text.rope_scaling.mrope_section,
            vocab_size: text.vocab_size,
            vision_depth: vision.depth,
            vision_hidden_size: vision.hidden_size,
            vision_intermediate_size: vision.intermediate_size,
            vision_num_heads: vision.num_heads,
            vision_num_position_embeddings: vision.num_position_embeddings,
            vision_out_hidden_size: vision.out_hidden_size,
            vision_patch_size: vision.patch_size,
            vision_spatial_merge_size: vision.spatial_merge_size,
            vision_temporal_patch_size: vision.temporal_patch_size,
            quantization_bits: document.quantization.bits,
            quantization_group_size: document.quantization.group_size,
        };
        require(
            document.architectures == ["Qwen3VLForConditionalGeneration"],
            DOCUMENT,
            "architectures",
        )?;
        require(document.model_type == "qwen3_vl", DOCUMENT, "model_type")?;
        require(document.dtype == "bfloat16", DOCUMENT, "dtype")?;
        require(
            document.tie_word_embeddings == false,
            DOCUMENT,
            "tie_word_embeddings",
        )?;
        require(
            document.transformers_version == "4.57.1",
            DOCUMENT,
            "transformers_version",
        )?;
        require(document.mlx_format, DOCUMENT, "mlx_format")?;
        require(
            text.attention_bias == false,
            DOCUMENT,
            "text_config.attention_bias",
        )?;
        require(
            text.attention_dropout == 0.0,
            DOCUMENT,
            "text_config.attention_dropout",
        )?;
        require(
            text.bos_token_id == 151_643,
            DOCUMENT,
            "text_config.bos_token_id",
        )?;
        require(
            text.eos_token_id == 151_645,
            DOCUMENT,
            "text_config.eos_token_id",
        )?;
        require(text.dtype == "bfloat16", DOCUMENT, "text_config.dtype")?;
        require(text.head_dim == 128, DOCUMENT, "text_config.head_dim")?;
        require(
            text.hidden_act == "silu",
            DOCUMENT,
            "text_config.hidden_act",
        )?;
        require(
            text.hidden_size == 4096,
            DOCUMENT,
            "text_config.hidden_size",
        )?;
        require(
            text.intermediate_size == 12288,
            DOCUMENT,
            "text_config.intermediate_size",
        )?;
        require(
            text.initializer_range == 0.02,
            DOCUMENT,
            "text_config.initializer_range",
        )?;
        require(
            text.max_position_embeddings == 262144,
            DOCUMENT,
            "text_config.max_position_embeddings",
        )?;
        require(
            text.model_type == "qwen3_vl_text",
            DOCUMENT,
            "text_config.model_type",
        )?;
        require(
            text.num_attention_heads == 32,
            DOCUMENT,
            "text_config.num_attention_heads",
        )?;
        require(
            text.num_hidden_layers == 36,
            DOCUMENT,
            "text_config.num_hidden_layers",
        )?;
        require(
            text.num_key_value_heads == 8,
            DOCUMENT,
            "text_config.num_key_value_heads",
        )?;
        require(
            text.rms_norm_eps == 1e-6,
            DOCUMENT,
            "text_config.rms_norm_eps",
        )?;
        require(
            text.rope_scaling.mrope_interleaved,
            DOCUMENT,
            "text_config.rope_scaling.mrope_interleaved",
        )?;
        require(
            text.rope_scaling.mrope_section == [24, 20, 20],
            DOCUMENT,
            "text_config.rope_scaling.mrope_section",
        )?;
        require(
            text.rope_scaling.rope_type == "default",
            DOCUMENT,
            "text_config.rope_scaling.rope_type",
        )?;
        require(
            text.rope_theta == 5_000_000,
            DOCUMENT,
            "text_config.rope_theta",
        )?;
        require(text.use_cache, DOCUMENT, "text_config.use_cache")?;
        require(
            text.vocab_size == 151_936,
            DOCUMENT,
            "text_config.vocab_size",
        )?;
        require(vision.depth == 27, DOCUMENT, "vision_config.depth")?;
        require(vision.dtype == "bfloat16", DOCUMENT, "vision_config.dtype")?;
        require(
            vision.hidden_act == "gelu_pytorch_tanh",
            DOCUMENT,
            "vision_config.hidden_act",
        )?;
        require(
            vision.hidden_size == 1152,
            DOCUMENT,
            "vision_config.hidden_size",
        )?;
        require(
            vision.in_channels == 3,
            DOCUMENT,
            "vision_config.in_channels",
        )?;
        require(
            vision.initializer_range == 0.02,
            DOCUMENT,
            "vision_config.initializer_range",
        )?;
        require(
            vision.intermediate_size == 4304,
            DOCUMENT,
            "vision_config.intermediate_size",
        )?;
        require(
            vision.model_type == "qwen3_vl",
            DOCUMENT,
            "vision_config.model_type",
        )?;
        require(vision.num_heads == 16, DOCUMENT, "vision_config.num_heads")?;
        require(
            vision.num_position_embeddings == 2304,
            DOCUMENT,
            "vision_config.num_position_embeddings",
        )?;
        require(
            vision.out_hidden_size == 4096,
            DOCUMENT,
            "vision_config.out_hidden_size",
        )?;
        require(
            vision.patch_size == 16,
            DOCUMENT,
            "vision_config.patch_size",
        )?;
        require(
            vision.spatial_merge_size == 2,
            DOCUMENT,
            "vision_config.spatial_merge_size",
        )?;
        require(
            vision.temporal_patch_size == 2,
            DOCUMENT,
            "vision_config.temporal_patch_size",
        )?;
        require(
            vision.deepstack_visual_indexes == [8, 16, 24],
            DOCUMENT,
            "vision_config.deepstack_visual_indexes",
        )?;
        require(
            document.image_token_id == 151_655,
            DOCUMENT,
            "image_token_id",
        )?;
        require(
            document.video_token_id == 151_656,
            DOCUMENT,
            "video_token_id",
        )?;
        require(
            document.vision_start_token_id == 151_652,
            DOCUMENT,
            "vision_start_token_id",
        )?;
        require(
            document.vision_end_token_id == 151_653,
            DOCUMENT,
            "vision_end_token_id",
        )?;
        reviewed_quantization(&document.quantization, DOCUMENT)?;
        Ok(config)
    }
}
