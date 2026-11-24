//! `transformer/config.json`: the denoising transformer's reviewed geometry.

use serde::Deserialize;

use super::{
    QuantizationDocument, QwenImage21ConfigError, parse_document, require, reviewed_quantization,
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransformerDocument {
    #[serde(rename = "_class_name")]
    class_name: String,
    #[serde(rename = "_diffusers_version")]
    diffusers_version: String,
    attention_head_dim: usize,
    axes_dims_rope: [usize; 3],
    context_in_dim: usize,
    eps: f64,
    in_channels: usize,
    mlp_ratio: u32,
    num_attention_heads: usize,
    num_layers: usize,
    out_channels: usize,
    patch_size: usize,
    causal_condition: bool,
    quantization: QuantizationDocument,
    mlx_format: bool,
}

/// Geometry consumed directly by the native transformer owner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QwenImage21TransformerConfig {
    pub attention_head_dim: usize,
    pub axes_dims_rope: [usize; 3],
    pub context_in_dim: usize,
    pub eps: f64,
    pub in_channels: usize,
    pub mlp_ratio: u32,
    pub num_attention_heads: usize,
    pub num_layers: usize,
    pub out_channels: usize,
    pub patch_size: usize,
    pub causal_condition: bool,
    pub quantization_bits: u32,
    pub quantization_group_size: u32,
}

impl QwenImage21TransformerConfig {
    pub fn parse(json_bytes: &[u8]) -> Result<Self, QwenImage21ConfigError> {
        const DOCUMENT: &str = "transformer/config.json";
        let document: TransformerDocument = parse_document(json_bytes, DOCUMENT)?;
        require(
            document.class_name == "QwenImage21Transformer2DModel",
            DOCUMENT,
            "_class_name",
        )?;
        require(
            document.diffusers_version == "0.37.0.dev0",
            DOCUMENT,
            "_diffusers_version",
        )?;
        let config = Self {
            attention_head_dim: document.attention_head_dim,
            axes_dims_rope: document.axes_dims_rope,
            context_in_dim: document.context_in_dim,
            eps: document.eps,
            in_channels: document.in_channels,
            mlp_ratio: document.mlp_ratio,
            num_attention_heads: document.num_attention_heads,
            num_layers: document.num_layers,
            out_channels: document.out_channels,
            patch_size: document.patch_size,
            causal_condition: document.causal_condition,
            quantization_bits: document.quantization.bits,
            quantization_group_size: document.quantization.group_size,
        };
        require(
            config.attention_head_dim == 128,
            DOCUMENT,
            "attention_head_dim",
        )?;
        require(
            config.axes_dims_rope == [16, 56, 56],
            DOCUMENT,
            "axes_dims_rope",
        )?;
        require(config.context_in_dim == 4096, DOCUMENT, "context_in_dim")?;
        require(config.eps == 1e-6, DOCUMENT, "eps")?;
        require(config.in_channels == 64, DOCUMENT, "in_channels")?;
        require(config.mlp_ratio == 3, DOCUMENT, "mlp_ratio")?;
        require(
            config.num_attention_heads == 32,
            DOCUMENT,
            "num_attention_heads",
        )?;
        require(config.num_layers == 32, DOCUMENT, "num_layers")?;
        require(config.out_channels == 64, DOCUMENT, "out_channels")?;
        require(config.patch_size == 1, DOCUMENT, "patch_size")?;
        require(config.causal_condition, DOCUMENT, "causal_condition")?;
        require(document.mlx_format, DOCUMENT, "mlx_format")?;
        reviewed_quantization(&document.quantization, DOCUMENT)?;
        Ok(config)
    }

    /// Inner hidden dimension (`num_attention_heads * attention_head_dim`).
    #[must_use]
    pub const fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// SwiGLU intermediate width (`inner_dim * mlp_ratio`).
    #[must_use]
    pub const fn mlp_hidden_dim(&self) -> usize {
        self.inner_dim() * (self.mlp_ratio as usize)
    }
}
