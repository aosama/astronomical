//! `vae/config.json`: the 3D causal VAE's reviewed geometry and latent denormalization
//! constants.

use serde::Deserialize;

use super::{QwenImage21ConfigError, parse_document, require};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VaeDocument {
    #[serde(rename = "_class_name")]
    class_name: String,
    #[serde(rename = "_diffusers_version")]
    diffusers_version: String,
    attn_scales: Vec<f64>,
    base_dim: usize,
    decoder_base_dim: usize,
    dim_mult: [u32; 5],
    dropout: f64,
    in_channels: usize,
    is_residual: bool,
    latents_mean: Vec<f64>,
    latents_std: Vec<f64>,
    num_res_blocks: usize,
    out_channels: usize,
    patch_size: Option<usize>,
    scale_factor_spatial: usize,
    scale_factor_temporal: usize,
    #[serde(rename = "temperal_downsample")]
    temporal_downsample: [bool; 4],
    z_dim: usize,
    mlx_format: bool,
}

/// 3D causal VAE geometry plus the latent denormalization constants.
#[derive(Clone, Debug, PartialEq)]
pub struct QwenImage21VaeConfig {
    pub base_dim: usize,
    pub decoder_base_dim: usize,
    pub dim_mult: [u32; 5],
    pub in_channels: usize,
    pub num_res_blocks: usize,
    pub out_channels: usize,
    pub scale_factor_spatial: usize,
    pub scale_factor_temporal: usize,
    pub temporal_downsample: [bool; 4],
    pub z_dim: usize,
    latents_mean: Vec<f64>,
    latents_std: Vec<f64>,
}

impl QwenImage21VaeConfig {
    pub fn parse(json_bytes: &[u8]) -> Result<Self, QwenImage21ConfigError> {
        const DOCUMENT: &str = "vae/config.json";
        let document: VaeDocument = parse_document(json_bytes, DOCUMENT)?;
        require(
            document.class_name == "AutoencoderKLQwenImage21",
            DOCUMENT,
            "_class_name",
        )?;
        require(
            document.diffusers_version == "0.37.0.dev0",
            DOCUMENT,
            "_diffusers_version",
        )?;
        let z_dim = document.z_dim;
        require(z_dim == 64, DOCUMENT, "z_dim")?;
        require(document.attn_scales.is_empty(), DOCUMENT, "attn_scales")?;
        require(document.base_dim == 96, DOCUMENT, "base_dim")?;
        require(
            document.decoder_base_dim == 144,
            DOCUMENT,
            "decoder_base_dim",
        )?;
        require(document.dim_mult == [1, 2, 4, 8, 8], DOCUMENT, "dim_mult")?;
        require(document.dropout == 0.0, DOCUMENT, "dropout")?;
        require(document.in_channels == 4, DOCUMENT, "in_channels")?;
        require(document.is_residual, DOCUMENT, "is_residual")?;
        require(document.num_res_blocks == 2, DOCUMENT, "num_res_blocks")?;
        require(document.out_channels == 4, DOCUMENT, "out_channels")?;
        require(document.patch_size.is_none(), DOCUMENT, "patch_size")?;
        require(
            document.scale_factor_spatial == 16,
            DOCUMENT,
            "scale_factor_spatial",
        )?;
        require(
            document.scale_factor_temporal == 8,
            DOCUMENT,
            "scale_factor_temporal",
        )?;
        require(
            document.temporal_downsample == [false, true, true, true],
            DOCUMENT,
            "temperal_downsample",
        )?;
        require(document.mlx_format, DOCUMENT, "mlx_format")?;
        require(
            document.latents_mean.len() == z_dim,
            DOCUMENT,
            "latents_mean",
        )?;
        require(document.latents_std.len() == z_dim, DOCUMENT, "latents_std")?;
        require(
            document
                .latents_std
                .iter()
                .all(|std_value| *std_value > 0.0),
            DOCUMENT,
            "latents_std",
        )?;
        Ok(Self {
            base_dim: document.base_dim,
            decoder_base_dim: document.decoder_base_dim,
            dim_mult: document.dim_mult,
            in_channels: document.in_channels,
            num_res_blocks: document.num_res_blocks,
            out_channels: document.out_channels,
            scale_factor_spatial: document.scale_factor_spatial,
            scale_factor_temporal: document.scale_factor_temporal,
            temporal_downsample: document.temporal_downsample,
            z_dim,
            latents_mean: document.latents_mean,
            latents_std: document.latents_std,
        })
    }

    pub fn latents_mean(&self) -> &[f64] {
        &self.latents_mean
    }
    pub fn latents_std(&self) -> &[f64] {
        &self.latents_std
    }
}
