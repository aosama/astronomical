//! Exact expected tensor profiles (name, dtype, shape) for the reviewed Qwen-Image-2.1 MLX
//! weight components.
//!
//! These profiles are the single source of truth shared by the artifact inventory validator and
//! the hermetic test fixtures: a package validates only if its safetensors header contains
//! exactly this set, and the synthetic fixtures are generated from the same generator, so
//! neither side is coupled to hand-transcribed golden constants.
//!
//! MLX dual-affine 4-bit convention: a quantized linear with `in` input features and `out`
//! output features stores `weight` as `U32[out, in / 8]` (8 nibbles per word), `scales` as
//! `BF16[out, in / group]`, and `biases` as `BF16[out, in / group]`. The reference linears are
//! bias-free; the `biases` tensors are the per-group dequantization biases, not linear biases.
//!
//! VAE convolutions are stored MLX-style as 4D `[out, kh, kw, in]` kernels — the temporal axis
//! of the causal 3D VAE is handled by the separate `time_conv` 1x1 convolutions, not by a fifth
//! kernel dimension.

use super::configuration::{quantized_group_count, quantized_row_count};

/// One expected physical tensor of a weight component.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenImage21TensorProfile {
    pub tensor_name: String,
    /// Safetensors dtype name (`"U32"`, `"BF16"`, `"F32"`).
    pub dtype: &'static str,
    pub shape: Vec<usize>,
}

impl QwenImage21TensorProfile {
    pub(super) fn bf16(tensor_name: &str, shape: Vec<usize>) -> Self {
        Self {
            tensor_name: tensor_name.to_owned(),
            dtype: "BF16",
            shape,
        }
    }

    pub(super) fn f32(tensor_name: &str, shape: Vec<usize>) -> Self {
        Self {
            tensor_name: tensor_name.to_owned(),
            dtype: "F32",
            shape,
        }
    }
}

pub(super) fn quantized_linear(
    profiles: &mut Vec<QwenImage21TensorProfile>,
    prefix: &str,
    out_features: usize,
    in_features: usize,
    bits: u32,
    group_size: u32,
) {
    let rows = quantized_row_count(in_features, bits);
    let groups = quantized_group_count(in_features, group_size);
    profiles.push(QwenImage21TensorProfile {
        tensor_name: format!("{prefix}.weight"),
        dtype: "U32",
        shape: vec![out_features, rows],
    });
    profiles.push(QwenImage21TensorProfile::bf16(
        &format!("{prefix}.scales"),
        vec![out_features, groups],
    ));
    profiles.push(QwenImage21TensorProfile::bf16(
        &format!("{prefix}.biases"),
        vec![out_features, groups],
    ));
}

mod text_encoder;
mod transformer;
mod vae;

pub use text_encoder::text_encoder_tensor_profiles;
pub use transformer::transformer_tensor_profiles;
pub use vae::vae_tensor_profiles;
