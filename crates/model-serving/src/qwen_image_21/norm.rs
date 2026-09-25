//! Zero-centered RMSNorm for Qwen-Image-2.1: the pure CPU reference the hermetic oracle compares
//! against.
//!
//! Mirrors diffusers [`QwenImage21ZeroCenterRMSNorm`]: the learnable weight is stored zero-centered
//! (checkpoints hold `scale - 1`), so the effective scale is `weight + 1`, and the normalization is
//! `out = x * rsqrt(mean(x^2) + eps) * (weight + 1)`. Computed in f64 and cast to f32 — RMSNorm is
//! well-conditioned, so this reproduces the reference formula exactly without float32 reduction-order
//! noise, and the ~1e-6 deviation from the reference float32 path is immaterial downstream.
//!
//! The runtime path uses `mlx_math::fp32_zero_center_rms_norm` instead; this one exists so the
//! formula stays checkable without a GPU. Change both together.

/// Apply the zero-centered RMSNorm to a single row of `dim` channels.
///
/// `input` and `weight` must have the same non-zero width; `eps` is the variance floor (the
/// reference uses `1e-5` for the standalone norm and `1e-6` inside the text projection).
///
/// # Panics
///
/// Panics when `input` is empty or `weight` has a different width. The widths come from the
/// validated artifact, so a mismatch means the test or the binding is wrong rather than a runtime
/// condition this oracle should recover from.
#[must_use]
pub fn zero_center_rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let dim = input.len();
    assert!(dim > 0, "input must be non-empty");
    assert_eq!(weight.len(), dim, "weight width must match input width");

    let mean_square = input
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        / dim as f64;
    let reciprocal_rms = 1.0 / (mean_square + f64::from(eps)).sqrt();

    input
        .iter()
        .zip(weight.iter())
        .map(|(&value, &weight)| {
            (f64::from(value) * reciprocal_rms * (f64::from(weight) + 1.0)) as f32
        })
        .collect()
}
