//! Shape and range checks every VAE component runs before it builds graph work.
//!
//! The decoder is assembled from a weight manifest, not from a config schema, so a mis-named or
//! mis-shaped tensor would otherwise surface as a wrong-shaped output or an MLX error deep inside
//! a convolution. Each component validates what it loads and what it is handed, and reports the
//! artifact path, role, and both shapes so the failure names the tensor that is wrong.
//!
//! These helpers live here rather than beside one component because the whole decode path uses
//! them; keeping them in `convolution.rs` made a convolution module the owner of VAE-wide
//! validation.

use astronomical_runtime_integration::MlxArray;

use super::QwenImage21VaeError;

/// Converts a Rust dimension to the `i32` MLX shapes are expressed in.
pub(super) fn as_i32(value: usize, role: &str) -> Result<i32, QwenImage21VaeError> {
    i32::try_from(value).map_err(|_| {
        QwenImage21VaeError::invalid_geometry(format!("{role} exceeds the MLX integer range"))
    })
}

/// Fails unless `tensor` has exactly `expected_shape`.
///
/// `prefix` is the artifact path and `tensor_role` the sub-tensor name (`weight`, `bias`,
/// `gamma`, …), so the message reads as the manifest entry a maintainer has to look at.
pub(super) fn validate_shape(
    prefix: &str,
    tensor_role: &str,
    tensor: &MlxArray,
    expected_shape: &[usize],
) -> Result<(), QwenImage21VaeError> {
    let expected_i32 = expected_shape
        .iter()
        .map(|dimension| as_i32(*dimension, "expected tensor dimension"))
        .collect::<Result<Vec<_>, _>>()?;
    if tensor.shape() != expected_i32 {
        return Err(QwenImage21VaeError::invalid_geometry(format!(
            "{prefix}.{tensor_role} expected shape {expected_shape:?}, received {:?}",
            tensor.shape()
        )));
    }
    Ok(())
}
