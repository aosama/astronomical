//! The Hadamard activation transform for folded ternary projections.
//!
//! Folded packs store projection weights in a rotated basis; every activation
//! passing through a rotated projection must carry the matching transform.
//! The arithmetic mirrors the upstream reference exactly: cast to float32,
//! apply the explicit ±1 signs (before the transform on the forward side,
//! after it on the inverse side), apply the normalized block Hadamard, and
//! restore the activation dtype.

use astronomical_mlx_c_rust::{MlxCError, MlxDtype};
use astronomical_runtime_integration::MlxRuntime;

use super::super::configuration::FOLDED_MODULE_DTYPE;

/// The float32 working domain of the transform, matching the upstream
/// reference's cast discipline.
const TRANSFORM_DTYPE: MlxDtype = MlxDtype::Float32;

/// The activation dtype every packed module declares. The wire schema pins
/// the folded modules to float16; this keeps the pin in one place.
pub(crate) fn activation_dtype() -> MlxDtype {
    match FOLDED_MODULE_DTYPE {
        "float16" => MlxDtype::Float16,
        _ => MlxDtype::Float16,
    }
}

/// Applies the forward Hadamard transform: signs, then the normalized block
/// transform. Used on the activation-input side of rotated projections.
pub(crate) fn apply_forward_transform(
    runtime: &MlxRuntime,
    activations: &astronomical_mlx_c_rust::MlxArray,
    signs: &astronomical_mlx_c_rust::MlxArray,
    block_size: u32,
) -> Result<astronomical_mlx_c_rust::MlxArray, MlxCError> {
    apply_transform(runtime, activations, signs, block_size, false)
}

/// Applies the inverse Hadamard transform: the normalized block transform,
/// then the signs. Used on the embedding-output side of the inverse module.
pub(crate) fn apply_inverse_transform(
    runtime: &MlxRuntime,
    activations: &astronomical_mlx_c_rust::MlxArray,
    signs: &astronomical_mlx_c_rust::MlxArray,
    block_size: u32,
) -> Result<astronomical_mlx_c_rust::MlxArray, MlxCError> {
    apply_transform(runtime, activations, signs, block_size, true)
}

fn apply_transform(
    runtime: &MlxRuntime,
    activations: &astronomical_mlx_c_rust::MlxArray,
    signs: &astronomical_mlx_c_rust::MlxArray,
    block_size: u32,
    inverse: bool,
) -> Result<astronomical_mlx_c_rust::MlxArray, MlxCError> {
    let activation_shape = activations.shape();
    let width = *activation_shape
        .last()
        .expect("activations always have at least one axis");
    debug_assert_eq!(
        signs.shape(),
        vec![width],
        "the sign vector must match the activation width"
    );
    debug_assert_eq!(
        width % block_size as i32,
        0,
        "the hadamard block must divide the activation width"
    );
    let activation_dtype = activations.dtype();

    let float32_activations = runtime.astype(activations, TRANSFORM_DTYPE)?;
    let signed = if inverse {
        float32_activations
    } else {
        runtime.multiply(&float32_activations, signs)?
    };
    // The transform applies per block along the last axis: flatten every
    // leading axis with the width-to-block segmentation into rows.
    let leading_element_count: i32 = activation_shape[..activation_shape.len() - 1]
        .iter()
        .product();
    let block_rows = leading_element_count * (width / block_size as i32);
    let blocked = runtime.reshape(&signed, &[block_rows, block_size as i32])?;
    let transformed =
        runtime.hadamard_transform(&blocked, Some(1.0 / (block_size as f32).sqrt()))?;
    let restored_shape: Vec<i32> = activation_shape.iter().map(|axis| *axis as i32).collect();
    let restored = runtime.reshape(&transformed, &restored_shape)?;
    let signed_result = if inverse {
        runtime.multiply(&restored, signs)?
    } else {
        restored
    };
    runtime.astype(&signed_result, activation_dtype)
}
