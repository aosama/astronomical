//! The Hadamard activation transform for folded ternary projections.
//!
//! Folded packs store projection weights in a rotated basis; every activation
//! passing through a rotated projection must carry the matching transform.
//! The arithmetic mirrors the upstream reference exactly: cast to float32,
//! apply the explicit ±1 signs (before the transform on the forward side,
//! after it on the inverse side), apply the normalized block Hadamard, and
//! restore the activation dtype. The float32 working domain is not optional:
//! the Metal Hadamard kernel rounds through its element type between radix
//! passes, so a float16 transform would diverge from the reference.
//!
//! The forward transform produces a [`RotatedActivations`] value, the only
//! input a packed projection accepts. Sibling projections that read the same
//! activation (the gated-delta query/key/value and gate projections, the
//! full-attention query/key/value projections, the MLP gate and up
//! projections) therefore share one transform instead of recomputing it.
//! When the model retains the one-launch rotation kernel
//! (`signed_rotation_kernel.rs`), the forward transform runs as that single
//! launch instead of the composed three; both routes produce the same bits.

use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::{MlxArray, MlxCError, MlxCompiledElementwiseGraphs, MlxDtype, MlxMetalKernel};

use super::super::configuration::FOLDED_MODULE_DTYPE;
use super::signed_rotation_kernel::apply_signed_rotation;

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

/// One validated sign vector and block size for a given activation width:
/// the complete activation-side contract a rotated projection of that input
/// width needs. Built once per width from the pack's Hadamard contract and
/// shared by every projection of that width, exactly as the upstream runtime
/// indexes its sign table by width.
#[derive(Debug)]
pub(crate) struct HadamardRotation {
    width: i32,
    block_size: u32,
    /// Float32 ±1 signs, the multiply operand of the transform.
    signs: MlxArray,
}

impl HadamardRotation {
    pub(crate) fn new(
        runtime: &MlxRuntime,
        width: u32,
        block_size: u32,
        sign_values: &[i8],
    ) -> Result<Self, MlxCError> {
        debug_assert_eq!(sign_values.len(), width as usize);
        debug_assert_eq!(width % block_size, 0);
        let float32_signs: Vec<f32> = sign_values.iter().map(|sign| f32::from(*sign)).collect();
        let signs = runtime.array_from_f32(&float32_signs, &[width as i32])?;
        Ok(Self {
            width: width as i32,
            block_size,
            signs,
        })
    }

    /// Applies the forward transform (signs, then the normalized block
    /// transform) to activations whose last axis is this rotation's width.
    ///
    /// With `fused_rotation_kernel` the whole transform is one launch; without
    /// it the composed route runs the compiled signed float32 cast, MLX's
    /// block Hadamard, and the cast back. Both produce the same bits.
    pub(crate) fn rotate(
        &self,
        runtime: &MlxRuntime,
        compiled_elementwise_graphs: &MlxCompiledElementwiseGraphs,
        fused_rotation_kernel: Option<&MlxMetalKernel>,
        activations: &MlxArray,
    ) -> Result<RotatedActivations, MlxCError> {
        let activation_shape = activations.shape();
        let width = *activation_shape
            .last()
            .expect("activations always have at least one axis");
        debug_assert_eq!(
            width, self.width,
            "the activation width must match the rotation"
        );
        if let Some(fused_rotation_kernel) = fused_rotation_kernel {
            let activations = apply_signed_rotation(
                runtime,
                fused_rotation_kernel,
                activations,
                &self.signs,
                self.block_size,
            )?;
            return Ok(RotatedActivations {
                activations,
                width: self.width,
            });
        }
        let activation_dtype = activations.dtype();
        // One fused launch for the float32 cast and the sign flip, replacing
        // a cast kernel plus a multiply kernel and their float32 round trip.
        let signed = runtime.apply_compiled_signed_float32_cast(
            compiled_elementwise_graphs,
            activations,
            &self.signs,
        )?;
        let transformed = block_hadamard(runtime, &signed, &activation_shape, self.block_size)?;
        let activations = runtime.astype(&transformed, activation_dtype)?;
        Ok(RotatedActivations {
            activations,
            width: self.width,
        })
    }

    /// Applies the inverse transform (the normalized block transform, then
    /// the signs) on the embedding-output side of the inverse module.
    pub(crate) fn rotate_inverse(
        &self,
        runtime: &MlxRuntime,
        activations: &MlxArray,
    ) -> Result<MlxArray, MlxCError> {
        let activation_shape = activations.shape();
        let activation_dtype = activations.dtype();
        let float32_activations = runtime.astype(activations, TRANSFORM_DTYPE)?;
        let transformed = block_hadamard(
            runtime,
            &float32_activations,
            &activation_shape,
            self.block_size,
        )?;
        let signed = runtime.multiply(&transformed, &self.signs)?;
        runtime.astype(&signed, activation_dtype)
    }
}

/// The normalized block Hadamard over the last axis: flatten every leading
/// axis with the width-to-block segmentation into rows, transform, restore.
fn block_hadamard(
    runtime: &MlxRuntime,
    float32_activations: &MlxArray,
    activation_shape: &[i32],
    block_size: u32,
) -> Result<MlxArray, MlxCError> {
    let width = activation_shape[activation_shape.len() - 1];
    debug_assert_eq!(
        width % block_size as i32,
        0,
        "the hadamard block must divide the activation width"
    );
    let leading_element_count: i32 = activation_shape[..activation_shape.len() - 1]
        .iter()
        .product();
    let block_rows = leading_element_count * (width / block_size as i32);
    let blocked = runtime.reshape(float32_activations, &[block_rows, block_size as i32])?;
    let transformed =
        runtime.hadamard_transform(&blocked, Some(1.0 / (block_size as f32).sqrt()))?;
    runtime.reshape(&transformed, activation_shape)
}

/// Activations already carried into the rotated basis, tagged with the
/// width they were rotated at. Only [`HadamardRotation::rotate`] constructs
/// one, so a packed projection cannot run on untransformed input, and a
/// projection whose input width differs from the tag refuses the value.
#[derive(Debug)]
pub(crate) struct RotatedActivations {
    activations: MlxArray,
    width: i32,
}

impl RotatedActivations {
    pub(crate) const fn activations(&self) -> &MlxArray {
        &self.activations
    }

    pub(crate) const fn width(&self) -> i32 {
        self.width
    }

    /// Releases the rotated array, for callers that measure the transform
    /// on its own rather than feeding a projection.
    pub(crate) fn into_activations(self) -> MlxArray {
        self.activations
    }
}
