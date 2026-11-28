//! The forward Hadamard rotation as one Metal launch.
//!
//! The composed rotation is three launches per activation: the compiled
//! float32 cast with the sign flip, MLX's block Hadamard kernel, and the
//! cast back to the activation dtype, with two float32 intermediates round
//! tripping through memory between them. Decode on a folded ternary pack
//! issues that rotation four times per decoder layer, so the composed route costs
//! roughly five hundred launches per generated token on a path whose
//! per-token time is already launch-bound. This kernel does the same work in
//! one launch: each threadgroup owns one Hadamard block of one row, loads
//! it as `float(activation) * sign`, runs the butterfly, and stores
//! `T(value * scale)`.
//!
//! The arithmetic is bit-identical to the composed route by construction.
//! The butterfly is MLX's own `hadamard_n` kernel (radix-16 passes followed
//! by the residual radix, working in float32 with the same thread-to-
//! butterfly assignment), reproduced here rather than paraphrased so every
//! float32 addition happens in the same order; see
//! third-party/THIRD_PARTY_NOTICES for the MLX attribution. The load matches
//! the compiled signed cast (`float(x) * sign` is exact for `sign` in ±1),
//! and the single store rounds `float * scale` to the activation dtype exactly
//! where the composed route's final cast rounds its float32 product. The
//! capability probe checks this equality bit for bit at one block, its tests
//! at every block a pack may declare, and the upstream parity
//! journey is the end-to-end witness. The Metal is in
//! `metal/astro_family_qwen35_signed_hadamard_rotation.metal`.

use mlx_bridge::{
    MlxArray, MlxCError, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput,
    MlxMetalKernelTemplateArgument,
};
use model_primitives::metal_kernel_declaration::{declare_metal_kernel, float_define};
use runtime_integration::MlxRuntime;

use crate::metal_kernels::ASTRO_FAMILY_QWEN35_SIGNED_HADAMARD_ROTATION;

/// MLX's largest butterfly radix, which the kernel's `max_radix` repeats; the
/// block must be at least one radix wide so every pass uses the same radix
/// MLX uses.
const MAX_RADIX: i32 = 16;
/// MLX splits transforms wider than this into two strided stages with a
/// different addition order, so blocks above it keep the composed route.
const MAX_SINGLE_STAGE_BLOCK: u32 = 8192;

const OPERATION: &str = "apply the fused signed Hadamard rotation";

/// True when the one-launch rotation reproduces MLX's single-stage block
/// transform for this block size: a power of two from one radix up to the
/// widest single-stage transform.
pub fn is_signed_rotation_eligible(block_size: u32) -> bool {
    block_size.is_power_of_two()
        && block_size >= MAX_RADIX as u32
        && block_size <= MAX_SINGLE_STAGE_BLOCK
}

/// Builds the rotation kernel for one block size. The normalization scale
/// is baked into the source as a float literal because Metal template
/// arguments carry no floats; the literal round-trips the exact float32 the
/// composed route hands MLX's kernel.
pub fn signed_rotation_kernel(block_size: u32) -> Result<MlxMetalKernel, MlxCError> {
    if !is_signed_rotation_eligible(block_size) {
        return Err(MlxCError {
            operation: OPERATION,
            description: format!(
                "block size {block_size} is outside the single-stage power-of-two range the fused rotation reproduces"
            ),
        });
    }
    let scale = 1.0 / (block_size as f32).sqrt();
    declare_metal_kernel(
        &ASTRO_FAMILY_QWEN35_SIGNED_HADAMARD_ROTATION,
        &[float_define("SIGNED_ROTATION_SCALE", scale)],
    )
    .map_err(|declaration_error| MlxCError {
        operation: OPERATION,
        description: declaration_error.to_string(),
    })
}

/// Rotates every row of `activations` (last axis `width`, a multiple of the
/// block size) with the `width`-long float32 `signs`, returning the rotated
/// activations in the input dtype.
pub fn apply_signed_rotation(
    runtime: &MlxRuntime,
    kernel: &MlxMetalKernel,
    activations: &MlxArray,
    signs: &MlxArray,
    block_size: u32,
) -> Result<MlxArray, MlxCError> {
    let activation_shape = activations.shape();
    let width = *activation_shape.last().ok_or_else(|| MlxCError {
        operation: OPERATION,
        description: "activations must have at least one axis".to_owned(),
    })?;
    let block = block_size as i32;
    if width <= 0 || width % block != 0 {
        return Err(MlxCError {
            operation: OPERATION,
            description: format!("the block size {block} must divide the activation width {width}"),
        });
    }
    if signs.shape() != [width] || signs.dtype() != MlxDtype::Float32 {
        return Err(MlxCError {
            operation: OPERATION,
            description: "signs must be a float32 vector of the activation width".to_owned(),
        });
    }
    let row_count: i32 = activation_shape[..activation_shape.len() - 1]
        .iter()
        .product();
    let block_row_count = row_count * (width / block);
    let threads_per_block = block / MAX_RADIX;
    let mut outputs = runtime.apply_metal_kernel(
        kernel,
        &[activations, signs],
        &[MlxMetalKernelOutput::new(
            activation_shape.clone(),
            activations.dtype(),
        )],
        [threads_per_block, block_row_count, 1],
        [threads_per_block, 1, 1],
        &[
            MlxMetalKernelTemplateArgument::Dtype {
                name: "T",
                dtype: activations.dtype(),
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "LOG_BLOCK",
                integer_template_argument: block.trailing_zeros() as i32,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "BLOCKS_PER_ROW",
                integer_template_argument: width / block,
            },
        ],
    )?;
    outputs.pop().ok_or_else(|| MlxCError {
        operation: OPERATION,
        description: "the fused rotation kernel returned no output".to_owned(),
    })
}
