//! The forward Hadamard rotation as one Metal launch.
//!
//! The composed rotation is three launches per activation: the compiled
//! float32 cast with the sign flip, MLX's block Hadamard kernel, and the
//! cast back to the activation dtype, with two float32 intermediates round
//! tripping through memory between them. Decode on the folded pack issues
//! that rotation four times per decoder layer, so the composed route costs
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
//! capability probe and the fixture-pack unit test both check this equality
//! bit for bit, and the upstream parity journey is the end-to-end witness.

use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::{
    MlxArray, MlxCError, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput,
    MlxMetalKernelTemplateArgument,
};

/// MLX's largest butterfly radix and its device read width; the block must
/// be at least one radix wide so every pass uses the same radix MLX uses.
const MAX_RADIX: i32 = 16;
const READ_WIDTH: i32 = 4;
/// MLX splits transforms wider than this into two strided stages with a
/// different addition order, so blocks above it keep the composed route.
const MAX_SINGLE_STAGE_BLOCK: u32 = 8192;

const KERNEL_NAME: &str = "astronomical_prism_hadamard_signed_rotation";
const OPERATION: &str = "apply the fused signed Hadamard rotation";

/// True when the one-launch rotation reproduces MLX's single-stage block
/// transform for this block size: a power of two from one radix up to the
/// widest single-stage transform.
pub(crate) fn is_signed_rotation_eligible(block_size: u32) -> bool {
    block_size.is_power_of_two()
        && block_size >= MAX_RADIX as u32
        && block_size <= MAX_SINGLE_STAGE_BLOCK
}

/// Builds the rotation kernel for one block size. The normalization scale
/// is baked into the source as a float literal because Metal template
/// arguments carry no floats; the literal round-trips the exact float32 the
/// composed route hands MLX's kernel.
pub(crate) fn signed_rotation_kernel(block_size: u32) -> Result<MlxMetalKernel, MlxCError> {
    if !is_signed_rotation_eligible(block_size) {
        return Err(MlxCError {
            operation: OPERATION,
            description: format!(
                "block size {block_size} is outside the single-stage power-of-two range the fused rotation reproduces"
            ),
        });
    }
    let scale = 1.0 / (block_size as f32).sqrt();
    MlxMetalKernel::new_with_header(
        KERNEL_NAME,
        &["activations", "signs"],
        &["rotated"],
        SIGNED_ROTATION_KERNEL_HEADER,
        &signed_rotation_kernel_source(scale),
    )
}

/// Rotates every row of `activations` (last axis `width`, a multiple of the
/// block size) with the `width`-long float32 `signs`, returning the rotated
/// activations in the input dtype.
pub(crate) fn apply_signed_rotation(
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

/// The thread-local radix butterfly of MLX's `hadamard.h` (`radix_func`).
/// The butterfly index names (`h` for the half-span, `j` and `k` for the
/// butterfly pair positions) are kept as MLX writes them so the two sources
/// can be audited side by side; the arithmetic order is what parity relies on.
const SIGNED_ROTATION_KERNEL_HEADER: &str = r#"
template <short R>
METAL_FUNC void astronomical_hadamard_radix(thread float* butterfly_values) {
  constexpr short logR = __builtin_ctz(R);
  short h = 1;
  // Full unrolling (as MLX's STEEL_PRAGMA_UNROLL does) keeps every index
  // constant so the butterfly array stays in registers; a rolled loop
  // indexes thread memory dynamically and runs several times slower.
#pragma clang loop unroll(full)
  for (short stage = 0; stage < logR; stage++) {
#pragma clang loop unroll(full)
    for (short pair_index = 0; pair_index < R / 2; pair_index++) {
      short k = pair_index & (h - 1);
      short j = ((pair_index - k) << 1) + k;
      float upper = butterfly_values[j];
      float lower = butterfly_values[j + h];
      butterfly_values[j] = upper + lower;
      butterfly_values[j + h] = upper - lower;
    }
    h <<= 1;
  }
}
"#;

/// The kernel body: MLX's `hadamard_n<T, N, 16, 4>` schedule with the signed
/// float32 load in place of its plain read and the dtype-rounding store in
/// place of its float32 write. One threadgroup of `N / 16` threads owns one
/// Hadamard block of one row; `threadgroup_position_in_grid.y` is that block
/// row, and the block's position within its activation row selects the slice
/// of the sign vector it multiplies by.
fn signed_rotation_kernel_source(scale: f32) -> String {
    let scale_literal = format!("{scale:.9e}");
    format!(
        r#"
    constexpr int N = 1 << LOG_BLOCK;
    constexpr short max_radix = {MAX_RADIX};
    constexpr short read_width = {READ_WIDTH};
    constexpr short num_threads = N / max_radix;
    constexpr short logN = LOG_BLOCK;
    constexpr short logR = 4;
    constexpr short num_steps = logN / logR;
    constexpr short logFinal = logN % logR;
    constexpr short final_radix = 1 << logFinal;

    const uint block_row = threadgroup_position_in_grid.y;
    const int block_offset = int(block_row) * N;
    const int sign_offset = int(block_row % uint(BLOCKS_PER_ROW)) * N;
    const short thread_index = short(thread_position_in_threadgroup.x);

    threadgroup float block_values[N];

#pragma clang loop unroll(full)
    for (short read_step = 0; read_step < max_radix / read_width; read_step++) {{
        short element_index = read_step * read_width * num_threads + thread_index * read_width;
#pragma clang loop unroll(full)
        for (short lane = 0; lane < read_width; lane++) {{
            block_values[element_index + lane] =
                float(activations[block_offset + element_index + lane]) * signs[sign_offset + element_index + lane];
        }}
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float butterfly_values[max_radix];
    short h = 1;

#pragma clang loop unroll(full)
    for (short pass = 0; pass < num_steps; pass++) {{
        short k = thread_index & (h - 1);
        short j = ((thread_index - k) << logR) + k;
#pragma clang loop unroll(full)
        for (short radix_index = 0; radix_index < max_radix; radix_index++) {{
            butterfly_values[radix_index] = block_values[j + h * radix_index];
        }}
        astronomical_hadamard_radix<max_radix>(butterfly_values);
#pragma clang loop unroll(full)
        for (short radix_index = 0; radix_index < max_radix; radix_index++) {{
            block_values[j + h * radix_index] = butterfly_values[radix_index];
        }}
        h <<= logR;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (final_radix > 1) {{
#pragma clang loop unroll(full)
        for (int butterfly = 0; butterfly < max_radix / final_radix; butterfly++) {{
            short butterfly_index = thread_index + butterfly * num_threads;
            short k = butterfly_index & (h - 1);
            short j = ((butterfly_index - k) << logFinal) + k;
#pragma clang loop unroll(full)
            for (short radix_index = 0; radix_index < final_radix; radix_index++) {{
                butterfly_values[radix_index] = block_values[j + h * radix_index];
            }}
            astronomical_hadamard_radix<final_radix>(butterfly_values);
#pragma clang loop unroll(full)
            for (short radix_index = 0; radix_index < final_radix; radix_index++) {{
                block_values[j + h * radix_index] = butterfly_values[radix_index];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

#pragma clang loop unroll(full)
    for (short read_step = 0; read_step < max_radix / read_width; read_step++) {{
        short element_index = read_step * read_width * num_threads + thread_index * read_width;
#pragma clang loop unroll(full)
        for (short lane = 0; lane < read_width; lane++) {{
            rotated[block_offset + element_index + lane] = T(block_values[element_index + lane] * {scale_literal}f);
        }}
    }}
"#
    )
}
