//! Fused gathered projection for jangtq2 codebook experts: decode and
//! multiply in one launch, with no bfloat16 weight materialization.
//!
//! The materializing path (`mlx_decode::decode_expert_weight_rows`) writes
//! every routed expert's decoded bfloat16 weights to memory before
//! `gather_mm` reads them back exactly once for a single decode token. On the
//! 122B jangtq2 bundle that round trip is the dominant decode-token cost: a
//! top-8 projection decodes to about 50 MB and takes about 4.3 ms, three
//! projections per layer across 48 layers — roughly 620 ms of each token
//! before any storage read. This kernel instead reads the packed 2/3/4-bit
//! codes once (about 6 MB per top-8 projection), evaluates each codebook
//! level in registers, and accumulates the dot product against the
//! activation row directly, so the decoded weight never exists in memory.
//!
//! The kernel is a gathered matrix-vector product by construction (one
//! thread per assignment and output row), so it is decode-only by contract:
//! chunked prefill keeps the materializing decoder plus `gather_mm`, whose
//! tensor-core matrix multiply amortizes the decode over many tokens.
//!
//! Measured at tier 0 (`cargo test-jangtq2-decode-throughput`, optimized
//! build, real top-8 page shapes): gate/up 2-bit 5.59 ms → 0.165 ms (34×),
//! down 2-bit 4.73 ms → 0.168 ms (28×), down 3-bit 4.71 ms → 0.204 ms (23×),
//! each timed to an evaluated output against the materializing decode plus
//! `gather_mm`. One thread per output row was chosen over a SIMD-split dot
//! product because the whole grid (8,192 to 24,576 threads) already fills the
//! graphics processor at this width; splitting rows across lanes would add a
//! reduction without more memory parallelism to recover.
//!
//! Activations arrive already rotated (`mlx_decode::rotate_activations_hadamard32`)
//! exactly as they do for the materializing path; this kernel never applies
//! the rotation itself so both paths consume identical inputs.

use std::cell::RefCell;
use std::collections::HashMap;

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};
use mlx_c_rust::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
};

use super::codebook;
use super::hadamard32;

mod kernel_sources;

const FUSED_GATHERED_PROJECTION_OPERATION: &str = "fused jangtq2 gathered projection";
/// Threads per threadgroup for the one-dimensional launch. Every projection of
/// the family has an output width that is a multiple of 256 (1024 and 3072),
/// so the grid divides evenly and no threadgroup straddles two assignments.
const THREADGROUP_SIZE: i32 = 256;

/// Owns the compiled fused projection kernel plus the per-width codebook
/// level tables it reads.
///
/// The level table is a pure function of the bit width (at most sixteen
/// float32 values), uploaded once per width and reused by every launch so a
/// decode token's 144 projections never repeat the host-to-device copy.
#[derive(Debug)]
pub struct FusedJangtq2GatheredProjectionKernel {
    kernel: MlxMetalKernel,
    codebook_level_tables_by_bit_width: RefCell<HashMap<u32, MlxArray>>,
}

/// Validated launch geometry for one fused projection.
struct FusedProjectionGeometry {
    assignment_count: i32,
    output_rows: i32,
    input_width: i32,
    words_per_row: i32,
    shared_hidden_row: bool,
}

impl FusedJangtq2GatheredProjectionKernel {
    /// Compiles the fused projection kernel once per owner.
    ///
    /// # Errors
    /// Returns an error when the Metal source fails to compile.
    pub fn new() -> Result<Self, MlxRuntimeError> {
        let kernel = MlxMetalKernel::new(
            "fused_jangtq2_gathered_projection",
            &[
                "hidden",
                "expert_indices",
                "packed_weights",
                "row_scales",
                "codebook_levels",
            ],
            &["projected"],
            kernel_sources::FUSED_GATHERED_PROJECTION_KERNEL_SOURCE,
        )
        .map_err(MlxRuntimeError::from)?;
        Ok(Self {
            kernel,
            codebook_level_tables_by_bit_width: RefCell::new(HashMap::new()),
        })
    }

    /// Projects every assignment's activation row through its routed expert.
    ///
    /// * `hidden` — `[1, input_width]` (one decode token shared by every
    ///   assignment) or `[assignment_count, input_width]` (one row per
    ///   assignment), in float16, bfloat16, or float32.
    /// * `expert_indices` — a non-empty integer array; its element count is
    ///   the assignment count and its shape is otherwise free.
    /// * `packed_code_words` — `[experts, output_rows, words_per_row]` uint32.
    /// * `row_scales` — `[experts, output_rows]` float16.
    ///
    /// Returns `[assignment_count, output_rows]` in the activation dtype, left
    /// lazy so the caller's chain (SwiGLU, down projection, weighted
    /// reduction) evaluates it together with the rest of the layer.
    ///
    /// # Errors
    /// Returns an error when any input disagrees with the format contract or
    /// the launch fails.
    pub fn apply(
        &self,
        runtime: &MlxRuntime,
        hidden: &MlxArray,
        expert_indices: &MlxArray,
        packed_code_words: &MlxArray,
        row_scales: &MlxArray,
        bit_width: u32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let geometry = validate_fused_projection_inputs(
            hidden,
            expert_indices,
            packed_code_words,
            row_scales,
            bit_width,
        )?;
        let codebook_level_table = self.codebook_level_table(bit_width)?;
        let bits_i32 = i32::try_from(bit_width)
            .map_err(|_| projection_error("bit width exceeds the template range".to_owned()))?;
        let template_arguments = [
            MlxMetalKernelTemplateArgument::Dtype {
                name: "OutputT",
                dtype: hidden.dtype(),
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "bits",
                integer_template_argument: bits_i32,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "input_width",
                integer_template_argument: geometry.input_width,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "words_per_row",
                integer_template_argument: geometry.words_per_row,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "output_rows",
                integer_template_argument: geometry.output_rows,
            },
            MlxMetalKernelTemplateArgument::Boolean {
                name: "shared_hidden_row",
                boolean_template_argument: geometry.shared_hidden_row,
            },
        ];
        let thread_count = geometry
            .assignment_count
            .checked_mul(geometry.output_rows)
            .ok_or_else(|| projection_error("launch grid exceeds the MLX grid range".to_owned()))?;
        let mut outputs = runtime
            .apply_metal_kernel(
                &self.kernel,
                &[
                    hidden,
                    expert_indices,
                    packed_code_words,
                    row_scales,
                    &codebook_level_table,
                ],
                &[MlxMetalKernelOutput::new(
                    vec![geometry.assignment_count, geometry.output_rows],
                    hidden.dtype(),
                )],
                [thread_count, 1, 1],
                [THREADGROUP_SIZE, 1, 1],
                &template_arguments,
            )
            .map_err(MlxRuntimeError::from)?;
        outputs.pop().ok_or_else(|| {
            projection_error("the fused projection kernel returned no output".to_owned())
        })
    }

    fn codebook_level_table(&self, bit_width: u32) -> Result<MlxArray, MlxRuntimeError> {
        if let Some(level_table) = self
            .codebook_level_tables_by_bit_width
            .borrow()
            .get(&bit_width)
        {
            return level_table.retain().map_err(MlxRuntimeError::from);
        }
        let levels = codebook::codebook_levels(bit_width).ok_or_else(|| {
            projection_error(format!(
                "unsupported jangtq2 codebook bit width {bit_width}"
            ))
        })?;
        let level_count = i32::try_from(levels.len())
            .map_err(|_| projection_error("level table exceeds the MLX shape range".to_owned()))?;
        let level_table = MlxArray::from_f32(&levels, &[level_count])?;
        // Materialize the table eagerly: it is an input of every later launch
        // and must not drag a host-upload graph node into each projection.
        level_table.evaluate()?;
        let retained = level_table.retain()?;
        self.codebook_level_tables_by_bit_width
            .borrow_mut()
            .insert(bit_width, level_table);
        Ok(retained)
    }
}

/// Checks everything the kernel assumes about its inputs before launch.
///
/// Expert index *values* are not checked against the expert count: they come
/// from the router's top-k, which is in range by construction, and reading
/// them back to the host would stall decode on a GPU round trip per token.
fn validate_fused_projection_inputs(
    hidden: &MlxArray,
    expert_indices: &MlxArray,
    packed_code_words: &MlxArray,
    row_scales: &MlxArray,
    bit_width: u32,
) -> Result<FusedProjectionGeometry, MlxRuntimeError> {
    validate_fused_projection_dtypes(&FusedProjectionDtypes {
        bit_width,
        packed_code_word_dtype: packed_code_words.dtype(),
        row_scale_dtype: row_scales.dtype(),
        expert_index_dtype: expert_indices.dtype(),
        activation_dtype: hidden.dtype(),
    })?;
    let packed_rows =
        packed_row_geometry(&packed_code_words.shape(), &row_scales.shape(), bit_width)?;
    let assignment_count = i32::try_from(expert_indices.element_count())
        .map_err(|_| projection_error("assignment count exceeds the MLX shape range".to_owned()))?;
    let shared_hidden_row =
        is_shared_activation_row(&hidden.shape(), packed_rows.input_width, assignment_count)?;
    Ok(FusedProjectionGeometry {
        assignment_count,
        output_rows: packed_rows.output_rows,
        input_width: packed_rows.input_width,
        words_per_row: packed_rows.words_per_row,
        shared_hidden_row,
    })
}

/// The dtypes of the kernel's inputs and the codebook width they decode at.
struct FusedProjectionDtypes {
    bit_width: u32,
    packed_code_word_dtype: MlxDtype,
    row_scale_dtype: MlxDtype,
    expert_index_dtype: MlxDtype,
    activation_dtype: MlxDtype,
}

/// The Metal source reads codes as `uint32` words and scales as `half`, so
/// any other dtype would be reinterpreted rather than converted.
fn validate_fused_projection_dtypes(
    projection_dtypes: &FusedProjectionDtypes,
) -> Result<(), MlxRuntimeError> {
    let bit_width = projection_dtypes.bit_width;
    if !codebook::is_supported_bit_width(bit_width) {
        return Err(projection_error(format!(
            "unsupported jangtq2 codebook bit width {bit_width}"
        )));
    }
    if projection_dtypes.packed_code_word_dtype != MlxDtype::UInt32 {
        return Err(projection_error(format!(
            "packed code words must use uint32, got {:?}",
            projection_dtypes.packed_code_word_dtype
        )));
    }
    if projection_dtypes.row_scale_dtype != MlxDtype::Float16 {
        return Err(projection_error(format!(
            "row scales must use float16, got {:?}",
            projection_dtypes.row_scale_dtype
        )));
    }
    if !matches!(
        projection_dtypes.expert_index_dtype,
        MlxDtype::Int32 | MlxDtype::UInt32
    ) {
        return Err(projection_error(format!(
            "expert indices must use int32 or uint32, got {:?}",
            projection_dtypes.expert_index_dtype
        )));
    }
    if !matches!(
        projection_dtypes.activation_dtype,
        MlxDtype::Float16 | MlxDtype::BFloat16 | MlxDtype::Float32
    ) {
        return Err(projection_error(format!(
            "activations must use float16, bfloat16, or float32, got {:?}",
            projection_dtypes.activation_dtype
        )));
    }
    Ok(())
}

/// The per-expert row layout implied by the packed codes and their scales.
#[derive(Debug, PartialEq, Eq)]
struct PackedRowGeometry {
    output_rows: i32,
    words_per_row: i32,
    input_width: i32,
}

fn packed_row_geometry(
    packed_shape: &[i32],
    scale_shape: &[i32],
    bit_width: u32,
) -> Result<PackedRowGeometry, MlxRuntimeError> {
    let ([expert_count, output_rows, words_per_row], [scale_expert_count, scale_output_rows]) =
        (packed_shape, scale_shape)
    else {
        return Err(projection_error(format!(
            "packed tensors must have rank three and row scales rank two, got ranks {} and {}",
            packed_shape.len(),
            scale_shape.len()
        )));
    };
    if [*expert_count, *output_rows, *words_per_row]
        .into_iter()
        .min()
        <= Some(0)
        || expert_count != scale_expert_count
        || output_rows != scale_output_rows
    {
        return Err(projection_error(format!(
            "packed code shape {packed_shape:?} and row scale shape {scale_shape:?} disagree"
        )));
    }
    let input_width = decoded_input_width(*words_per_row, bit_width)?;
    Ok(PackedRowGeometry {
        output_rows: *output_rows,
        words_per_row: *words_per_row,
        input_width,
    })
}

/// The activation width one packed row decodes to: `words_per_row` 32-bit
/// words of `bit_width`-bit codes, which must be whole 32-code groups and
/// align to the Hadamard rotation block.
fn decoded_input_width(words_per_row: i32, bit_width: u32) -> Result<i32, MlxRuntimeError> {
    let bits_i32 = i32::try_from(bit_width)
        .map_err(|_| projection_error("bit width exceeds the template range".to_owned()))?;
    if words_per_row % bits_i32 != 0 {
        return Err(projection_error(format!(
            "packed row of {words_per_row} words is not a whole number of 32-code groups at {bit_width} bits"
        )));
    }
    // Checked: a malformed packed shape must be refused, not wrapped into a
    // small plausible width in an optimized build.
    let Some(packed_bit_count) = words_per_row.checked_mul(32) else {
        return Err(projection_error(format!(
            "packed row of {words_per_row} words exceeds the MLX shape range"
        )));
    };
    let input_width = packed_bit_count / bits_i32;
    // Whole 32-code groups always decode to a multiple of 32, so this holds
    // while the Hadamard block is 32 wide; it guards a change to that width.
    if !hadamard32::is_applicable_to_input_width(input_width as usize) {
        return Err(projection_error(format!(
            "decoded input width {input_width} is not aligned to the jangtq2 packing width"
        )));
    }
    Ok(input_width)
}

/// Whether one activation row serves every assignment (decode broadcasts
/// the token's hidden state) or each assignment brings its own row.
fn is_shared_activation_row(
    hidden_shape: &[i32],
    input_width: i32,
    assignment_count: i32,
) -> Result<bool, MlxRuntimeError> {
    let [activation_row_count, activation_width] = hidden_shape else {
        return Err(projection_error(format!(
            "activations must be [1 | assignments, {input_width}], got {hidden_shape:?}"
        )));
    };
    if *activation_width != input_width {
        return Err(projection_error(format!(
            "activations must be [1 | assignments, {input_width}], got {hidden_shape:?}"
        )));
    }
    if assignment_count == 0 {
        return Err(projection_error(
            "expert indices must name at least one assignment".to_owned(),
        ));
    }
    match *activation_row_count {
        1 => Ok(true),
        activation_rows if activation_rows == assignment_count => Ok(false),
        activation_rows => Err(projection_error(format!(
            "activations carry {activation_rows} rows but the route has {assignment_count} assignments"
        ))),
    }
}

fn projection_error(description: String) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: FUSED_GATHERED_PROJECTION_OPERATION,
        description,
    }
}

#[cfg(test)]
#[path = "fused_gathered_projection_tests.rs"]
mod fused_gathered_projection_tests;
