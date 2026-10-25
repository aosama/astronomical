//! MLX operations for decoding one bounded jangtq2 expert page.
//!
//! The CPU format contract owns bit extraction and cubic codebook semantics;
//! this module mirrors those rules on the graphics processor so selected
//! expert pages can be decoded without a full weight-stack copy through host
//! memory. The decode is one custom Metal launch per page: it reads the
//! packed words once and writes the bfloat16 weights once, with no
//! intermediate arrays.
//!
//! The previous decoder expressed the same arithmetic as an MLX graph of
//! gathers, shifts, and masks. Even compiled, that graph materialized four
//! full-width uint32 temporaries per projection (about 1.5 GB per
//! complete-layer projection, 9 GB before expert blocking), ran at about
//! 10 GB/s of decoded output (146 ms per complete-layer projection), and
//! forced page admission to reserve the temporaries and block-assembly copies
//! before any expert could load. The custom kernel is bandwidth-bound on the
//! output write, needs no workspace, and lets admission price only the packed
//! source plus the decoded output.
//!
//! Serving uses this decoder only for multi-token prefill; decode tokens
//! multiply straight from the packed codes through
//! `fused_gathered_projection` and never materialize weights.

use std::cell::RefCell;
use std::collections::HashMap;

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};
use mlx_c_rust::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
};

use super::codebook;
use super::hadamard32;

mod kernel_sources;

const CODEBOOK_DECODE_OPERATION: &str = "decode jangtq2 codebook expert rows";
const HADAMARD_ACTIVATION_ROTATION_OPERATION: &str = "rotate jangtq2 expert activations";
/// Threads per threadgroup for the one-dimensional launch. Every page has a
/// row count times group count that is a multiple of 256 on the served
/// geometry, and the launcher dispatches exact thread counts regardless.
const THREADGROUP_SIZE: i32 = 256;

/// Decodes packed codebook rows and applies each row's float16 scale.
///
/// The returned bfloat16 weights remain in the format's stored basis. When
/// `activation_rotation` is Hadamard-32, callers rotate the corresponding
/// projection activations with [`rotate_activations_hadamard32`] before the
/// existing dense expert multiplication. Keeping that rotation on activations
/// avoids materializing a second full expert-weight tensor in the native basis.
///
/// The decoded page is evaluated and detached before it is returned so the
/// retained output never pins its packed source in the allocator cache.
pub fn decode_expert_weight_rows(
    runtime: &MlxRuntime,
    packed_code_words: &MlxArray,
    row_scales: &MlxArray,
    bit_width: u32,
) -> Result<MlxArray, MlxRuntimeError> {
    if !codebook::is_supported_bit_width(bit_width) {
        return Err(decode_error(format!(
            "unsupported jangtq2 codebook bit width {bit_width}"
        )));
    }
    if packed_code_words.dtype() != MlxDtype::UInt32 {
        return Err(decode_error(format!(
            "packed code words must use uint32, got {:?}",
            packed_code_words.dtype()
        )));
    }
    if row_scales.dtype() != MlxDtype::Float16 {
        return Err(decode_error(format!(
            "row scales must use float16, got {:?}",
            row_scales.dtype()
        )));
    }

    let packed_shape = packed_code_words.shape();
    let scale_shape = row_scales.shape();
    if packed_shape.len() != 3 || scale_shape.len() != 2 {
        return Err(decode_error(format!(
            "packed tensors must have rank three and row scales rank two, got ranks {} and {}",
            packed_shape.len(),
            scale_shape.len()
        )));
    }
    if packed_shape[0] <= 0
        || packed_shape[1] <= 0
        || packed_shape[2] <= 0
        || packed_shape[0] != scale_shape[0]
        || packed_shape[1] != scale_shape[1]
    {
        return Err(decode_error(format!(
            "packed code shape {packed_shape:?} and row scale shape {scale_shape:?} disagree"
        )));
    }

    let packed_source_shape = packed_shape
        .iter()
        .map(|&dimension| {
            usize::try_from(dimension)
                .map_err(|_| decode_error("packed shape exceeds the host index range".to_owned()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let selected_expert_count = packed_source_shape[0];
    let decoded_weight_shape =
        codebook::decoded_weight_page_shape(&packed_source_shape, bit_width, selected_expert_count)
            .ok_or_else(|| {
                decode_error("packed source shape is not a valid codebook page".to_owned())
            })?;
    let input_width = decoded_weight_shape[2];
    if !hadamard32::is_applicable_to_input_width(input_width) {
        return Err(decode_error(format!(
            "decoded input width {input_width} is not aligned to the jangtq2 packing width"
        )));
    }
    let input_width_i32 = i32::try_from(input_width)
        .map_err(|_| decode_error("decoded input width exceeds the MLX shape range".to_owned()))?;
    let total_row_count = packed_shape[0]
        .checked_mul(packed_shape[1])
        .ok_or_else(|| decode_error("page row count exceeds the MLX grid range".to_owned()))?;
    let thread_count = total_row_count
        .checked_mul(input_width_i32 / hadamard32::HADAMARD_BLOCK_WIDTH as i32)
        .ok_or_else(|| decode_error("page decode grid exceeds the MLX grid range".to_owned()))?;
    let bits_i32 = i32::try_from(bit_width)
        .map_err(|_| decode_error("bit width exceeds the template range".to_owned()))?;

    let decoded_weight_bfloat16 = PAGE_DECODE_KERNEL.with(|page_decode_kernel| {
        let mut page_decode_kernel = page_decode_kernel.borrow_mut();
        let page_decode_kernel = match page_decode_kernel.as_mut() {
            Some(page_decode_kernel) => page_decode_kernel,
            None => page_decode_kernel.insert(PageDecodeKernel::new()?),
        };
        let codebook_level_table = page_decode_kernel.codebook_level_table(bit_width)?;
        let template_arguments = [
            MlxMetalKernelTemplateArgument::Dtype {
                name: "OutputT",
                dtype: MlxDtype::BFloat16,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "bits",
                integer_template_argument: bits_i32,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "input_width",
                integer_template_argument: input_width_i32,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "words_per_row",
                integer_template_argument: packed_shape[2],
            },
        ];
        let mut outputs = runtime
            .apply_metal_kernel(
                &page_decode_kernel.kernel,
                &[packed_code_words, row_scales, &codebook_level_table],
                &[MlxMetalKernelOutput::new(
                    vec![packed_shape[0], packed_shape[1], input_width_i32],
                    MlxDtype::BFloat16,
                )],
                [thread_count, 1, 1],
                [THREADGROUP_SIZE, 1, 1],
                &template_arguments,
            )
            .map_err(MlxRuntimeError::from)?;
        outputs
            .pop()
            .ok_or_else(|| decode_error("the page decode kernel returned no output".to_owned()))
    })?;
    materialize_and_detach_jangtq2_array(runtime, &decoded_weight_bfloat16)?;
    Ok(decoded_weight_bfloat16)
}

/// The compiled page decode kernel plus the per-width codebook level tables
/// it reads, built once per inference thread.
struct PageDecodeKernel {
    kernel: MlxMetalKernel,
    codebook_level_tables_by_bit_width: HashMap<u32, MlxArray>,
}

thread_local! {
    /// Inference decodes run on the inference owner thread; the kernel owner
    /// stays thread-local because MLX handles are not shared across threads.
    static PAGE_DECODE_KERNEL: RefCell<Option<PageDecodeKernel>> = const { RefCell::new(None) };
}

impl PageDecodeKernel {
    fn new() -> Result<Self, MlxRuntimeError> {
        let kernel = MlxMetalKernel::new(
            "jangtq2_page_decode",
            &["packed_weights", "row_scales", "codebook_levels"],
            &["decoded"],
            kernel_sources::PAGE_DECODE_KERNEL_SOURCE,
        )
        .map_err(MlxRuntimeError::from)?;
        Ok(Self {
            kernel,
            codebook_level_tables_by_bit_width: HashMap::new(),
        })
    }

    /// Returns the level table for one width, uploading it on first use so
    /// a complete-layer decode of three projections never repeats the copy.
    fn codebook_level_table(&mut self, bit_width: u32) -> Result<MlxArray, MlxRuntimeError> {
        if let Some(level_table) = self.codebook_level_tables_by_bit_width.get(&bit_width) {
            return level_table.retain().map_err(MlxRuntimeError::from);
        }
        let levels = codebook::codebook_levels(bit_width).ok_or_else(|| {
            decode_error(format!(
                "unsupported jangtq2 codebook bit width {bit_width}"
            ))
        })?;
        let level_count = i32::try_from(levels.len())
            .map_err(|_| decode_error("level table exceeds the MLX shape range".to_owned()))?;
        let level_table = MlxArray::from_f32(&levels, &[level_count])?;
        level_table.evaluate()?;
        let retained = level_table.retain()?;
        self.codebook_level_tables_by_bit_width
            .insert(bit_width, level_table);
        Ok(retained)
    }
}

fn materialize_and_detach_jangtq2_array(
    runtime: &MlxRuntime,
    array: &MlxArray,
) -> Result<(), MlxRuntimeError> {
    runtime.evaluate_arrays(&[array])?;
    astronomical_mlx_c_fork::detach_array(array).map_err(|error| {
        MlxRuntimeError::RuntimeOperation {
            operation: "detach a jangtq2 array",
            description: error.to_string(),
        }
    })
}

/// Applies the blockwise normalized Hadamard-32 to an activation's final axis.
///
/// Reshaping the last axis into 32-wide blocks is essential: a Hadamard over the
/// complete hidden width is a different transform from jangtq2's blockwise one.
pub fn rotate_activations_hadamard32(
    runtime: &MlxRuntime,
    activations: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let activation_shape = activations.shape();
    let Some(&input_width_i32) = activation_shape.last() else {
        return Err(rotation_error(
            "activation arrays must have a non-empty shape".to_owned(),
        ));
    };
    if input_width_i32 <= 0 || !hadamard32::is_applicable_to_input_width(input_width_i32 as usize) {
        return Err(rotation_error(format!(
            "activation input width {input_width_i32} is not divisible by {}",
            hadamard32::HADAMARD_BLOCK_WIDTH
        )));
    }
    let mut blocked_activation_shape = activation_shape.clone();
    let Some(last_axis_shape) = blocked_activation_shape.last_mut() else {
        return Err(rotation_error(
            "activation arrays must have a non-empty shape".to_owned(),
        ));
    };
    *last_axis_shape /= hadamard32::HADAMARD_BLOCK_WIDTH as i32;
    blocked_activation_shape.push(hadamard32::HADAMARD_BLOCK_WIDTH as i32);

    let activation_dtype = activations.dtype();
    let activations_float32 = runtime.astype(activations, MlxDtype::Float32)?;
    let blocked_activations = runtime.reshape(&activations_float32, &blocked_activation_shape)?;
    let rotated_blocks = runtime.hadamard_transform(&blocked_activations, None)?;
    let rotated_activation_float32 = runtime.reshape(&rotated_blocks, &activation_shape)?;
    // The rotation stays lazy on purpose: the projection kernel consumes it in
    // the same graph, and forcing an evaluation here drained the whole pending
    // decode graph once per expert projection (~109 synchronous waits per
    // token on the 122B bundle, about 1 ms each), which dominated decode time.
    runtime
        .astype(&rotated_activation_float32, activation_dtype)
        .map_err(MlxRuntimeError::from)
}

fn decode_error(description: String) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: CODEBOOK_DECODE_OPERATION,
        description,
    }
}

fn rotation_error(description: String) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: HADAMARD_ACTIVATION_ROTATION_OPERATION,
        description,
    }
}
