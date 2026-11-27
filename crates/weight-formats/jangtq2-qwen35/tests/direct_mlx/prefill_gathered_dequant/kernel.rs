//! The prototype prefill gathered-dequant kernel and its launcher.
//!
//! Split from the measurement harness so each file stays inside the
//! source-size budget; the kernel source is a `.metal` file and the
//! launcher is the only bridge between fixture tensors and the launch.

use mlx_bridge::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
};
use runtime_integration::{MlxRuntime, MlxRuntimeError};

/// One threadgroup owns one expert and one block of output rows; 64 rows
/// divides every projection width of the family (1024 and 3072).
pub(super) const ROW_BLOCK_SIZE: i32 = 64;
pub(super) const THREADS_PER_THREADGROUP: i32 = ROW_BLOCK_SIZE;

/// The prototype kernel's declaration, generated from its `.metal` file under
/// `tests/direct_mlx/metal/`, where its description and template parameters
/// are written.
mod test_metal_kernels {
    include!(concat!(env!("OUT_DIR"), "/test_metal_kernels.rs"));
}

pub(super) use test_metal_kernels::ASTRO_FORMAT_JANGTQ2_PROTOTYPE_PREFILL_GATHERED_DEQUANT;

/// The data-plane arguments of one gathered-dequant launch, mirroring the
/// Metal kernel's input buffers and scalars one-to-one. `runtime` and
/// `kernel` stay explicit launcher arguments as the execution plane.
#[derive(Clone, Copy)]
pub(super) struct PrefillGatheredDequantInputs<'a> {
    pub(super) assignments: &'a MlxArray,
    pub(super) expert_assignment_starts: &'a MlxArray,
    pub(super) packed_code_words: &'a MlxArray,
    pub(super) row_scales: &'a MlxArray,
    pub(super) codebook_level_table: &'a MlxArray,
    pub(super) bit_width: u32,
    pub(super) expert_count: usize,
    pub(super) output_rows: usize,
}

pub(super) fn launch_prefill_gathered_dequant(
    runtime: &MlxRuntime,
    kernel: &MlxMetalKernel,
    inputs: PrefillGatheredDequantInputs<'_>,
) -> Result<MlxArray, MlxRuntimeError> {
    let PrefillGatheredDequantInputs {
        assignments,
        expert_assignment_starts,
        packed_code_words,
        row_scales,
        codebook_level_table,
        bit_width,
        expert_count,
        output_rows,
    } = inputs;
    let words_per_row = packed_code_words.shape()[2];
    let input_width = words_per_row * 32 / bit_width as i32;
    let assignment_count = assignments.shape()[0];
    let row_blocks = (output_rows as i32 + ROW_BLOCK_SIZE - 1) / ROW_BLOCK_SIZE;
    let template_arguments = [
        MlxMetalKernelTemplateArgument::Dtype {
            name: "OutputT",
            dtype: MlxDtype::Float32,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "bits",
            integer_template_argument: bit_width as i32,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "input_width",
            integer_template_argument: input_width,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "words_per_row",
            integer_template_argument: words_per_row,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "output_rows",
            integer_template_argument: output_rows as i32,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "row_block_size",
            integer_template_argument: ROW_BLOCK_SIZE,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "row_block_count",
            integer_template_argument: row_blocks,
        },
    ];
    let total_thread_count = expert_count as i32 * row_blocks * ROW_BLOCK_SIZE;
    let mut outputs = runtime.apply_metal_kernel(
        kernel,
        &[
            assignments,
            expert_assignment_starts,
            packed_code_words,
            row_scales,
            codebook_level_table,
        ],
        &[MlxMetalKernelOutput::new(
            vec![assignment_count, output_rows as i32],
            MlxDtype::Float32,
        )],
        [total_thread_count, 1, 1],
        [THREADS_PER_THREADGROUP, 1, 1],
        &template_arguments,
    )?;
    outputs
        .pop()
        .ok_or_else(|| MlxRuntimeError::RuntimeOperation {
            operation: "prototype prefill gathered dequant",
            description: "the kernel returned no output".to_owned(),
        })
}
