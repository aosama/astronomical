//! The prototype prefill gathered-dequant kernel and its launcher.
//!
//! Split from the measurement harness so each file stays inside the
//! source-size budget; the kernel source is a pure string constant and the
//! launcher is the only bridge between fixture tensors and the launch.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};
use mlx_c_rust::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
};

/// One threadgroup owns one expert and one block of output rows; 64 rows
/// divides every projection width of the family (1024 and 3072).
pub(super) const ROW_BLOCK_SIZE: i32 = 64;
pub(super) const THREADS_PER_THREADGROUP: i32 = ROW_BLOCK_SIZE;

/// The prototype kernel: a gathered matrix multiply that dequantizes packed
/// codebook codes in registers and applies the blockwise Hadamard-32 rotation
/// inside the K loop, so neither input is ever materialized in memory.
///
/// Template parameters: `OutputT` (float32 in the prototype), `bits`,
/// `input_width`, `words_per_row`, `output_rows`, `row_block_size`.
pub(super) const PREFILL_GATHERED_DEQUANT_KERNEL_SOURCE: &str = r#"
  constexpr int group_count = input_width / 32;
  constexpr uint32_t code_mask = (1u << bits) - 1u;
  // The launch grid counts THREADS, not threadgroups: one flat thread index
  // decomposes into (expert, row block, lane) with the block size and block
  // count as compile-time constants.
  const uint global_thread_index = thread_position_in_grid.x;
  const uint block_index = global_thread_index / (uint)row_block_size;
  const uint expert_index = block_index / (uint)row_block_count;
  const uint row_block_base = (block_index % (uint)row_block_count) * (uint)row_block_size;
  const uint lane_index = global_thread_index % (uint)row_block_size;
  const uint output_row = row_block_base + lane_index;
  const bool row_is_live = output_row < (uint)output_rows;

  float level_table[1 << bits];
  for (int code = 0; code < (1 << bits); ++code) {
    level_table[code] = float(codebook_levels[code]);
  }

  const uint assignment_start = expert_assignment_starts[expert_index];
  const uint assignment_end = expert_assignment_starts[expert_index + 1];
  const size_t packed_row_base =
      ((size_t)expert_index * (size_t)output_rows + (size_t)output_row) * (size_t)words_per_row;
  const float row_scale =
      row_is_live
          ? float(row_scales[(size_t)expert_index * (size_t)output_rows + (size_t)output_row])
          : 0.0f;

  threadgroup float activation_group[32];
  threadgroup float rotated_group[32];

  for (uint assignment_index = assignment_start; assignment_index < assignment_end;
       ++assignment_index) {
    const device float* activation_row =
        activations + (size_t)assignment_index * (size_t)input_width;
    float dot = 0.0f;
    for (int group_index = 0; group_index < group_count; ++group_index) {
      if (lane_index < 32u) {
        activation_group[lane_index] = activation_row[group_index * 32 + lane_index];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // The rotation is per assignment and block, shared by every row of the
      // threadgroup: 32 lanes each compute one rotated element (32 multiply-
      // accumulate operations), instead of the graph path's materialized
      // rotated copy of every activation row.
      if (lane_index < 32u) {
        float rotated_sum = 0.0f;
        for (int block_element = 0; block_element < 32; ++block_element) {
          const float sign =
              ((popcount(lane_index & (uint)block_element) & 1u) == 0u) ? 1.0f : -1.0f;
          rotated_sum += sign * activation_group[block_element];
        }
        rotated_group[lane_index] = rotated_sum * 0.17677669f;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (row_is_live) {
        uint32_t group_words[bits];
        for (int word_index = 0; word_index < bits; ++word_index) {
          group_words[word_index] =
              packed_weights[packed_row_base + (size_t)group_index * (size_t)bits + word_index];
        }
        for (int code_index = 0; code_index < 32; ++code_index) {
          const int start_bit = code_index * bits;
          const int word_index = start_bit >> 5;
          const int bit_offset = start_bit & 31;
          uint32_t spread_value = group_words[word_index] >> bit_offset;
          if (bit_offset + bits > 32) {
            spread_value |= group_words[word_index + 1] << (32 - bit_offset);
          }
          const uint32_t code = spread_value & code_mask;
          dot += level_table[code] * rotated_group[code_index];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row_is_live) {
      projected[(size_t)assignment_index * (size_t)output_rows + (size_t)output_row] =
          (OutputT)(dot * row_scale);
    }
  }
"#;

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
