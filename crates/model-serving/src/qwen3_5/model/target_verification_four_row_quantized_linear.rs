//! Four-row 4-bit split-K quantized matmul for MTP target-verification windows.
//!
//! Stock MLX qmv is tuned for M=1 decode. A depth-three verify window is M=4:
//! one threadgroup-per-column-tile with K split across simdgroups keeps occupancy
//! high without a second full weight stream. Prefill and one-token decode stay
//! on the existing routes.

use astronomical_runtime_integration::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
    MlxRuntime, MlxRuntimeError,
};

const FOUR_ROW_COUNT: i32 = 4;
const COLUMN_TILE: i32 = 4;
const LARGE_OUTPUT_DIMENSION_FOR_TWO_WAY_SPLIT: i32 = 4_096;

const FOUR_ROW_SPLIT_K_SOURCE: &str = r#"
using namespace metal;

constexpr int ROW_COUNT = 4;
constexpr int COLUMN_TILE = 4;
constexpr int ACCUMULATOR_COUNT = COLUMN_TILE * ROW_COUNT;
constexpr int PACK_VALUES = 8;

uint split_part = simdgroup_index_in_threadgroup;
uint simd_lane = thread_index_in_simdgroup;
uint output_tile = threadgroup_position_in_grid.y;

int packs_along_input = K_SIZE / PACK_VALUES;
int groups_along_input = K_SIZE / GS;
int packs_per_split = packs_along_input / K_PARTS;
int first_output_column = int(output_tile) * COLUMN_TILE;
int pack_begin = int(split_part) * packs_per_split;
int pack_end = (int(split_part) == K_PARTS - 1) ? packs_along_input : pack_begin + packs_per_split;

float accumulators[ACCUMULATOR_COUNT];
for (int accumulator_index = 0; accumulator_index < ACCUMULATOR_COUNT; ++accumulator_index) {
  accumulators[accumulator_index] = 0.0f;
}

using PackedActivation = vec<T, PACK_VALUES>;
const device PackedActivation* packed_activations =
    (const device PackedActivation*)activations;

for (int pack_index = pack_begin + int(simd_lane); pack_index < pack_end; pack_index += 32) {
  int input_offset = pack_index * PACK_VALUES;
  int group_index = input_offset / GS;
  PackedActivation row0 = packed_activations[(0 * K_SIZE + input_offset) / PACK_VALUES];
  PackedActivation row1 = packed_activations[(1 * K_SIZE + input_offset) / PACK_VALUES];
  PackedActivation row2 = packed_activations[(2 * K_SIZE + input_offset) / PACK_VALUES];
  PackedActivation row3 = packed_activations[(3 * K_SIZE + input_offset) / PACK_VALUES];
  uint packed0 = packed_weights[(first_output_column + 0) * packs_along_input + pack_index];
  uint packed1 = packed_weights[(first_output_column + 1) * packs_along_input + pack_index];
  uint packed2 = packed_weights[(first_output_column + 2) * packs_along_input + pack_index];
  uint packed3 = packed_weights[(first_output_column + 3) * packs_along_input + pack_index];
  float scale0 = float(quantization_scales[(first_output_column + 0) * groups_along_input + group_index]);
  float scale1 = float(quantization_scales[(first_output_column + 1) * groups_along_input + group_index]);
  float scale2 = float(quantization_scales[(first_output_column + 2) * groups_along_input + group_index]);
  float scale3 = float(quantization_scales[(first_output_column + 3) * groups_along_input + group_index]);
  float bias0 = float(quantization_biases[(first_output_column + 0) * groups_along_input + group_index]);
  float bias1 = float(quantization_biases[(first_output_column + 1) * groups_along_input + group_index]);
  float bias2 = float(quantization_biases[(first_output_column + 2) * groups_along_input + group_index]);
  float bias3 = float(quantization_biases[(first_output_column + 3) * groups_along_input + group_index]);
  {
    uint packed = packed0;
    float scale = scale0;
    float bias = bias0;
    for (int pack_value_index = 0; pack_value_index < PACK_VALUES; ++pack_value_index) {
      float dequantized = float((packed >> (pack_value_index * 4)) & 0xFu) * scale + bias;
      accumulators[0 * ROW_COUNT + 0] += float(row0[pack_value_index]) * dequantized;
      accumulators[0 * ROW_COUNT + 1] += float(row1[pack_value_index]) * dequantized;
      accumulators[0 * ROW_COUNT + 2] += float(row2[pack_value_index]) * dequantized;
      accumulators[0 * ROW_COUNT + 3] += float(row3[pack_value_index]) * dequantized;
    }
  }
  {
    uint packed = packed1;
    float scale = scale1;
    float bias = bias1;
    for (int pack_value_index = 0; pack_value_index < PACK_VALUES; ++pack_value_index) {
      float dequantized = float((packed >> (pack_value_index * 4)) & 0xFu) * scale + bias;
      accumulators[1 * ROW_COUNT + 0] += float(row0[pack_value_index]) * dequantized;
      accumulators[1 * ROW_COUNT + 1] += float(row1[pack_value_index]) * dequantized;
      accumulators[1 * ROW_COUNT + 2] += float(row2[pack_value_index]) * dequantized;
      accumulators[1 * ROW_COUNT + 3] += float(row3[pack_value_index]) * dequantized;
    }
  }
  {
    uint packed = packed2;
    float scale = scale2;
    float bias = bias2;
    for (int pack_value_index = 0; pack_value_index < PACK_VALUES; ++pack_value_index) {
      float dequantized = float((packed >> (pack_value_index * 4)) & 0xFu) * scale + bias;
      accumulators[2 * ROW_COUNT + 0] += float(row0[pack_value_index]) * dequantized;
      accumulators[2 * ROW_COUNT + 1] += float(row1[pack_value_index]) * dequantized;
      accumulators[2 * ROW_COUNT + 2] += float(row2[pack_value_index]) * dequantized;
      accumulators[2 * ROW_COUNT + 3] += float(row3[pack_value_index]) * dequantized;
    }
  }
  {
    uint packed = packed3;
    float scale = scale3;
    float bias = bias3;
    for (int pack_value_index = 0; pack_value_index < PACK_VALUES; ++pack_value_index) {
      float dequantized = float((packed >> (pack_value_index * 4)) & 0xFu) * scale + bias;
      accumulators[3 * ROW_COUNT + 0] += float(row0[pack_value_index]) * dequantized;
      accumulators[3 * ROW_COUNT + 1] += float(row1[pack_value_index]) * dequantized;
      accumulators[3 * ROW_COUNT + 2] += float(row2[pack_value_index]) * dequantized;
      accumulators[3 * ROW_COUNT + 3] += float(row3[pack_value_index]) * dequantized;
    }
  }
}

for (int accumulator_index = 0; accumulator_index < ACCUMULATOR_COUNT; ++accumulator_index) {
  accumulators[accumulator_index] = simd_sum(accumulators[accumulator_index]);
}

threadgroup float split_partials[K_PARTS * ACCUMULATOR_COUNT];
if (simd_lane == 0) {
  for (int accumulator_index = 0; accumulator_index < ACCUMULATOR_COUNT; ++accumulator_index) {
    split_partials[int(split_part) * ACCUMULATOR_COUNT + accumulator_index] =
        accumulators[accumulator_index];
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (split_part == 0 && simd_lane < ACCUMULATOR_COUNT) {
  float total = 0.0f;
  for (int part_index = 0; part_index < K_PARTS; ++part_index) {
    total += split_partials[part_index * ACCUMULATOR_COUNT + int(simd_lane)];
  }
  int tile_column = int(simd_lane) / ROW_COUNT;
  int row = int(simd_lane) - tile_column * ROW_COUNT;
  projected_activations[row * N_SIZE + first_output_column + tile_column] = T(total);
}
"#;

pub fn four_row_split_k_quantized_linear_kernel() -> Result<MlxMetalKernel, MlxRuntimeError> {
    MlxMetalKernel::new(
        "astronomical_qwen3_5_four_row_split_k_quantized_linear",
        &[
            "activations",
            "packed_weights",
            "quantization_scales",
            "quantization_biases",
        ],
        &["projected_activations"],
        FOUR_ROW_SPLIT_K_SOURCE,
    )
}

pub(super) fn four_row_split_k_is_eligible(
    activations: &MlxArray,
    quantization_bits: i32,
    quantization_group_size: i32,
    input_dimension: i32,
    output_dimension: i32,
) -> bool {
    let activation_shape = activations.shape();
    quantization_bits == 4
        && matches!(quantization_group_size, 32 | 64 | 128)
        && matches!(activations.dtype(), MlxDtype::BFloat16 | MlxDtype::Float16)
        && activation_shape.len() == 3
        && activation_shape[0] == 1
        && activation_shape[1] == FOUR_ROW_COUNT
        && input_dimension > 0
        && input_dimension % 64 == 0
        && output_dimension > 0
        && output_dimension % COLUMN_TILE == 0
}

pub(super) fn project_four_row_split_k(
    runtime: &MlxRuntime,
    four_row_kernel: &MlxMetalKernel,
    activations: &MlxArray,
    packed_weight: &MlxArray,
    quantization_scales: &MlxArray,
    quantization_biases: &MlxArray,
    quantization_group_size: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let activation_shape = activations.shape();
    let input_dimension = activation_shape[2];
    let output_dimension = packed_weight.shape()[0];
    let split_part_count = if output_dimension >= LARGE_OUTPUT_DIMENSION_FOR_TWO_WAY_SPLIT {
        2
    } else {
        4
    };
    let row_major_activations = runtime.reshape(activations, &[FOUR_ROW_COUNT, input_dimension])?;
    let mut kernel_outputs = runtime.apply_metal_kernel(
        four_row_kernel,
        &[
            &row_major_activations,
            packed_weight,
            quantization_scales,
            quantization_biases,
        ],
        &[MlxMetalKernelOutput::new(
            vec![FOUR_ROW_COUNT, output_dimension],
            activations.dtype(),
        )],
        [32 * split_part_count, output_dimension / COLUMN_TILE, 1],
        [32 * split_part_count, 1, 1],
        &[
            MlxMetalKernelTemplateArgument::Dtype {
                name: "T",
                dtype: activations.dtype(),
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "GS",
                integer_template_argument: quantization_group_size,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "K_SIZE",
                integer_template_argument: input_dimension,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "N_SIZE",
                integer_template_argument: output_dimension,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "K_PARTS",
                integer_template_argument: split_part_count,
            },
        ],
    )?;
    let projected_rows = kernel_outputs
        .pop()
        .ok_or_else(|| MlxRuntimeError::RuntimeOperation {
            operation: "project four-row target-verification tokens",
            description: "four-row split-K kernel returned no projected activations".to_owned(),
        })?;
    runtime.reshape(&projected_rows, &[1, FOUR_ROW_COUNT, output_dimension])
}
