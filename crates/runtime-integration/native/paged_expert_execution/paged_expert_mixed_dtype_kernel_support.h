#pragma once

namespace astronomical::paged_expert_execution {

// MLX promotes affine activations, scales, and biases to one compute dtype.
// Paged storage retains artifact-native scale and bias widths, so these Metal
// helpers perform that lossless promotion while reading immutable page slots.
inline constexpr const char* kPagedExpertMixedDtypeKernelSupportSource = R"METAL(
template <
    typename ComputeT,
    typename ScaleT,
    typename BiasT,
    int group_size,
    int bits>
METAL_FUNC void astronomical_mixed_dtype_qmv_fast_impl(
    const device uint32_t* packed_weight,
    const device ScaleT* scales,
    const device BiasT* biases,
    const device ComputeT* activations,
    device ComputeT* output,
    const constant int& input_dimension,
    const constant int& output_dimension,
    uint3 threadgroup_position,
    uint simd_group_id,
    uint simd_lane_id) {
  constexpr int packed_values_per_thread = bits == 2 ? 1 : 2;
  constexpr int simdgroups_per_threadgroup = 2;
  constexpr int output_rows_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packed_values_per_thread;
  constexpr int input_block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* packed_weight_bytes =
      reinterpret_cast<const device uint8_t*>(packed_weight);
  typedef float AccumulatorT;
  thread AccumulatorT activation_values[values_per_thread];
  thread AccumulatorT partial_sums[output_rows_per_simdgroup] = {0};

  const int packed_weight_bytes_per_row =
      input_dimension * bytes_per_pack / pack_factor;
  const int affine_groups_per_row = input_dimension / group_size;
  const int first_output_row =
      threadgroup_position.y *
          (simdgroups_per_threadgroup * output_rows_per_simdgroup) +
      simd_group_id * output_rows_per_simdgroup;

  packed_weight_bytes +=
      first_output_row * packed_weight_bytes_per_row +
      simd_lane_id * packed_values_per_thread * bytes_per_pack;
  scales +=
      first_output_row * affine_groups_per_row +
      simd_lane_id / scale_step_per_thread;
  biases +=
      first_output_row * affine_groups_per_row +
      simd_lane_id / scale_step_per_thread;
  activations +=
      threadgroup_position.x * input_dimension +
      simd_lane_id * values_per_thread;
  output +=
      threadgroup_position.x * output_dimension + first_output_row;

  for (int input_offset = 0;
       input_offset < input_dimension;
       input_offset += input_block_size) {
    const AccumulatorT activation_sum =
        load_vector<ComputeT, AccumulatorT, values_per_thread, bits>(
            activations, activation_values);
    for (int output_row_offset = 0;
         output_row_offset < output_rows_per_simdgroup;
         ++output_row_offset) {
      const device uint8_t* packed_weight_row =
          packed_weight_bytes +
          output_row_offset * packed_weight_bytes_per_row;
      const device ScaleT* scale_row =
          scales + output_row_offset * affine_groups_per_row;
      const device BiasT* bias_row =
          biases + output_row_offset * affine_groups_per_row;
      const AccumulatorT scale = static_cast<AccumulatorT>(scale_row[0]);
      const AccumulatorT bias = static_cast<AccumulatorT>(bias_row[0]);
      partial_sums[output_row_offset] +=
          qdot<AccumulatorT, values_per_thread, bits>(
              packed_weight_row,
              activation_values,
              scale,
              bias,
              activation_sum);
    }
    packed_weight_bytes +=
        input_block_size * bytes_per_pack / pack_factor;
    scales += input_block_size / group_size;
    biases += input_block_size / group_size;
    activations += input_block_size;
  }

  for (int output_row_offset = 0;
       output_row_offset < output_rows_per_simdgroup;
       ++output_row_offset) {
    partial_sums[output_row_offset] = simd_sum(partial_sums[output_row_offset]);
    if (simd_lane_id == 0) {
      output[output_row_offset] =
          static_cast<ComputeT>(partial_sums[output_row_offset]);
    }
  }
}

template <
    typename ComputeT,
    typename ScaleT,
    typename BiasT,
    int group_size,
    int bits>
METAL_FUNC void astronomical_mixed_dtype_qmv_impl(
    const device uint32_t* packed_weight,
    const device ScaleT* scales,
    const device BiasT* biases,
    const device ComputeT* activations,
    device ComputeT* output,
    const constant int& input_dimension,
    const constant int& output_dimension,
    uint3 threadgroup_position,
    uint simd_group_id,
    uint simd_lane_id) {
  constexpr int simdgroups_per_threadgroup = 2;
  constexpr int output_rows_per_simdgroup = 4;
  constexpr int packed_values_per_thread = 1;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packed_values_per_thread;
  constexpr int input_block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* packed_weight_bytes =
      reinterpret_cast<const device uint8_t*>(packed_weight);
  typedef float AccumulatorT;
  thread AccumulatorT activation_values[values_per_thread];
  thread AccumulatorT partial_sums[output_rows_per_simdgroup] = {0};

  const int packed_weight_bytes_per_row =
      input_dimension * bytes_per_pack / pack_factor;
  const int affine_groups_per_row = input_dimension / group_size;
  const int first_output_row =
      threadgroup_position.y *
          (simdgroups_per_threadgroup * output_rows_per_simdgroup) +
      simd_group_id * output_rows_per_simdgroup;
  const int bounded_first_output_row =
      min(output_dimension - output_rows_per_simdgroup, first_output_row);
  if (first_output_row >= output_dimension) {
    return;
  }

  if (output_dimension <
      simdgroups_per_threadgroup * output_rows_per_simdgroup) {
    packed_weight_bytes +=
        first_output_row * packed_weight_bytes_per_row +
        simd_lane_id * packed_values_per_thread * bytes_per_pack;
    scales +=
        first_output_row * affine_groups_per_row +
        simd_lane_id / scale_step_per_thread;
    biases +=
        first_output_row * affine_groups_per_row +
        simd_lane_id / scale_step_per_thread;
    activations +=
        threadgroup_position.x * input_dimension +
        simd_lane_id * values_per_thread;
    output +=
        threadgroup_position.x * output_dimension + first_output_row;

    int input_offset = 0;
    for (;
         input_offset < input_dimension - input_block_size;
         input_offset += input_block_size) {
      const AccumulatorT activation_sum =
          load_vector<ComputeT, AccumulatorT, values_per_thread, bits>(
              activations, activation_values);
      for (int output_row_offset = 0;
           output_row_offset < output_rows_per_simdgroup &&
               first_output_row + output_row_offset < output_dimension;
           ++output_row_offset) {
        const device uint8_t* packed_weight_row =
            packed_weight_bytes +
            output_row_offset * packed_weight_bytes_per_row;
        const device ScaleT* scale_row =
            scales + output_row_offset * affine_groups_per_row;
        const device BiasT* bias_row =
            biases + output_row_offset * affine_groups_per_row;
        const AccumulatorT scale = static_cast<AccumulatorT>(scale_row[0]);
        const AccumulatorT bias = static_cast<AccumulatorT>(bias_row[0]);
        partial_sums[output_row_offset] +=
            qdot<AccumulatorT, values_per_thread, bits>(
                packed_weight_row,
                activation_values,
                scale,
                bias,
                activation_sum);
      }
      packed_weight_bytes +=
          input_block_size * bytes_per_pack / pack_factor;
      scales += input_block_size / group_size;
      biases += input_block_size / group_size;
      activations += input_block_size;
    }
    const int remaining_value_count = clamp(
        static_cast<int>(
            input_dimension - input_offset - simd_lane_id * values_per_thread),
        0,
        values_per_thread);
    if (remaining_value_count > 0) {
      const AccumulatorT activation_sum =
          load_vector_safe<ComputeT, AccumulatorT, values_per_thread, bits>(
              activations, activation_values, remaining_value_count);
      for (int output_row_offset = 0;
           output_row_offset < output_rows_per_simdgroup &&
               first_output_row + output_row_offset < output_dimension;
           ++output_row_offset) {
        const device uint8_t* packed_weight_row =
            packed_weight_bytes +
            output_row_offset * packed_weight_bytes_per_row;
        const device ScaleT* scale_row =
            scales + output_row_offset * affine_groups_per_row;
        const device BiasT* bias_row =
            biases + output_row_offset * affine_groups_per_row;
        const AccumulatorT scale = static_cast<AccumulatorT>(scale_row[0]);
        const AccumulatorT bias = static_cast<AccumulatorT>(bias_row[0]);
        partial_sums[output_row_offset] +=
            qdot_safe<AccumulatorT, values_per_thread, bits>(
                packed_weight_row,
                activation_values,
                scale,
                bias,
                activation_sum,
                remaining_value_count);
      }
    }
    for (int output_row_offset = 0;
         output_row_offset < output_rows_per_simdgroup &&
             first_output_row + output_row_offset < output_dimension;
         ++output_row_offset) {
      partial_sums[output_row_offset] =
          simd_sum(partial_sums[output_row_offset]);
      if (simd_lane_id == 0) {
        output[output_row_offset] =
            static_cast<ComputeT>(partial_sums[output_row_offset]);
      }
    }
    return;
  }

  packed_weight_bytes +=
      bounded_first_output_row * packed_weight_bytes_per_row +
      simd_lane_id * packed_values_per_thread * bytes_per_pack;
  scales +=
      bounded_first_output_row * affine_groups_per_row +
      simd_lane_id / scale_step_per_thread;
  biases +=
      bounded_first_output_row * affine_groups_per_row +
      simd_lane_id / scale_step_per_thread;
  activations +=
      threadgroup_position.x * input_dimension +
      simd_lane_id * values_per_thread;
  output +=
      threadgroup_position.x * output_dimension + bounded_first_output_row;

  int input_offset = 0;
  for (;
       input_offset < input_dimension - input_block_size;
       input_offset += input_block_size) {
    const AccumulatorT activation_sum =
        load_vector<ComputeT, AccumulatorT, values_per_thread, bits>(
            activations, activation_values);
    for (int output_row_offset = 0;
         output_row_offset < output_rows_per_simdgroup;
         ++output_row_offset) {
      const device uint8_t* packed_weight_row =
          packed_weight_bytes +
          output_row_offset * packed_weight_bytes_per_row;
      const device ScaleT* scale_row =
          scales + output_row_offset * affine_groups_per_row;
      const device BiasT* bias_row =
          biases + output_row_offset * affine_groups_per_row;
      const AccumulatorT scale = static_cast<AccumulatorT>(scale_row[0]);
      const AccumulatorT bias = static_cast<AccumulatorT>(bias_row[0]);
      partial_sums[output_row_offset] +=
          qdot<AccumulatorT, values_per_thread, bits>(
              packed_weight_row,
              activation_values,
              scale,
              bias,
              activation_sum);
    }
    packed_weight_bytes +=
        input_block_size * bytes_per_pack / pack_factor;
    scales += input_block_size / group_size;
    biases += input_block_size / group_size;
    activations += input_block_size;
  }
  const int remaining_value_count = clamp(
      static_cast<int>(
          input_dimension - input_offset - simd_lane_id * values_per_thread),
      0,
      values_per_thread);
  if (remaining_value_count > 0) {
    const AccumulatorT activation_sum =
        load_vector_safe<ComputeT, AccumulatorT, values_per_thread, bits>(
            activations, activation_values, remaining_value_count);
    for (int output_row_offset = 0;
         output_row_offset < output_rows_per_simdgroup;
         ++output_row_offset) {
      const device uint8_t* packed_weight_row =
          packed_weight_bytes +
          output_row_offset * packed_weight_bytes_per_row;
      const device ScaleT* scale_row =
          scales + output_row_offset * affine_groups_per_row;
      const device BiasT* bias_row =
          biases + output_row_offset * affine_groups_per_row;
      const AccumulatorT scale = static_cast<AccumulatorT>(scale_row[0]);
      const AccumulatorT bias = static_cast<AccumulatorT>(bias_row[0]);
      partial_sums[output_row_offset] +=
          qdot_safe<AccumulatorT, values_per_thread, bits>(
              packed_weight_row,
              activation_values,
              scale,
              bias,
              activation_sum,
              remaining_value_count);
    }
  }
  for (int output_row_offset = 0;
       output_row_offset < output_rows_per_simdgroup;
       ++output_row_offset) {
    partial_sums[output_row_offset] = simd_sum(partial_sums[output_row_offset]);
    if (simd_lane_id == 0) {
      output[output_row_offset] =
          static_cast<ComputeT>(partial_sums[output_row_offset]);
    }
  }
}

template <
    typename ComputeT,
    typename ScaleT,
    typename BiasT,
    short source_rows,
    short source_columns,
    short destination_leading_dimension,
    short reduction_dimension,
    short threadgroup_size,
    short group_size,
    short bits>
struct AstronomicalQuantizedBlockLoader {
  static_assert(
      source_columns <= group_size,
      "The group size should be larger than the columns");
  static_assert(
      group_size % source_columns == 0,
      "The group size should be divisible by the columns");

  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short packed_source_columns = source_columns / pack_factor;
  MLX_MTL_CONST short read_count =
      (packed_source_columns * source_rows < threadgroup_size)
      ? 1
      : (packed_source_columns * source_rows) / threadgroup_size;
  MLX_MTL_CONST short group_steps = group_size / source_columns;

  const int source_leading_dimension;
  const int tile_stride;
  short group_step_count;
  const int group_stride;
  const short thread_index;
  const short source_row;
  const short packed_source_column;
  threadgroup ComputeT* destination;
  const device uint8_t* source;
  const device ScaleT* scales;
  const device BiasT* biases;

  AstronomicalQuantizedBlockLoader(
      const device uint8_t* source_pointer,
      const device ScaleT* scale_pointer,
      const device BiasT* bias_pointer,
      int source_leading_dimension_value,
      threadgroup ComputeT* destination_pointer,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]) thread
      : source_leading_dimension(source_leading_dimension_value),
        tile_stride(
            reduction_dimension
                ? packed_source_columns * bytes_per_pack
                : source_rows * source_leading_dimension * bytes_per_pack /
                    pack_factor),
        group_step_count(0),
        group_stride(source_rows * source_leading_dimension / group_size),
        thread_index(simd_group_id * SIMD_SIZE + simd_lane_id),
        source_row(read_count * thread_index / packed_source_columns),
        packed_source_column(
            (read_count * thread_index) % packed_source_columns),
        destination(
            destination_pointer + source_row * destination_leading_dimension +
            packed_source_column * pack_factor),
        source(
            source_pointer +
            source_row * source_leading_dimension * bytes_per_pack /
                pack_factor +
            packed_source_column * bytes_per_pack),
        scales(
            scale_pointer +
            source_row * source_leading_dimension / group_size),
        biases(
            bias_pointer +
            source_row * source_leading_dimension / group_size) {}

  void load_unsafe() const thread {
    if (packed_source_columns * source_rows < threadgroup_size &&
        source_row >= source_rows) {
      return;
    }
    const ComputeT scale = static_cast<ComputeT>(*scales);
    const ComputeT bias = static_cast<ComputeT>(*biases);
    for (int read_index = 0; read_index < read_count; ++read_index) {
      dequantize<ComputeT, pack_factor, bits>(
          source + read_index * bytes_per_pack,
          scale,
          bias,
          destination + read_index * pack_factor);
    }
  }

  void load_safe(short2 source_tile_dimensions) const thread {
    if (packed_source_columns * source_rows < threadgroup_size &&
        source_row >= source_rows) {
      return;
    }
    if ((reduction_dimension == 1 && source_row >= source_tile_dimensions.x) ||
        (reduction_dimension == 0 && source_row >= source_tile_dimensions.y)) {
      for (int destination_index = 0;
           destination_index < read_count * pack_factor;
           ++destination_index) {
        destination[destination_index] = ComputeT(0);
      }
      return;
    }
    const ComputeT scale = static_cast<ComputeT>(*scales);
    const ComputeT bias = static_cast<ComputeT>(*biases);
    for (int read_index = 0; read_index < read_count; ++read_index) {
      dequantize<ComputeT, pack_factor, bits>(
          source + read_index * bytes_per_pack,
          scale,
          bias,
          destination + read_index * pack_factor);
    }
  }

  void next() thread {
    source += tile_stride;
    if (reduction_dimension == 1) {
      if (group_steps > 1) {
        ++group_step_count;
        if (group_step_count == group_steps) {
          group_step_count = 0;
          ++scales;
          ++biases;
        }
      } else {
        ++scales;
        ++biases;
      }
    } else {
      scales += group_stride;
      biases += group_stride;
    }
  }
};
)METAL";

}  // namespace astronomical::paged_expert_execution
