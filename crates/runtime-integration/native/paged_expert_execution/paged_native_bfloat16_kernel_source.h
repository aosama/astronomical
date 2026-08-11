#pragma once

namespace astronomical::paged_expert_execution {

// Native BF16 uses the common nine-address page-table layout but reads only one
// weight address per projection. No affine parameters or hidden conversion exist.
inline constexpr const char* kPagedNativeBfloat16KernelSource = R"METAL(
template <typename T>
struct AstronomicalNativeExpertPageEntry {
  const device T* gate_weight;
  const device T* gate_unused_1;
  const device T* gate_unused_2;
  const device T* up_weight;
  const device T* up_unused_1;
  const device T* up_unused_2;
  const device T* down_weight;
  const device T* down_unused_1;
  const device T* down_unused_2;
  uint presence;
  uint generation;
};

template <typename T, int projection_index>
METAL_FUNC const device T* astronomical_native_projection_weight(
    const constant AstronomicalNativeExpertPageEntry<T>& expert_page) {
  if constexpr (projection_index == 0) {
    return expert_page.gate_weight;
  } else if constexpr (projection_index == 1) {
    return expert_page.up_weight;
  } else {
    return expert_page.down_weight;
  }
}

template <typename T, int projection_index>
[[kernel]] void astronomical_paged_gather_native_bfloat16_matrix(
    const constant AstronomicalNativeExpertPageEntry<T>* page_table [[buffer(0)]],
    const device T* activations [[buffer(1)]],
    const device uint* selected_indices [[buffer(2)]],
    device T* output [[buffer(3)]],
    const constant int& input_dimension [[buffer(4)]],
    const constant int& output_dimension [[buffer(5)]],
    uint2 threadgroup_position [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  const uint output_column = threadgroup_position.x;
  const uint assignment_index = threadgroup_position.y;
  const uint expert_id = selected_indices[assignment_index];
  const device T* weight = astronomical_native_projection_weight<T, projection_index>(
      page_table[expert_id]);
  weight += size_t(output_column) * input_dimension;
  activations += size_t(assignment_index) * input_dimension;
  float partial_sum = 0.0f;
  for (uint input_column = simd_lane_id;
       input_column < uint(input_dimension);
       input_column += 32) {
    partial_sum += float(activations[input_column]) * float(weight[input_column]);
  }
  const float output_sum = simd_sum(partial_sum);
  if (simd_lane_id == 0) {
    output[size_t(assignment_index) * output_dimension + output_column] = T(output_sum);
  }
}
)METAL";

}  // namespace astronomical::paged_expert_execution
