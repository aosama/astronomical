//! Raw Metal kernel sources for the fused expert decode path.
//!
//! Split from the kernel owners and launch logic in the parent so each file
//! stays inside the source-size budget; the sources are pure string
//! constants consumed once by `FusedExpertDecodeKernels::new`.

pub(super) const FUSED_DECODE_KERNEL_HEADER: &str = r#"
#include <metal_atomic>
using namespace metal;

template <typename ScaleRowT, typename HiddenT>
METAL_FUNC float quantized_row_dot(
    const device uint32_t* packed_row,
    ScaleRowT scale_row,
    ScaleRowT bias_row,
    const device HiddenT* hidden,
    int groups_per_output,
    int group_size,
    int words_per_group) {
  float dot = 0.0f;
  for (int group_index = 0; group_index < groups_per_output; ++group_index) {
    float scale = float(scale_row[group_index]);
    float bias = float(bias_row[group_index]);
    const device uint32_t* packed_group = packed_row + group_index * words_per_group;
    int element_base = group_index * group_size;
    for (int word_index = 0; word_index < words_per_group; ++word_index) {
      uint32_t packed_word = packed_group[word_index];
      int element_base_word = element_base + word_index * 8;
      for (int nibble_index = 0; nibble_index < 8; ++nibble_index) {
        uint32_t byte_value = (packed_word >> (8u * (nibble_index / 2u))) & 0xFFu;
        uint32_t nibble_value =
            ((nibble_index % 2u) == 0u) ? (byte_value & 0xFu) : (byte_value >> 4u);
        dot += (scale * float(nibble_value) + bias) *
            float(hidden[element_base_word + nibble_index]);
      }
    }
  }
  return dot;
}
"#;

pub(super) const VALUE_EXPERT_KERNEL_SOURCE: &str = r#"
  auto global_index = thread_position_in_grid.x;
  auto assignment_index = global_index / out_dimension;
  auto output_index = global_index % out_dimension;
  auto expert_index = expert_indices[assignment_index];
  auto weight_row_offset = (expert_index * out_dimension + output_index) * words_per_output;
  auto scale_row_offset = (expert_index * out_dimension + output_index) * groups_per_output;
  float dot = quantized_row_dot(
      packed_weights + weight_row_offset,
      scales + scale_row_offset,
      biases + scale_row_offset,
      hidden,
      groups_per_output,
      group_size,
      words_per_group);
  float activated = dot / (1.0f + metal::exp(-dot));
  float contribution = activated * float(scores[assignment_index]);
  metal::atomic_fetch_add_explicit(
      &weighted_outputs[output_index], contribution, metal::memory_order_relaxed);
"#;

pub(super) const ROUTED_GATE_UP_KERNEL_SOURCE: &str = r#"
  auto global_index = thread_position_in_grid.x;
  auto assignment_index = global_index / intermediate_dimension;
  auto output_index = global_index % intermediate_dimension;
  auto expert_index = expert_indices[assignment_index];
  auto gate_row = expert_index * fused_dimension + output_index;
  auto up_row = gate_row + intermediate_dimension;
  float gate_dot = quantized_row_dot(
      packed_weights + gate_row * words_per_output,
      scales + gate_row * groups_per_output,
      biases + gate_row * groups_per_output,
      hidden,
      groups_per_output,
      group_size,
      words_per_group);
  float up_dot = quantized_row_dot(
      packed_weights + up_row * words_per_output,
      scales + up_row * groups_per_output,
      biases + up_row * groups_per_output,
      hidden,
      groups_per_output,
      group_size,
      words_per_group);
  float gate_activated = gate_dot / (1.0f + metal::exp(-gate_dot));
  swiglu_hidden[global_index] = (OutputT)(gate_activated * up_dot);
"#;

pub(super) const ROUTED_DOWN_KERNEL_SOURCE: &str = r#"
  auto global_index = thread_position_in_grid.x;
  auto assignment_index = global_index / out_dimension;
  auto output_index = global_index % out_dimension;
  auto expert_index = expert_indices[assignment_index];
  auto weight_row_offset = (expert_index * out_dimension + output_index) * words_per_output;
  auto scale_row_offset = (expert_index * out_dimension + output_index) * groups_per_output;
  float dot = quantized_row_dot(
      packed_weights + weight_row_offset,
      scales + scale_row_offset,
      biases + scale_row_offset,
      swiglu_hidden + assignment_index * intermediate_dimension,
      groups_per_output,
      group_size,
      words_per_group);
  float contribution = dot * float(scores[assignment_index]);
  metal::atomic_fetch_add_explicit(
      &weighted_outputs[output_index], contribution, metal::memory_order_relaxed);
"#;
