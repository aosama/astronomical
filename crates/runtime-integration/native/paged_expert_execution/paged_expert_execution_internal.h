#pragma once

#include <array>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "mlx/array.h"

namespace astronomical::paged_expert_execution {

inline constexpr int kProjectionCount = 3;

// One affine projection's exact artifact profile. Gate, up, and down may use
// different bit widths or group sizes within the same expert.
struct QuantizedProjectionArrays {
  mlx::core::array packed_weight;
  mlx::core::array scales;
  mlx::core::array biases;
  int group_size;
  int bits;
};

// Semantic array owners for one independently retained expert page.
struct ExpertPageArrays {
  std::array<QuantizedProjectionArrays, kProjectionCount> projections;
};

// Native bfloat16 keeps one weight array per projection and never invents
// affine scales, biases, or hidden quantization.
struct NativeBfloat16ExpertPageArrays {
  std::array<mlx::core::array, kProjectionCount> projection_weights;
};

enum class ExpertPageStorageMode {
  QuantizedAffine,
  NativeBfloat16,
};

// Immutable bridge between cache policy and lazy Metal execution.
//
// The Metal table stores graphics-processor addresses only; the optional array
// vectors are the actual lifetime owners. Every primitive captures a shared
// snapshot so a later publication or eviction cannot invalidate those addresses.
struct PageTableSnapshot {
  uint64_t generation;
  size_t expert_capacity;
  size_t resident_expert_count;
  ExpertPageStorageMode storage_mode;
  std::vector<std::optional<ExpertPageArrays>> expert_pages;
  std::vector<std::optional<NativeBfloat16ExpertPageArrays>>
      native_bfloat16_expert_pages;
  mlx::core::array metal_page_table;
};

size_t page_table_metadata_byte_count(size_t expert_capacity);

std::shared_ptr<const PageTableSnapshot> publish_snapshot(
    size_t expert_capacity,
    uint64_t generation,
    const std::vector<std::pair<size_t, ExpertPageArrays>>& source_pages);

std::shared_ptr<const PageTableSnapshot> publish_native_bfloat16_snapshot(
    size_t expert_capacity,
    uint64_t generation,
    const std::vector<
        std::pair<size_t, NativeBfloat16ExpertPageArrays>>& source_pages);

mlx::core::array build_paged_quantized_product(
    const std::shared_ptr<const PageTableSnapshot>& snapshot,
    const std::optional<std::vector<size_t>>& selected_expert_ids,
    int projection_index,
    const mlx::core::array& activations,
    const mlx::core::array& selected_indices,
    bool sorted_indices,
    mlx::core::Stream stream);

mlx::core::array build_paged_native_bfloat16_product(
    const std::shared_ptr<const PageTableSnapshot>& snapshot,
    const std::optional<std::vector<size_t>>& selected_expert_ids,
    int projection_index,
    const mlx::core::array& activations,
    const mlx::core::array& selected_indices,
    mlx::core::Stream stream);

bool dispatch_paged_quantized_matrix_nax(
    const PageTableSnapshot& snapshot,
    const std::optional<std::vector<size_t>>& selected_expert_ids,
    int projection_index,
    int group_size,
    int bits,
    const mlx::core::array& activations,
    mlx::core::Dtype scale_dtype,
    mlx::core::Dtype bias_dtype,
    const mlx::core::array& selected_indices,
    mlx::core::array& output,
    int matrix_row_count,
    int output_dimension,
    int input_dimension,
    mlx::core::Stream stream);

std::vector<const mlx::core::array*> routed_quantized_projection_resources(
    const PageTableSnapshot& snapshot,
    const std::optional<std::vector<size_t>>& selected_expert_ids,
    int projection_index);

}  // namespace astronomical::paged_expert_execution
