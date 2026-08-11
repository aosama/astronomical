#include "native_expert_cache_internal.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <tuple>

#include "mlx/c/private/enums.h"

namespace astronomical::native_expert_cache {

// This translation unit owns startup validation and immutable source planning.
// It opens reusable readers and computes aligned one-expert slot layouts, but it
// never reads expert payload bytes or admits a cache entry.

namespace {

constexpr size_t kProjectionCount = 3;
constexpr size_t kParameterCount = 3;
constexpr size_t kRequiredAffineSourceCount =
    kProjectionCount * kParameterCount;
constexpr size_t kRequiredNativeBfloat16SourceCount = kProjectionCount;
constexpr size_t kSlotRegionAlignmentBytes = 16;

mx::fast::CustomKernelFunction make_expert_selection_bitmap_kernel() {
  // Router output may be strided and can contain repeated assignments. Reduce
  // it on the graphics processor so Rust never copies routed IDs through host
  // memory. The extra final word records any out-of-range assignment.
  return mx::fast::metal_kernel(
      "astronomical_native_expert_selection_bitmap",
      {"selected_indices"},
      {"selected_expert_bitmap"},
      R"METAL(
        uint assignment_index = thread_position_in_grid.x;
        uint expert_id = (uint)selected_indices[assignment_index];
        if (expert_id >= expert_capacity) {
          atomic_store_explicit(
              &selected_expert_bitmap[bitmap_word_count],
              1u,
              memory_order_relaxed);
          return;
        }
        uint bitmap_word_index = expert_id / 32;
        uint bitmap_bit_index = expert_id % 32;
        atomic_fetch_or_explicit(
            &selected_expert_bitmap[bitmap_word_index],
            1u << bitmap_bit_index,
            memory_order_relaxed);
      )METAL",
      "",
      true,
      true);
}

size_t checked_add(size_t left, size_t right, const char* description) {
  if (right > std::numeric_limits<size_t>::max() - left) {
    throw std::overflow_error(description);
  }
  return left + right;
}

size_t checked_multiply(
    size_t left,
    size_t right,
    const char* description) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
    throw std::overflow_error(description);
  }
  return left * right;
}

size_t align_up(size_t byte_count, size_t alignment) {
  const size_t remainder = byte_count % alignment;
  return remainder == 0
      ? byte_count
      : checked_add(
            byte_count,
            alignment - remainder,
            "expert slot alignment exceeds the host size range");
}

size_t shape_element_count(const mx::Shape& shape) {
  size_t element_count = 1;
  for (const auto dimension : shape) {
    if (dimension <= 0) {
      throw std::invalid_argument(
          "expert source shape dimensions must be positive");
    }
    element_count = checked_multiply(
        element_count,
        static_cast<size_t>(dimension),
        "expert source shape exceeds the host size range");
  }
  return element_count;
}

}  // namespace

NativeExpertCache::NativeExpertCache(
    const astronomical_native_expert_layer_descriptor* layer_descriptors,
    size_t layer_descriptor_count,
    uint64_t maximum_resident_payload_byte_count)
    : configured_maximum_resident_payload_byte_count_(
          maximum_resident_payload_byte_count),
      maximum_resident_payload_byte_count_(
          maximum_resident_payload_byte_count),
      expert_selection_bitmap_kernel_(
          make_expert_selection_bitmap_kernel()) {
  if (layer_descriptors == nullptr || layer_descriptor_count == 0 ||
      maximum_resident_payload_byte_count == 0) {
    throw std::invalid_argument(
        "native expert cache requires layers and a positive byte ceiling");
  }
  // Startup performs every shape, dtype, quantization, and file-range check.
  // Decode can then compute offsets with fixed metadata and fail only on an
  // actual read or memory-policy operation.
  layer_profiles_.reserve(layer_descriptor_count);
  for (size_t layer_index = 0; layer_index < layer_descriptor_count;
       ++layer_index) {
    const auto& source_layer = layer_descriptors[layer_index];
    const bool is_native_bfloat16_layer =
        source_layer.tensor_sources != nullptr &&
        source_layer.tensor_source_count > 0 &&
        source_layer.tensor_sources[0].quantization_group_size == 0 &&
        source_layer.tensor_sources[0].quantization_bits == 0;
    const size_t required_source_count = is_native_bfloat16_layer
        ? kRequiredNativeBfloat16SourceCount
        : kRequiredAffineSourceCount;
    if (source_layer.expert_capacity == 0 ||
        source_layer.tensor_sources == nullptr ||
        source_layer.tensor_source_count != required_source_count) {
      throw std::invalid_argument(
          "native expert layer has an incompatible tensor-source count");
    }
    LayerProfile layer_profile{
        source_layer.expert_capacity,
        {},
        {},
        is_native_bfloat16_layer
            ? paged::ExpertPageStorageMode::NativeBfloat16
             : paged::ExpertPageStorageMode::QuantizedAffine,
        0,
        0,
        {}};
    std::array<bool, kRequiredAffineSourceCount> observed_sources{};
    std::array<bool, kProjectionCount> observed_projection_profiles{};
    layer_profile.tensor_sources.reserve(source_layer.tensor_source_count);
    for (size_t source_index = 0;
         source_index < source_layer.tensor_source_count;
         ++source_index) {
      const auto& source = source_layer.tensor_sources[source_index];
      if (source.projection_index < 0 || source.projection_index >= 3 ||
          source.parameter_index < 0 || source.parameter_index >= 3 ||
          (is_native_bfloat16_layer && source.parameter_index != 0) ||
          (is_native_bfloat16_layer &&
           (source.quantization_group_size != 0 ||
            source.quantization_bits != 0)) ||
          (!is_native_bfloat16_layer &&
           (source.quantization_group_size == 0 ||
            source.quantization_bits == 0)) ||
          source.source_file_path == nullptr ||
          source.expert_shape == nullptr ||
          source.expert_shape_dimension_count == 0 ||
          source.bytes_per_expert == 0) {
        throw std::invalid_argument("native expert source descriptor is invalid");
      }
      const size_t source_identity =
          static_cast<size_t>(source.projection_index) * kParameterCount +
          static_cast<size_t>(source.parameter_index);
      if (observed_sources[source_identity]) {
        throw std::invalid_argument("native expert tensor source is duplicated");
      }
      observed_sources[source_identity] = true;
      const auto projection_index = static_cast<size_t>(source.projection_index);
      if (!observed_projection_profiles[projection_index]) {
        layer_profile.projection_quantization_group_sizes[projection_index] =
            source.quantization_group_size;
        layer_profile.projection_quantization_bits[projection_index] =
            source.quantization_bits;
        observed_projection_profiles[projection_index] = true;
      } else if (
          layer_profile.projection_quantization_group_sizes[projection_index] !=
              source.quantization_group_size ||
          layer_profile.projection_quantization_bits[projection_index] !=
              source.quantization_bits) {
        throw std::invalid_argument(
            "native expert projection sources disagree about quantization");
      }

      std::string source_file_path(source.source_file_path);
      auto opened_source_file_iterator =
          opened_source_files_.find(source_file_path);
      if (opened_source_file_iterator == opened_source_files_.end()) {
        std::error_code file_size_error;
        const auto file_size_bytes =
            std::filesystem::file_size(source_file_path, file_size_error);
        auto source_reader =
            std::make_shared<mx::io::ParallelFileReader>(source_file_path);
        if (file_size_error || !source_reader->is_open() ||
            !source_reader->good()) {
          throw std::runtime_error("native expert source file could not be opened");
        }
        auto opened_source_file = std::make_shared<OpenedSourceFile>(
            file_size_bytes, std::move(source_reader));
        opened_source_file_iterator =
            opened_source_files_
                .emplace(source_file_path, std::move(opened_source_file))
                .first;
      }

      mx::Shape expert_shape(
          source.expert_shape,
          source.expert_shape + source.expert_shape_dimension_count);
      const auto dtype = mlx_dtype_to_cpp(source.dtype);
      if (is_native_bfloat16_layer && dtype != mx::bfloat16) {
        throw std::invalid_argument(
            "native bfloat16 expert source must use bfloat16");
      }
      const size_t expected_byte_count = checked_multiply(
          shape_element_count(expert_shape),
          dtype.size(),
          "expert tensor byte count exceeds the host size range");
      if (expected_byte_count != source.bytes_per_expert) {
        throw std::invalid_argument(
            "expert tensor source shape does not match bytes per expert");
      }
      const uint64_t complete_tensor_byte_count = checked_multiply(
          source.bytes_per_expert,
          source_layer.expert_capacity,
          "expert tensor source exceeds the host size range");
      if (source.tensor_payload_offset >
          opened_source_file_iterator->second->file_size_bytes ||
          complete_tensor_byte_count >
              opened_source_file_iterator->second->file_size_bytes -
                  source.tensor_payload_offset) {
        throw std::invalid_argument(
            "expert tensor source range exceeds its file");
      }
      layer_profile.tensor_sources.push_back(TensorSource{
          source.projection_index,
          source.parameter_index,
          std::move(source_file_path),
          opened_source_file_iterator->second,
          source.tensor_payload_offset,
          source.bytes_per_expert,
          std::move(expert_shape),
          dtype,
          0});
    }
    std::sort(
        layer_profile.tensor_sources.begin(),
        layer_profile.tensor_sources.end(),
        [](const TensorSource& left, const TensorSource& right) {
          return std::tie(left.projection_index, left.parameter_index) <
              std::tie(right.projection_index, right.parameter_index);
        });
    // One page contains all gate, up, and down parameters for one expert.
    // Independent aligned regions make every typed MLX view valid even when the
    // original safetensors payload offset was not aligned for its dtype.
    size_t next_slot_byte_offset = 0;
    for (auto& tensor_source : layer_profile.tensor_sources) {
      const size_t alignment =
          std::max(
              kSlotRegionAlignmentBytes,
              static_cast<size_t>(tensor_source.dtype.size()));
      tensor_source.slot_byte_offset =
          align_up(next_slot_byte_offset, alignment);
      next_slot_byte_offset = checked_add(
          tensor_source.slot_byte_offset,
          tensor_source.bytes_per_expert,
          "expert slot exceeds the host size range");
    }
    layer_profile.slot_byte_count =
        align_up(next_slot_byte_offset, kSlotRegionAlignmentBytes);
    for (const auto& tensor_source : layer_profile.tensor_sources) {
      layer_profile.payload_byte_count_per_expert = checked_add(
          layer_profile.payload_byte_count_per_expert,
          tensor_source.bytes_per_expert,
          "expert payload exceeds the host size range");
    }
    layer_profiles_.push_back(std::move(layer_profile));
  }
  cumulative_statistics_.maximum_resident_payload_byte_count =
      maximum_resident_payload_byte_count_;
}

}  // namespace astronomical::native_expert_cache
