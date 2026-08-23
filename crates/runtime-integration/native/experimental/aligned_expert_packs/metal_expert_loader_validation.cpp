// Validates and allocates complete MLX destinations before Metal I/O submission
// so asynchronous work never receives malformed or partially covered ranges.

#include "metal_expert_loader_validation.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/enums.h"

std::vector<size_t> validate_output_tensors(
    const astronomical_metal_expert_loader_output_tensor* output_tensors,
    size_t output_tensor_count) {
  if (output_tensors == nullptr || output_tensor_count == 0) {
    throw std::invalid_argument(
        "at least one Metal I/O output tensor is required");
  }
  std::vector<size_t> output_tensor_byte_counts;
  output_tensor_byte_counts.reserve(output_tensor_count);
  for (size_t output_tensor_index = 0;
       output_tensor_index < output_tensor_count;
       ++output_tensor_index) {
    const auto& output_tensor = output_tensors[output_tensor_index];
    if (output_tensor.shape == nullptr || output_tensor.dimension_count <= 0) {
      throw std::invalid_argument("Metal I/O output tensor shape is invalid");
    }
    mlx::core::Shape output_shape(
        output_tensor.shape,
        output_tensor.shape + output_tensor.dimension_count);
    for (const auto output_dimension : output_shape) {
      if (output_dimension <= 0) {
        throw std::invalid_argument("Metal I/O output tensor shape is invalid");
      }
    }
    const auto output_dtype = mlx_dtype_to_cpp(output_tensor.dtype);
    output_tensor_byte_counts.push_back(
        mlx::core::array(output_shape, output_dtype, nullptr, {}).nbytes());
  }
  return output_tensor_byte_counts;
}

void validate_load_ranges(
    const astronomical_metal_expert_loader_load_range* load_ranges,
    size_t load_range_count,
    const std::vector<size_t>& output_tensor_byte_counts,
    uint64_t source_file_size_bytes) {
  if (load_ranges == nullptr || load_range_count == 0) {
    throw std::invalid_argument("at least one Metal I/O load range is required");
  }
  struct DestinationRange {
    size_t output_tensor_index;
    size_t start_offset_bytes;
    size_t end_offset_bytes;
  };
  std::vector<DestinationRange> destination_ranges;
  destination_ranges.reserve(load_range_count);
  for (size_t load_range_index = 0; load_range_index < load_range_count;
       ++load_range_index) {
    const auto& load_range = load_ranges[load_range_index];
    if (load_range.output_tensor_index >= output_tensor_byte_counts.size()) {
      throw std::out_of_range("Metal I/O output tensor index is invalid");
    }
    if (load_range.byte_count == 0) {
      throw std::invalid_argument(
          "Metal I/O load range byte count must be positive");
    }
    if (load_range.byte_count >
            std::numeric_limits<uint64_t>::max() -
                load_range.source_file_offset_bytes ||
        load_range.byte_count >
            std::numeric_limits<size_t>::max() -
                load_range.output_tensor_offset_bytes) {
      throw std::out_of_range("Metal I/O expert-pack range overflowed");
    }
    const auto source_range_end_offset_bytes =
        load_range.source_file_offset_bytes + load_range.byte_count;
    const auto destination_range_end_offset_bytes =
        load_range.output_tensor_offset_bytes + load_range.byte_count;
    if (source_range_end_offset_bytes > source_file_size_bytes ||
        destination_range_end_offset_bytes >
            output_tensor_byte_counts[load_range.output_tensor_index]) {
      throw std::out_of_range("Metal I/O expert-pack range is invalid");
    }
    destination_ranges.push_back({
        load_range.output_tensor_index,
        load_range.output_tensor_offset_bytes,
        destination_range_end_offset_bytes,
    });
  }
  std::sort(
      destination_ranges.begin(),
      destination_ranges.end(),
      [](const DestinationRange& first_range,
         const DestinationRange& second_range) {
        return std::tie(
                   first_range.output_tensor_index,
                   first_range.start_offset_bytes) <
            std::tie(
                   second_range.output_tensor_index,
                   second_range.start_offset_bytes);
      });
  size_t destination_range_index = 0;
  for (size_t output_tensor_index = 0;
       output_tensor_index < output_tensor_byte_counts.size();
       ++output_tensor_index) {
    size_t expected_start_offset_bytes = 0;
    while (destination_range_index < destination_ranges.size() &&
           destination_ranges[destination_range_index].output_tensor_index ==
               output_tensor_index) {
      const auto& destination_range =
          destination_ranges[destination_range_index];
      if (destination_range.start_offset_bytes != expected_start_offset_bytes) {
        throw std::invalid_argument(
            "Metal I/O load ranges must exactly cover every output tensor");
      }
      expected_start_offset_bytes = destination_range.end_offset_bytes;
      ++destination_range_index;
    }
    if (expected_start_offset_bytes !=
        output_tensor_byte_counts[output_tensor_index]) {
      throw std::invalid_argument(
          "Metal I/O load ranges must exactly cover every output tensor");
    }
  }
}

void allocate_output_arrays(
    const astronomical_metal_expert_loader_output_tensor* output_tensors,
    size_t output_tensor_count,
    const std::vector<size_t>& output_tensor_byte_counts,
    mlx_array* output_arrays) {
  if (output_arrays == nullptr) {
    throw std::invalid_argument("Metal I/O output arrays are required");
  }
  for (size_t output_tensor_index = 0;
       output_tensor_index < output_tensor_count;
       ++output_tensor_index) {
    const auto& output_tensor = output_tensors[output_tensor_index];
    const auto output_dtype = mlx_dtype_to_cpp(output_tensor.dtype);
    mlx::core::Shape output_shape(
        output_tensor.shape,
        output_tensor.shape + output_tensor.dimension_count);
    mlx::core::array output_array(
        mlx::core::allocator::malloc(
            output_tensor_byte_counts[output_tensor_index]),
        output_shape,
        output_dtype,
        mlx::core::allocator::free);
    output_arrays[output_tensor_index] = mlx_array_new_(std::move(output_array));
  }
}
