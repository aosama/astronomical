#ifndef ASTRONOMICAL_METAL_EXPERT_LOADER_VALIDATION_H
#define ASTRONOMICAL_METAL_EXPERT_LOADER_VALIDATION_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "astronomical_metal_expert_loader.h"

std::vector<size_t> validate_output_tensors(
    const astronomical_metal_expert_loader_output_tensor* output_tensors,
    size_t output_tensor_count);

void validate_load_ranges(
    const astronomical_metal_expert_loader_load_range* load_ranges,
    size_t load_range_count,
    const std::vector<size_t>& output_tensor_byte_counts,
    uint64_t source_file_size_bytes);

void allocate_output_arrays(
    const astronomical_metal_expert_loader_output_tensor* output_tensors,
    size_t output_tensor_count,
    const std::vector<size_t>& output_tensor_byte_counts,
    mlx_array* output_arrays);

#endif
