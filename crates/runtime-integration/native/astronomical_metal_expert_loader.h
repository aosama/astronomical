#ifndef ASTRONOMICAL_METAL_EXPERT_LOADER_H
#define ASTRONOMICAL_METAL_EXPERT_LOADER_H

#include <stddef.h>
#include <stdint.h>

#include "mlx/c/array.h"
#include "mlx/c/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct astronomical_metal_expert_loader_output_tensor_ {
  const int* shape;
  int dimension_count;
  mlx_dtype dtype;
} astronomical_metal_expert_loader_output_tensor;

typedef struct astronomical_metal_expert_loader_load_range_ {
  size_t output_tensor_index;
  size_t output_tensor_offset_bytes;
  uint64_t source_file_offset_bytes;
  size_t byte_count;
} astronomical_metal_expert_loader_load_range;

typedef struct astronomical_metal_expert_loader_metrics_ {
  uint64_t requested_byte_count;
  size_t command_count;
  uint64_t host_encoding_elapsed_nanoseconds;
  uint64_t queue_elapsed_nanoseconds;
  int final_status;
} astronomical_metal_expert_loader_metrics;

typedef struct astronomical_metal_expert_loader_handle_
    astronomical_metal_expert_loader_handle;

int astronomical_metal_expert_loader_start(
    const char* source_file_path,
    const astronomical_metal_expert_loader_output_tensor* output_tensors,
    size_t output_tensor_count,
    const astronomical_metal_expert_loader_load_range* load_ranges,
    size_t load_range_count,
    mlx_stream target_gpu_stream,
    mlx_array* output_arrays,
    astronomical_metal_expert_loader_handle** output_handle);

int astronomical_metal_expert_loader_wait(
    astronomical_metal_expert_loader_handle* load_handle,
    astronomical_metal_expert_loader_metrics* output_metrics);

void astronomical_metal_expert_loader_free(
    astronomical_metal_expert_loader_handle* load_handle);

#ifdef __cplusplus
}
#endif

#endif
