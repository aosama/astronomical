#ifndef ASTRONOMICAL_NATIVE_EXPERT_CACHE_H
#define ASTRONOMICAL_NATIVE_EXPERT_CACHE_H

#include <stddef.h>
#include <stdint.h>

#include "mlx/c/array.h"
#include "mlx/c/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque cache policy owner. Construction copies every nested descriptor, so
// callers may release paths and shapes after astronomical_native_expert_cache_new.
typedef struct astronomical_native_expert_cache_
    astronomical_native_expert_cache;
// Immutable page-table generation. It retains every MLX page array that a lazy
// gathered product can dereference, independently of later cache eviction.
typedef struct astronomical_native_expert_snapshot_
    astronomical_native_expert_snapshot;

typedef enum astronomical_native_expert_projection_ {
  ASTRONOMICAL_NATIVE_EXPERT_GATE = 0,
  ASTRONOMICAL_NATIVE_EXPERT_UP = 1,
  ASTRONOMICAL_NATIVE_EXPERT_DOWN = 2,
} astronomical_native_expert_projection;

typedef enum astronomical_native_expert_parameter_ {
  ASTRONOMICAL_NATIVE_EXPERT_PACKED_WEIGHT = 0,
  ASTRONOMICAL_NATIVE_EXPERT_SCALES = 1,
  ASTRONOMICAL_NATIVE_EXPERT_BIASES = 2,
} astronomical_native_expert_parameter;

typedef struct astronomical_native_expert_tensor_source_ {
  // Projection and parameter use the stable enum values above. Quantization
  // fields are zero only for native bfloat16 weight sources.
  int projection_index;
  int parameter_index;
  int quantization_group_size;
  int quantization_bits;
  const char* source_file_path;
  uint64_t tensor_payload_offset;
  size_t bytes_per_expert;
  const int* expert_shape;
  size_t expert_shape_dimension_count;
  mlx_dtype dtype;
} astronomical_native_expert_tensor_source;

typedef struct astronomical_native_expert_layer_descriptor_ {
  size_t expert_capacity;
  const astronomical_native_expert_tensor_source* tensor_sources;
  size_t tensor_source_count;
} astronomical_native_expert_layer_descriptor;

typedef struct astronomical_native_expert_cache_request_report_ {
  // Request-local route evidence. Wall-clock timing fields remain zero when
  // collection is disabled; counts and bytes still describe cache behavior.
  uint64_t cache_hit_count;
  uint64_t cache_miss_count;
  uint64_t disk_page_load_count;
  uint64_t disk_batch_load_count;
  uint64_t successful_source_read_count;
  uint64_t successful_source_read_byte_count;
  uint64_t successful_source_read_elapsed_nanoseconds;
  uint64_t route_dependency_synchronization_count;
  uint64_t route_dependency_synchronization_elapsed_nanoseconds;
  uint64_t maximum_route_dependency_synchronization_elapsed_nanoseconds;
  uint64_t payload_copy_byte_count;
  uint64_t page_table_publication_count;
  uint64_t complete_layer_route_synchronization_elision_count;
} astronomical_native_expert_cache_request_report;

typedef struct astronomical_native_expert_cache_statistics_ {
  // Current residency is mixed with cumulative process-lifetime policy totals.
  uint64_t resident_expert_count;
  uint64_t resident_payload_byte_count;
  uint64_t maximum_resident_payload_byte_count;
  uint64_t eviction_count;
  uint64_t cache_hit_count;
  uint64_t cache_miss_count;
  uint64_t disk_page_load_count;
  uint64_t disk_batch_load_count;
} astronomical_native_expert_cache_statistics;

// Creates one byte-bounded, layer-balanced least-recently-used cache across all
// layer-qualified experts. Returns null and reports through mlx_error on
// validation failure.
astronomical_native_expert_cache* astronomical_native_expert_cache_new(
    const astronomical_native_expert_layer_descriptor* layer_descriptors,
    size_t layer_descriptor_count,
    uint64_t maximum_resident_payload_byte_count);

// Evaluates router evidence, updates recency, reads missing ranges, and returns
// a newly owned immutable snapshot. The caller must free output_snapshot.
int astronomical_native_expert_cache_prepare_layer(
    astronomical_native_expert_cache* cache,
    size_t layer_index,
    mlx_array selected_expert_indices,
    mlx_stream stream,
    int collect_performance_metrics,
    astronomical_native_expert_snapshot** output_snapshot,
    astronomical_native_expert_cache_request_report* output_report);

// Replaces the configured and active retention ceiling, rebalancing layer
// shares and evicting least-recently-used entries when growth is not frozen.
int astronomical_native_expert_cache_update_maximum_resident_payload_bytes(
    astronomical_native_expert_cache* cache,
    uint64_t maximum_resident_payload_byte_count);

// Pins the active retention ceiling to current residency without evicting hot
// pages. Returns one only when the policy changed.
int astronomical_native_expert_cache_freeze_retention_growth(
    astronomical_native_expert_cache* cache);

// Evicts up to the requested retained payload and leaves growth frozen. Existing
// snapshots remain executable because they own their page generations.
int astronomical_native_expert_cache_reclaim_retained_payload_bytes(
    astronomical_native_expert_cache* cache,
    uint64_t reclamation_target_byte_count,
    int* output_did_reclaim);

// Restores the configured retention ceiling. It does not prewarm any expert.
int astronomical_native_expert_cache_resume_retention_growth(
    astronomical_native_expert_cache* cache);

// Builds a lazy Metal gathered product against the immutable snapshot.
int astronomical_native_expert_snapshot_gather_matmul(
    mlx_array* output,
    const astronomical_native_expert_snapshot* snapshot,
    int projection_index,
    mlx_array activations,
    mlx_array selected_expert_indices,
    int transpose_weights,
    int sorted_indices,
    mlx_stream stream);

astronomical_native_expert_cache_statistics
astronomical_native_expert_cache_get_statistics(
    const astronomical_native_expert_cache* cache);

void astronomical_native_expert_snapshot_free(
    astronomical_native_expert_snapshot* snapshot);

void astronomical_native_expert_cache_free(
    astronomical_native_expert_cache* cache);

#ifdef __cplusplus
}
#endif

#endif
