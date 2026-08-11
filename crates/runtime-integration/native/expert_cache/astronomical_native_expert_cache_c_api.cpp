#include "astronomical_native_expert_cache.h"

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

#include "mlx/c/error.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/stream.h"
#include "native_expert_cache_internal.h"

namespace {

// This file is the only exception and ownership translation boundary between
// bindgen-generated C calls and the C++ cache. Every fallible entry point resets
// outputs, catches standard exceptions, and reports through MLX's error channel.

void report_failure(const std::exception& failure) {
  // No C++ exception may cross bindgen-generated C or Rust frames. MLX's
  // thread-local error channel preserves the native description for Rust's
  // typed runtime-error classifier.
  const std::string description =
      std::string("native expert cache failed: ") + failure.what();
  mlx_error(description.c_str());
}

}  // namespace

extern "C" astronomical_native_expert_cache*
astronomical_native_expert_cache_new(
    const astronomical_native_expert_layer_descriptor* layer_descriptors,
    size_t layer_descriptor_count,
    uint64_t maximum_resident_payload_byte_count) {
  try {
    return new astronomical_native_expert_cache{
        std::make_unique<
            astronomical::native_expert_cache::NativeExpertCache>(
            layer_descriptors,
            layer_descriptor_count,
            maximum_resident_payload_byte_count)};
  } catch (const std::exception& failure) {
    report_failure(failure);
    return nullptr;
  }
}

extern "C" int astronomical_native_expert_cache_prepare_layer(
    astronomical_native_expert_cache* cache,
    size_t layer_index,
    mlx_array selected_expert_indices,
    mlx_stream stream,
    int collect_performance_metrics,
    astronomical_native_expert_snapshot** output_snapshot,
    astronomical_native_expert_cache_request_report* output_report) {
  try {
    if (cache == nullptr || !cache->cache || output_snapshot == nullptr ||
        output_report == nullptr) {
      throw std::invalid_argument(
          "native expert route preparation arguments are invalid");
    }
    // Initialize outputs before any fallible work so Rust never receives a
    // partially owned snapshot or stale report after an exception.
    *output_snapshot = nullptr;
    *output_report = astronomical_native_expert_cache_request_report{};
    auto prepared_route = cache->cache->prepare_layer(
        layer_index,
        mlx_array_get_(selected_expert_indices),
        mlx_stream_get_(stream),
        collect_performance_metrics != 0,
        *output_report);
    *output_snapshot =
        new astronomical_native_expert_snapshot{std::move(prepared_route)};
    return 0;
  } catch (const std::exception& failure) {
    report_failure(failure);
    return 1;
  }
}

extern "C" int astronomical_native_expert_cache_analyze_layer(
    astronomical_native_expert_cache* cache,
    size_t layer_index,
    mlx_array selected_expert_indices,
    mlx_stream stream,
    int collect_performance_metrics,
    astronomical_native_expert_route_analysis** output_analysis,
    astronomical_native_expert_cache_request_report* output_report) {
  try {
    if (cache == nullptr || !cache->cache || output_analysis == nullptr ||
        output_report == nullptr) {
      throw std::invalid_argument(
          "native expert route analysis arguments are invalid");
    }
    // Start with empty outputs. If any later step fails, Rust receives neither
    // a half-owned analysis handle nor stale numbers from an earlier call.
    *output_analysis = nullptr;
    *output_report = astronomical_native_expert_cache_request_report{};
    auto route_analysis = cache->cache->analyze_layer(
        layer_index,
        mlx_array_get_(selected_expert_indices),
        mlx_stream_get_(stream),
        collect_performance_metrics != 0,
        *output_report);
    auto route_analysis_owner = std::make_unique<
        astronomical_native_expert_route_analysis>();
    // The handle remembers its cache owner. Commit rejects an analysis created
    // by a different cache because expert IDs and page geometry would not match.
    route_analysis_owner->owner = cache;
    route_analysis_owner->route = std::make_unique<
        astronomical::native_expert_cache::NativeExpertRouteAnalysis>(
        std::move(route_analysis));
    route_analysis_owner->report = *output_report;
    *output_analysis = route_analysis_owner.release();
    return 0;
  } catch (const std::exception& failure) {
    report_failure(failure);
    return 1;
  }
}

extern "C" int astronomical_native_expert_cache_commit_layer(
    astronomical_native_expert_cache* cache,
    astronomical_native_expert_route_analysis* route_analysis,
    uint64_t maximum_resident_payload_byte_count,
    mlx_stream stream,
    int collect_performance_metrics,
    astronomical_native_expert_snapshot** output_snapshot,
    astronomical_native_expert_cache_request_report* output_report) {
  try {
    if (cache == nullptr || !cache->cache || route_analysis == nullptr ||
        route_analysis->owner != cache || !route_analysis->route ||
        output_snapshot == nullptr || output_report == nullptr) {
      throw std::invalid_argument(
          "native expert route commit arguments are invalid");
    }
    // Carry phase-1 counters forward. Native commit adds hits, misses, reads,
    // evictions, publications, and the final ceiling to the same report.
    *output_snapshot = nullptr;
    *output_report = route_analysis->report;
    auto prepared_route = cache->cache->commit_layer(
        std::move(*route_analysis->route),
        maximum_resident_payload_byte_count,
        mlx_stream_get_(stream),
        collect_performance_metrics != 0,
        *output_report);
    route_analysis->route.reset();
    *output_snapshot =
        new astronomical_native_expert_snapshot{std::move(prepared_route)};
    return 0;
  } catch (const std::exception& failure) {
    report_failure(failure);
    return 1;
  }
}

extern "C" void astronomical_native_expert_route_analysis_free(
    astronomical_native_expert_route_analysis* route_analysis) {
  delete route_analysis;
}

extern "C" int astronomical_native_expert_snapshot_gather_matmul(
    mlx_array* output,
    const astronomical_native_expert_snapshot* snapshot,
    int projection_index,
    mlx_array activations,
    mlx_array selected_expert_indices,
    int transpose_weights,
    int sorted_indices,
    mlx_stream stream) {
  try {
    if (output == nullptr || snapshot == nullptr ||
        !snapshot->route.page_table_snapshot ||
        projection_index < 0 || projection_index >= 3) {
      throw std::invalid_argument(
          "native expert gathered product arguments are invalid");
    }
    if (transpose_weights == 0) {
      throw std::invalid_argument(
          "native expert gathered product requires transposed weights");
    }
    auto product = snapshot->route.page_table_snapshot->storage_mode ==
            astronomical::paged_expert_execution::ExpertPageStorageMode::NativeBfloat16
        ? astronomical::paged_expert_execution::build_paged_native_bfloat16_product(
              snapshot->route.page_table_snapshot,
              snapshot->route.selected_expert_ids,
              projection_index,
              mlx_array_get_(activations),
              mlx_array_get_(selected_expert_indices),
              mlx_stream_get_(stream))
        : astronomical::paged_expert_execution::build_paged_quantized_product(
              snapshot->route.page_table_snapshot,
              snapshot->route.selected_expert_ids,
              projection_index,
              mlx_array_get_(activations),
              mlx_array_get_(selected_expert_indices),
              sorted_indices != 0,
              mlx_stream_get_(stream));
    mlx_array_set_(*output, std::move(product));
    return 0;
  } catch (const std::exception& failure) {
    report_failure(failure);
    return 1;
  }
}

extern "C" int
astronomical_native_expert_cache_update_maximum_resident_payload_bytes(
    astronomical_native_expert_cache* cache,
    uint64_t maximum_resident_payload_byte_count) {
  try {
    if (cache == nullptr || !cache->cache) {
      throw std::invalid_argument("native expert cache owner is invalid");
    }
    cache->cache->update_maximum_resident_payload_byte_count(
        maximum_resident_payload_byte_count);
    return 0;
  } catch (const std::exception& failure) {
    report_failure(failure);
    return 1;
  }
}

extern "C" int astronomical_native_expert_cache_freeze_retention_growth(
    astronomical_native_expert_cache* cache) {
  if (cache == nullptr || !cache->cache) {
    return 0;
  }
  return cache->cache->freeze_retention_growth() ? 1 : 0;
}

extern "C" int astronomical_native_expert_cache_reclaim_retained_payload_bytes(
    astronomical_native_expert_cache* cache,
    uint64_t reclamation_target_byte_count,
    int* output_did_reclaim) {
  try {
    if (cache == nullptr || !cache->cache || output_did_reclaim == nullptr) {
      throw std::invalid_argument(
          "native expert cache reclamation arguments are invalid");
    }
    *output_did_reclaim = cache->cache->reclaim_retained_payload_bytes(
                              reclamation_target_byte_count)
        ? 1
        : 0;
    return 0;
  } catch (const std::exception& failure) {
    report_failure(failure);
    return 1;
  }
}

extern "C" int astronomical_native_expert_cache_resume_retention_growth(
    astronomical_native_expert_cache* cache) {
  if (cache == nullptr || !cache->cache) {
    return 0;
  }
  return cache->cache->resume_retention_growth() ? 1 : 0;
}

extern "C" astronomical_native_expert_cache_statistics
astronomical_native_expert_cache_get_statistics(
    const astronomical_native_expert_cache* cache) {
  if (cache == nullptr || !cache->cache) {
    return astronomical_native_expert_cache_statistics{};
  }
  return cache->cache->statistics();
}

extern "C" void astronomical_native_expert_snapshot_free(
    astronomical_native_expert_snapshot* snapshot) {
  delete snapshot;
}

extern "C" void astronomical_native_expert_cache_free(
    astronomical_native_expert_cache* cache) {
  delete cache;
}
