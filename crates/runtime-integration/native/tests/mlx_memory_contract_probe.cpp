#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <mach/vm_page_size.h>

#include "mlx/device.h"
#include "mlx/backend/metal/allocator.h"
#include "mlx/memory.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

namespace {

using namespace mlx::core;

constexpr std::size_t kMebibyte = 1024 * 1024;
constexpr std::size_t kGraphDimension = 4096;
constexpr std::size_t kCacheLimitBytes = 256 * kMebibyte;
constexpr std::size_t kLowCacheLimitBytes = 1 * kMebibyte;
constexpr std::size_t kActiveMemoryEnforcementLimitBytes = 50 * kMebibyte;
constexpr std::size_t kActiveMemoryEnforcementAllowanceBytes =
    kActiveMemoryEnforcementLimitBytes / 100;
constexpr std::size_t kAllowedActiveMemoryBytes =
    kActiveMemoryEnforcementLimitBytes + kActiveMemoryEnforcementAllowanceBytes;
constexpr std::size_t kOversizedAllocationBytes =
    kGraphDimension * kGraphDimension * sizeof(float);
constexpr const char* kActiveMemoryLimitErrorMarker =
    "ASTRONOMICAL_MLX_ACTIVE_MEMORY_LIMIT_EXCEEDED";

struct MemoryCounters {
  std::size_t active_bytes;
  std::size_t cache_bytes;
  std::size_t peak_bytes;
};

MemoryCounters read_memory_counters() {
  return {
      get_active_memory(),
      get_cache_memory(),
      get_peak_memory(),
  };
}

void require_condition(bool condition, const std::string& description) {
  if (!condition) {
    throw std::runtime_error(description);
  }
}

void print_counters(const std::string& phase_name, const MemoryCounters& counters) {
  std::cout << "[mlx-memory-contract] phase=" << phase_name
            << " active_bytes=" << counters.active_bytes
            << " cache_bytes=" << counters.cache_bytes
            << " peak_bytes=" << counters.peak_bytes << std::endl;
}

array evaluated_graph(const Stream& gpu_stream) {
  auto left = zeros(
      Shape{static_cast<ShapeElem>(kGraphDimension), static_cast<ShapeElem>(kGraphDimension)},
      float32,
      gpu_stream);
  auto right = zeros(
      Shape{static_cast<ShapeElem>(kGraphDimension), static_cast<ShapeElem>(kGraphDimension)},
      float32,
      gpu_stream);
  auto sum = add(left, right, gpu_stream);
  sum.eval();
  synchronize(gpu_stream);
  return sum;
}

void require_lazy_evaluation_boundary(const Stream& gpu_stream) {
  clear_cache();
  reset_peak_memory();
  const auto before_graph = read_memory_counters();
  auto left = zeros(
      Shape{static_cast<ShapeElem>(kGraphDimension), static_cast<ShapeElem>(kGraphDimension)},
      float32,
      gpu_stream);
  auto right = zeros(
      Shape{static_cast<ShapeElem>(kGraphDimension), static_cast<ShapeElem>(kGraphDimension)},
      float32,
      gpu_stream);
  auto lazy_sum = add(left, right, gpu_stream);
  const auto after_graph = read_memory_counters();
  print_counters("lazy_graph_construction", after_graph);
  const auto graph_construction_active_growth =
      after_graph.active_bytes > before_graph.active_bytes
      ? after_graph.active_bytes - before_graph.active_bytes
      : 0;
  require_condition(
      graph_construction_active_growth < kGraphDimension * kGraphDimension * sizeof(float),
      "lazy graph construction allocated the final evaluated payload before evaluation");
  require_condition(
      after_graph.cache_bytes == before_graph.cache_bytes,
      "lazy graph construction changed allocator-cache bytes before evaluation");

  lazy_sum.eval();
  synchronize(gpu_stream);
  const auto after_evaluation = read_memory_counters();
  require_condition(
      after_evaluation.active_bytes > before_graph.active_bytes,
      "evaluation did not create active MLX allocation bytes");
  require_condition(
      after_evaluation.peak_bytes >= after_evaluation.active_bytes,
      "MLX peak bytes did not include the live evaluated allocation");
  print_counters("lazy_evaluation_boundary", after_evaluation);
}

void require_peak_reset_and_cache_release(const Stream& gpu_stream) {
  clear_cache();
  reset_peak_memory();
  const auto baseline_counters = read_memory_counters();
  MemoryCounters live_counters;
  {
    auto live_graph = evaluated_graph(gpu_stream);
    live_counters = read_memory_counters();
    require_condition(live_counters.active_bytes > 0, "evaluated graph has no active bytes");
    reset_peak_memory();
    const auto after_peak_reset = read_memory_counters();
    require_condition(after_peak_reset.peak_bytes == 0, "peak reset did not return zero");
    require_condition(
        after_peak_reset.active_bytes == live_counters.active_bytes,
        "peak reset changed active bytes for a live array");
  }
  const auto after_owner_drop = read_memory_counters();
  require_condition(
      after_owner_drop.active_bytes == baseline_counters.active_bytes,
      "dropping evaluated array owners did not restore baseline active bytes");
  require_condition(
      after_owner_drop.cache_bytes > 0,
      "dropping evaluated array owners did not retain reclaimable cache bytes");
  clear_cache();
  const auto after_cache_clear = read_memory_counters();
  require_condition(
      after_cache_clear.active_bytes == baseline_counters.active_bytes,
      "clear_cache changed baseline active bytes after owner drop");
  require_condition(after_cache_clear.cache_bytes == 0, "clear_cache retained allocator bytes");
  print_counters("peak_reset_and_cache_release", after_cache_clear);
}

void require_cache_reuse(const Stream& gpu_stream) {
  clear_cache();
  {
    auto first_graph = evaluated_graph(gpu_stream);
    (void)first_graph;
  }
  const auto before_reuse = read_memory_counters();
  require_condition(before_reuse.cache_bytes > 0, "cache reuse phase has no cached allocation");
  {
    auto replacement_graph = evaluated_graph(gpu_stream);
    const auto during_reuse = read_memory_counters();
    require_condition(
        during_reuse.cache_bytes < before_reuse.cache_bytes,
        "same-shaped evaluated allocation did not consume cached bytes");
  }
  const auto after_reuse = read_memory_counters();
  require_condition(
      after_reuse.cache_bytes > 0,
      "dropping the reused allocation did not leave reclaimable bytes");
  clear_cache();
  print_counters("cache_reuse", after_reuse);
}

void require_zero_cache_limit(const Stream& gpu_stream) {
  clear_cache();
  const auto previous_cache_limit = set_cache_limit(0);
  {
    auto graph = evaluated_graph(gpu_stream);
    (void)graph;
  }
  const auto after_owner_drop = read_memory_counters();
  require_condition(
      after_owner_drop.cache_bytes == 0,
      "zero cache limit retained bytes after evaluated array owners were dropped");
  set_cache_limit(previous_cache_limit);
  print_counters("zero_cache_limit", after_owner_drop);
}

void require_allocation_triggered_cache_reclamation(const Stream& gpu_stream) {
  clear_cache();
  const auto previous_cache_limit = set_cache_limit(kCacheLimitBytes);
  {
    auto graph = evaluated_graph(gpu_stream);
    (void)graph;
  }
  const auto before_lowering = read_memory_counters();
  require_condition(
      before_lowering.cache_bytes > kLowCacheLimitBytes,
      "cache is too small for the allocation-triggered limit test");
  set_cache_limit(kLowCacheLimitBytes);
  const auto after_lowering = read_memory_counters();
  require_condition(
      after_lowering.cache_bytes == before_lowering.cache_bytes,
      "lowering cache limit reclaimed bytes without a later allocation");
  {
    auto incompatible_graph = zeros(
        Shape{static_cast<ShapeElem>(kGraphDimension * 2), static_cast<ShapeElem>(kGraphDimension)},
        float32,
        gpu_stream);
    incompatible_graph.eval();
    synchronize(gpu_stream);
  }
  const auto after_allocation = read_memory_counters();
  require_condition(
      after_allocation.cache_bytes < after_lowering.cache_bytes,
      "a later incompatible allocation did not reclaim cache excess");
  set_cache_limit(previous_cache_limit);
  clear_cache();
  print_counters("allocation_triggered_cache_reclamation", after_allocation);
}

void require_active_memory_limit_rejection(
    const std::string& allocation_path_name,
    std::size_t expected_attempted_allocation_bytes,
    const std::function<void()>& attempt_oversized_allocation) {
  const auto before_rejection = read_memory_counters();
  bool allocation_was_rejected = false;
  try {
    attempt_oversized_allocation();
  } catch (const std::runtime_error& allocation_error) {
    allocation_was_rejected = true;
    const std::string rejection_message = allocation_error.what();
    const std::string expected_rejection_message =
        std::string(kActiveMemoryLimitErrorMarker) +
        " active_bytes=" + std::to_string(before_rejection.active_bytes) +
        " allocation_bytes=" + std::to_string(expected_attempted_allocation_bytes) +
        " allowed_bytes=" + std::to_string(kAllowedActiveMemoryBytes);
    require_condition(
        rejection_message == expected_rejection_message,
        allocation_path_name + " rejection message mismatch: expected '" +
            expected_rejection_message + "', got '" + rejection_message + "'");
  }
  require_condition(
      allocation_was_rejected,
      allocation_path_name + " did not reject an allocation above the active-memory limit");
  const auto after_rejection = read_memory_counters();
  require_condition(
      after_rejection.active_bytes == before_rejection.active_bytes,
      allocation_path_name + " changed active MLX memory after rejection");
  require_condition(
      after_rejection.peak_bytes == before_rejection.peak_bytes,
      allocation_path_name + " changed peak MLX memory after rejection");
  print_counters("active_memory_limit_" + allocation_path_name, after_rejection);
}

void require_evaluation_rejection_preserves_prior_tape_work(const Stream& gpu_stream) {
  clear_cache();
  const auto previous_allocator_cache_limit_bytes = set_cache_limit(0);
  const auto previous_active_memory_limit_bytes =
      set_memory_limit(kActiveMemoryEnforcementLimitBytes);
  try {
    auto valid_pending_values = full(Shape{1024}, 3.0F, float32, gpu_stream);
    auto valid_pending_result = multiply(
        valid_pending_values,
        full(Shape{1024}, 2.0F, float32, gpu_stream),
        gpu_stream);
    auto valid_pending_sum = sum(valid_pending_result, gpu_stream);
    auto expanded_valid_sum = broadcast_to(
        valid_pending_sum,
        Shape{static_cast<ShapeElem>(kGraphDimension),
              static_cast<ShapeElem>(kGraphDimension)},
        gpu_stream);
    auto oversized_dependent_result = contiguous(expanded_valid_sum, false, gpu_stream);

    bool dependent_evaluation_was_rejected = false;
    try {
      eval(oversized_dependent_result);
    } catch (const std::runtime_error& evaluation_error) {
      dependent_evaluation_was_rejected = true;
      require_condition(
          std::string(evaluation_error.what()).starts_with(kActiveMemoryLimitErrorMarker),
          "evaluation replaced the original active-memory rejection: " +
              std::string(evaluation_error.what()));
    }
    require_condition(
        dependent_evaluation_was_rejected,
        "dependent oversized evaluation did not reject above the active-memory limit");

    const float preserved_valid_sum = valid_pending_sum.item<float>();
    synchronize(gpu_stream);
    require_condition(
        preserved_valid_sum == 6144.0F,
        "active-memory rejection corrupted valid work evaluated earlier in the same tape");

    const float fresh_fitting_sum =
        sum(add(
                full(Shape{512}, 2.0F, float32, gpu_stream),
                full(Shape{512}, 1.0F, float32, gpu_stream),
                gpu_stream),
            gpu_stream)
            .item<float>();
    require_condition(
        fresh_fitting_sum == 1536.0F,
        "a fresh fitting graph produced incorrect values after active-memory rejection");
    print_counters(
        "evaluation_rejection_preserves_prior_tape_work", read_memory_counters());
  } catch (...) {
    clear_cache();
    set_memory_limit(previous_active_memory_limit_bytes);
    set_cache_limit(previous_allocator_cache_limit_bytes);
    throw;
  }
  clear_cache();
  set_memory_limit(previous_active_memory_limit_bytes);
  set_cache_limit(previous_allocator_cache_limit_bytes);
}

void require_strict_active_memory_limit(const Stream& gpu_stream) {
  clear_cache();
  const auto previous_allocator_cache_limit_bytes = set_cache_limit(kCacheLimitBytes);
  const auto previous_active_memory_limit_bytes =
      set_memory_limit(kActiveMemoryEnforcementLimitBytes);
  try {
    const auto before_exact_boundary_allocation = read_memory_counters();
    auto exact_boundary_buffer = metal::allocator().malloc(kAllowedActiveMemoryBytes);
    const auto at_exact_boundary = read_memory_counters();
    require_condition(
        at_exact_boundary.active_bytes ==
            before_exact_boundary_allocation.active_bytes + kAllowedActiveMemoryBytes,
        "an allocation ending exactly at the allowed active-memory boundary was rejected");
    metal::allocator().free(exact_boundary_buffer);
    clear_cache();

    const std::size_t one_byte_beyond_boundary = kAllowedActiveMemoryBytes + 1;
    const std::size_t aligned_one_byte_beyond_boundary =
        vm_page_size *
        ((one_byte_beyond_boundary + vm_page_size - 1) / vm_page_size);
    require_active_memory_limit_rejection(
        "one_byte_beyond_boundary", aligned_one_byte_beyond_boundary, [&] {
          (void)metal::allocator().malloc(one_byte_beyond_boundary);
        });

    require_active_memory_limit_rejection("new_buffer", kOversizedAllocationBytes, [] {
      (void)metal::allocator().malloc(kOversizedAllocationBytes);
    });

    set_memory_limit(kCacheLimitBytes);
    {
      auto cached_graph = evaluated_graph(gpu_stream);
      (void)cached_graph;
    }
    const auto cached_allocation_counters = read_memory_counters();
    require_condition(
        cached_allocation_counters.cache_bytes > 0,
        "cached-buffer enforcement setup did not retain a reclaimable allocation");
    set_memory_limit(kActiveMemoryEnforcementLimitBytes);
    require_active_memory_limit_rejection(
        "cached_buffer_reuse", kOversizedAllocationBytes, [] {
      (void)metal::allocator().malloc(kOversizedAllocationBytes);
    });

    require_active_memory_limit_rejection(
        "host_backed_buffer", kOversizedAllocationBytes, [] {
      std::vector<float> host_values(kGraphDimension * kGraphDimension, 0.0F);
      array host_backed_array(
          host_values.data(),
          Shape{static_cast<ShapeElem>(kGraphDimension),
                static_cast<ShapeElem>(kGraphDimension)},
          float32,
          [](void*) {});
      (void)host_backed_array;
    });

    set_memory_limit(kCacheLimitBytes);
    {
      auto live_graph = evaluated_graph(gpu_stream);
      set_memory_limit(kActiveMemoryEnforcementLimitBytes);
      require_active_memory_limit_rejection("lowered_below_active", 1, [] {
        (void)metal::allocator().malloc(1);
      });
      set_memory_limit(kCacheLimitBytes);
    }
    clear_cache();
    set_memory_limit(kActiveMemoryEnforcementLimitBytes);
    {
      auto fitting_buffer = metal::allocator().malloc(1);
      metal::allocator().free(fitting_buffer);
    }
  } catch (...) {
    clear_cache();
    set_memory_limit(previous_active_memory_limit_bytes);
    set_cache_limit(previous_allocator_cache_limit_bytes);
    throw;
  }
  clear_cache();
  set_memory_limit(previous_active_memory_limit_bytes);
  set_cache_limit(previous_allocator_cache_limit_bytes);
}

void require_async_evaluation_cleanup_boundary(const Stream& gpu_stream) {
  clear_cache();
  const auto baseline_counters = read_memory_counters();
  {
    auto async_graph = add(
        zeros(
            Shape{static_cast<ShapeElem>(kGraphDimension), static_cast<ShapeElem>(kGraphDimension)},
            float32,
            gpu_stream),
        zeros(
            Shape{static_cast<ShapeElem>(kGraphDimension), static_cast<ShapeElem>(kGraphDimension)},
            float32,
            gpu_stream),
        gpu_stream);
    async_eval(async_graph);
    synchronize(gpu_stream);
    clear_cache();
    const auto after_synchronized_cleanup = read_memory_counters();
    require_condition(
        after_synchronized_cleanup.cache_bytes == 0,
        "synchronized async cleanup retained reclaimable cache bytes");
    require_condition(
        after_synchronized_cleanup.active_bytes > baseline_counters.active_bytes,
        "synchronized async cleanup released the live asynchronous result");
    print_counters("async_evaluation_cleanup_boundary", after_synchronized_cleanup);
  }
  clear_cache();
  const auto after_owner_drop = read_memory_counters();
  require_condition(
      after_owner_drop.active_bytes == baseline_counters.active_bytes,
      "dropping the asynchronous result did not restore baseline active bytes");
  require_condition(
      after_owner_drop.cache_bytes == 0,
      "final asynchronous owner cleanup retained allocator-cache bytes");
  print_counters("async_evaluation_owners_dropped", after_owner_drop);
}

void require_policy_round_trips() {
  const auto current_memory_limit = get_memory_limit();
  require_condition(current_memory_limit > 1, "MLX memory limit must allow a round-trip probe");
  const auto temporary_memory_limit = current_memory_limit / 2;
  const auto previous_memory_limit = set_memory_limit(temporary_memory_limit);
  require_condition(
      previous_memory_limit == current_memory_limit,
      "set_memory_limit did not return the prior memory limit");
  require_condition(
      get_memory_limit() == temporary_memory_limit,
      "get_memory_limit did not expose the temporary memory limit");
  const auto memory_limit_before_restore = set_memory_limit(previous_memory_limit);
  require_condition(
      memory_limit_before_restore == temporary_memory_limit,
      "set_memory_limit did not report the temporary memory limit during restore");
  require_condition(
      get_memory_limit() == current_memory_limit,
      "get_memory_limit did not expose the restored memory limit");

  print_counters("policy_round_trips", read_memory_counters());
}

void run_probe() {
  require_condition(is_available(Device{Device::gpu}), "MLX GPU device is unavailable");
  const auto gpu_stream = default_stream(Device{Device::gpu});
  clear_cache();
  reset_peak_memory();
  print_counters("baseline", read_memory_counters());
  require_lazy_evaluation_boundary(gpu_stream);
  require_peak_reset_and_cache_release(gpu_stream);
  require_cache_reuse(gpu_stream);
  require_zero_cache_limit(gpu_stream);
  require_allocation_triggered_cache_reclamation(gpu_stream);
  require_async_evaluation_cleanup_boundary(gpu_stream);
  require_evaluation_rejection_preserves_prior_tape_work(gpu_stream);
  require_strict_active_memory_limit(gpu_stream);
  synchronize(gpu_stream);
  clear_cache();
  require_policy_round_trips();
  synchronize(gpu_stream);
  clear_cache();
  print_counters("success", read_memory_counters());
}

}  // namespace

int main() {
  try {
    run_probe();
    return 0;
  } catch (const std::exception& probe_error) {
    std::cerr << "[mlx-memory-contract] status=failure error=" << probe_error.what() << std::endl;
    return 1;
  }
}
