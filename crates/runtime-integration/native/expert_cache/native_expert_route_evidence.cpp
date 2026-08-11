#include "native_expert_cache_internal.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <stdexcept>

namespace astronomical::native_expert_cache {

// Router choices stay as lazy MLX arrays until ordinary demand-only paging must
// know which file ranges to read. Fully resident layers defer even that host
// boundary until exact recency can affect policy.

mx::array NativeExpertCache::build_expert_selection_bitmap(
    const mx::array& selected_expert_indices,
    size_t expert_capacity,
    mx::Stream stream) const {
  if (selected_expert_indices.size() == 0 || expert_capacity == 0 ||
      (selected_expert_indices.dtype() != mx::uint32 &&
       selected_expert_indices.dtype() != mx::int32)) {
    throw std::invalid_argument(
        "selected expert indices must be non-empty UInt32 or Int32 values");
  }
  if (selected_expert_indices.size() > static_cast<size_t>(INT_MAX) ||
      expert_capacity > static_cast<size_t>(INT_MAX)) {
    throw std::overflow_error(
        "selected expert route exceeds the Metal launch range");
  }
  // One word per 32 experts is bounded by model capacity rather than assignment
  // count. The trailing validation word lets the kernel reject an invalid route
  // without reading beyond the page table.
  const int bitmap_word_count =
      static_cast<int>((expert_capacity + 31) / 32);
  auto bitmap_outputs = expert_selection_bitmap_kernel_(
      {selected_expert_indices},
      {{bitmap_word_count + 1}},
      {mx::uint32},
      {static_cast<int>(selected_expert_indices.size()), 1, 1},
      {std::min(static_cast<int>(selected_expert_indices.size()), 256), 1, 1},
      {
          {"expert_capacity", static_cast<int>(expert_capacity)},
          {"bitmap_word_count", bitmap_word_count},
      },
      0.0f,
      false,
      stream);
  if (bitmap_outputs.size() != 1) {
    throw std::runtime_error(
        "expert-selection bitmap kernel returned no output");
  }
  return std::move(bitmap_outputs.front());
}

std::vector<size_t> NativeExpertCache::copy_sorted_unique_expert_ids_from_bitmap(
    mx::array selected_expert_bitmap,
    size_t expert_capacity,
    bool collect_performance_metrics,
    astronomical_native_expert_cache_request_report* report) const {
  const auto synchronization_started_at = collect_performance_metrics
      ? std::chrono::steady_clock::now()
      : std::chrono::steady_clock::time_point{};
  // This evaluation is the unavoidable host boundary for ordinary demand-only
  // paging: source reads cannot begin until router choices are concrete. The
  // wait can also execute preceding lazy model work, so it is attributed
  // separately from solid-state-drive reads.
  selected_expert_bitmap.eval();
  if (report != nullptr) {
    report->route_dependency_synchronization_count += 1;
    if (collect_performance_metrics) {
      const auto synchronization_elapsed_nanoseconds =
          static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() -
                  synchronization_started_at)
                  .count());
      report->route_dependency_synchronization_elapsed_nanoseconds +=
          synchronization_elapsed_nanoseconds;
      report->maximum_route_dependency_synchronization_elapsed_nanoseconds =
          std::max(
              report
                  ->maximum_route_dependency_synchronization_elapsed_nanoseconds,
              synchronization_elapsed_nanoseconds);
    }
  }
  const size_t bitmap_word_count = (expert_capacity + 31) / 32;
  if (selected_expert_bitmap.size() != bitmap_word_count + 1 ||
      selected_expert_bitmap.data<uint32_t>()[bitmap_word_count] != 0) {
    throw std::invalid_argument(
        "selected expert route contains an out-of-range expert ID");
  }
  // Ascending bitmap traversal yields the sorted unique route expected by
  // binary-search protection in layer and global eviction.
  std::vector<size_t> selected_expert_ids;
  for (size_t bitmap_word_index = 0;
       bitmap_word_index < bitmap_word_count;
       ++bitmap_word_index) {
    const uint32_t bitmap_word =
        selected_expert_bitmap.data<uint32_t>()[bitmap_word_index];
    for (size_t bitmap_bit_index = 0; bitmap_bit_index < 32;
         ++bitmap_bit_index) {
      const size_t expert_id = bitmap_word_index * 32 + bitmap_bit_index;
      if (expert_id >= expert_capacity) {
        break;
      }
      if ((bitmap_word & (uint32_t{1} << bitmap_bit_index)) != 0) {
        selected_expert_ids.push_back(expert_id);
      }
    }
  }
  if (selected_expert_ids.empty()) {
    throw std::invalid_argument(
        "selected expert route contains no valid expert IDs");
  }
  return selected_expert_ids;
}

bool NativeExpertCache::layer_is_fully_resident(size_t layer_index) const {
  return layer_profiles_[layer_index].resident_expert_count ==
      layer_profiles_[layer_index].expert_capacity;
}

void NativeExpertCache::reconcile_pending_route_evidence(
    bool collect_performance_metrics,
    astronomical_native_expert_cache_request_report* report) {
  // Reconciliation is deferred only while no cache-policy decision needs exact
  // recency. Once policy can evict, process pending routes in submission order
  // to preserve the same global access sequence as eager synchronization.
  for (auto& pending_route : pending_route_evidence_) {
    const auto selected_expert_ids =
        copy_sorted_unique_expert_ids_from_bitmap(
            std::move(pending_route.selected_expert_bitmap),
            layer_profiles_[pending_route.layer_index].expert_capacity,
            collect_performance_metrics,
            report);
    for (const auto expert_id : selected_expert_ids) {
      const auto cache_entry = cache_entries_.find(
          CacheKey{pending_route.layer_index, expert_id});
      if (cache_entry == cache_entries_.end()) {
        throw std::runtime_error(
            "pending complete-layer route references an evicted expert");
      }
      next_access_sequence_number_ += 1;
      cache_entry->second.last_access_sequence_number =
          next_access_sequence_number_;
      cumulative_statistics_.cache_hit_count += 1;
    }
  }
  pending_route_evidence_.clear();
}

}  // namespace astronomical::native_expert_cache
