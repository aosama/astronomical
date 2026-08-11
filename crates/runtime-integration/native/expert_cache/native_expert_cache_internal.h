#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "astronomical_native_expert_cache.h"
#include "mlx/array.h"
#include "mlx/fast.h"
#include "mlx/paged_buffer_store.h"
#include "paged_expert_execution/paged_expert_execution_internal.h"

namespace astronomical::native_expert_cache {

namespace mx = mlx::core;
namespace paged = astronomical::paged_expert_execution;

// One opened reader is shared by every tensor source in the same shard. The
// underlying product-neutral MLX reader owns positional input/output policy.
struct OpenedSourceFile {
  uint64_t file_size_bytes;
  std::shared_ptr<mx::io::ParallelFileReader> reader;

  OpenedSourceFile(
      uint64_t file_size_bytes,
      std::shared_ptr<mx::io::ParallelFileReader> reader)
      : file_size_bytes(file_size_bytes), reader(std::move(reader)) {}
  OpenedSourceFile(const OpenedSourceFile&) = delete;
  OpenedSourceFile& operator=(const OpenedSourceFile&) = delete;
};

// Immutable startup metadata for one complete safetensors tensor. A route reads
// one bytes_per_expert range into slot_byte_offset of the destination page.
struct TensorSource {
  int projection_index;
  int parameter_index;
  std::string source_file_path;
  std::shared_ptr<OpenedSourceFile> opened_source_file;
  uint64_t tensor_payload_offset;
  size_t bytes_per_expert;
  mx::Shape expert_shape;
  mx::Dtype dtype;
  size_t slot_byte_offset;
};

// One sparse layer's validated geometry and last published page-table generation.
struct LayerProfile {
  size_t expert_capacity;
  std::array<int, 3> projection_quantization_group_sizes;
  std::array<int, 3> projection_quantization_bits;
  paged::ExpertPageStorageMode storage_mode;
  size_t slot_byte_count;
  size_t payload_byte_count_per_expert;
  std::vector<TensorSource> tensor_sources;
  size_t resident_expert_count{0};
  uint64_t resident_payload_byte_count{0};
  uint64_t publication_generation{0};
  std::shared_ptr<const paged::PageTableSnapshot> current_snapshot;
};

// One independently allocated expert page. Typed arrays are views into storage;
// they must not outlive the shared PagedBufferSlot that owns their Metal buffer.
struct ExpertSlot {
  std::shared_ptr<mx::PagedBufferSlot> storage;
  std::optional<paged::ExpertPageArrays> quantized_page_arrays;
  std::optional<paged::NativeBfloat16ExpertPageArrays>
      native_bfloat16_page_arrays;
  size_t payload_byte_count;
};

// Retention policy owns one slot per (layer, expert) and one global recency clock.
struct CacheEntry {
  std::shared_ptr<ExpertSlot> slot;
  uint64_t last_access_sequence_number;
};

// A fully resident layer can defer the router bitmap's host synchronization.
// Policy must reconcile this evidence before any operation can evict a page.
struct PendingRouteEvidence {
  size_t layer_index;
  mx::array selected_expert_bitmap;
};

// Think of this as a shopping list, not a shopping cart. Analysis answers
// "which experts does this layer need?" and "which are missing?" without
// loading any weight bytes. Rust can therefore measure memory and choose a safe
// cache ceiling before commit spends memory or reads the solid-state drive.
struct NativeExpertRouteAnalysis {
  size_t layer_index;
  // Ordinary incomplete layers have exact, sorted, unique expert IDs here.
  // A complete layer leaves this empty so its lazy bitmap need not synchronize.
  std::optional<std::vector<size_t>> selected_expert_ids;
  std::optional<mx::array> selected_expert_bitmap;
  // If a complete layer's ceiling shrinks between analysis and commit, this
  // snapshot keeps the already-valid route alive while cache ownership changes.
  std::shared_ptr<const paged::PageTableSnapshot> complete_layer_snapshot;
};

struct PreparedExpertRoute {
  // This shared snapshot is captured by lazy MLX primitives. Later cache policy
  // may publish a new generation without invalidating this route's addresses.
  std::shared_ptr<const paged::PageTableSnapshot> page_table_snapshot;
  // Ordinary routes declare their exact host-evaluated experts. A missing list
  // is reserved for an entirely resident layer.
  std::optional<std::vector<size_t>> selected_expert_ids;
};

// Compares two independently owned payloads without adding them first. Unsigned
// wraparound must fail closed rather than make an impossible route appear small.
constexpr bool combined_payload_exceeds_byte_ceiling(
    uint64_t resident_payload_byte_count,
    uint64_t pending_payload_byte_count,
    uint64_t payload_byte_ceiling) {
  return pending_payload_byte_count > payload_byte_ceiling ||
      resident_payload_byte_count >
          payload_byte_ceiling - pending_payload_byte_count;
}

struct CacheKey {
  size_t layer_index;
  size_t expert_id;

  bool operator==(const CacheKey& other) const {
    return layer_index == other.layer_index && expert_id == other.expert_id;
  }
};

struct CacheKeyHash {
  size_t operator()(const CacheKey& key) const {
    return (key.layer_index * 1315423911u) ^ key.expert_id;
  }
};

std::shared_ptr<const paged::PageTableSnapshot> build_page_table_snapshot(
    size_t expert_capacity,
    uint64_t generation,
    paged::ExpertPageStorageMode storage_mode,
    const std::vector<std::pair<size_t, std::shared_ptr<ExpertSlot>>>& expert_slots);

// Mutable cache policy has one owner and is called serially by one model-serving
// request. Snapshot owners are immutable and may outlive cache entry eviction.
class NativeExpertCache {
 public:
  NativeExpertCache(
      const astronomical_native_expert_layer_descriptor* layer_descriptors,
      size_t layer_descriptor_count,
      uint64_t maximum_resident_payload_byte_count);

  PreparedExpertRoute prepare_layer(
      size_t layer_index,
      const mx::array& selected_expert_indices,
      mx::Stream stream,
      bool collect_performance_metrics,
      astronomical_native_expert_cache_request_report& report);

  NativeExpertRouteAnalysis analyze_layer(
      size_t layer_index,
      const mx::array& selected_expert_indices,
      mx::Stream stream,
      bool collect_performance_metrics,
      astronomical_native_expert_cache_request_report& report);

  PreparedExpertRoute commit_layer(
      NativeExpertRouteAnalysis route_analysis,
      uint64_t maximum_resident_payload_byte_count,
      mx::Stream stream,
      bool collect_performance_metrics,
      astronomical_native_expert_cache_request_report& report);

  astronomical_native_expert_cache_statistics statistics() const;
  void update_maximum_resident_payload_byte_count(
      uint64_t maximum_resident_payload_byte_count);
  bool freeze_retention_growth() noexcept;
  bool reclaim_retained_payload_bytes(uint64_t reclamation_target_byte_count);
  bool resume_retention_growth() noexcept;

 private:
  mx::array build_expert_selection_bitmap(
      const mx::array& selected_expert_indices,
      size_t expert_capacity,
      mx::Stream stream) const;
  std::vector<size_t> copy_sorted_unique_expert_ids_from_bitmap(
      mx::array selected_expert_bitmap,
      size_t expert_capacity,
      bool collect_performance_metrics,
      astronomical_native_expert_cache_request_report* report) const;
  bool layer_is_fully_resident(size_t layer_index) const;
  // Layer shares are fairness guides. They are not separate hard cache limits.
  // A busy layer may borrow bytes that other layers are not using.
  uint64_t proportional_retention_share_byte_count(
      size_t layer_index) const;
  uint64_t protected_retention_floor_byte_count_for_layer(
      size_t layer_index) const;
  uint64_t maximum_borrowable_retained_payload_byte_count_for_route(
      size_t layer_index) const;
  void enforce_maximum_resident_payload_byte_count(
      uint64_t maximum_resident_payload_byte_count);
  void reconcile_pending_route_evidence(
      bool collect_performance_metrics = false,
      astronomical_native_expert_cache_request_report* report = nullptr);
  std::shared_ptr<ExpertSlot> load_expert_slot(
      const LayerProfile& layer_profile,
      size_t expert_id,
      bool collect_performance_metrics,
      astronomical_native_expert_cache_request_report& report) const;
  void evict_to_fit(
      size_t protected_layer_index,
      const std::vector<size_t>& protected_expert_ids,
      uint64_t pending_payload_byte_count,
      std::vector<size_t>& changed_layer_indices,
      uint64_t* evicted_payload_byte_count_output = nullptr);
  uint64_t publish_changed_layers(
      const std::vector<size_t>& changed_layer_indices);
  void publish_layer(size_t layer_index);
  uint64_t payload_byte_count_for_expert_ids(
      size_t layer_index,
      const std::vector<size_t>& expert_ids) const;
  std::vector<size_t> missing_expert_ids_for_route(
      size_t layer_index,
      const std::vector<size_t>& selected_expert_ids) const;

  std::vector<LayerProfile> layer_profiles_;
  std::unordered_map<std::string, std::shared_ptr<OpenedSourceFile>>
      opened_source_files_;
  std::unordered_map<CacheKey, CacheEntry, CacheKeyHash> cache_entries_;
  uint64_t resident_payload_byte_count_{0};
  // configured_ remembers the newest long-lived policy. maximum_ is the active
  // request ceiling and may be temporarily lower while growth is frozen.
  uint64_t configured_maximum_resident_payload_byte_count_;
  uint64_t maximum_resident_payload_byte_count_;
  bool retention_growth_frozen_{false};
  uint64_t next_access_sequence_number_{0};
  mx::fast::CustomKernelFunction expert_selection_bitmap_kernel_;
  std::vector<PendingRouteEvidence> pending_route_evidence_;
  astronomical_native_expert_cache_statistics cumulative_statistics_{};
};

}  // namespace astronomical::native_expert_cache

struct astronomical_native_expert_cache_ {
  std::unique_ptr<astronomical::native_expert_cache::NativeExpertCache> cache;
};

struct astronomical_native_expert_route_analysis_ {
  astronomical_native_expert_cache* owner;
  std::unique_ptr<astronomical::native_expert_cache::NativeExpertRouteAnalysis>
      route;
  astronomical_native_expert_cache_request_report report{};
};

struct astronomical_native_expert_snapshot_ {
  astronomical::native_expert_cache::PreparedExpertRoute route;
};
