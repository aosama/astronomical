#include "native_expert_cache_internal.h"

#include <array>
#include <chrono>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_set>

namespace astronomical::native_expert_cache {

// This translation unit owns one ordinary route transaction: load exact missing
// ranges, protect the selected route during global eviction, publish changed
// generations, and return an immutable snapshot for lazy execution.

namespace {

constexpr size_t kProjectionCount = 3;
constexpr size_t kParameterCount = 3;

}  // namespace

std::shared_ptr<ExpertSlot> NativeExpertCache::load_expert_slot(
    const LayerProfile& layer_profile,
    size_t expert_id,
    bool collect_performance_metrics,
    astronomical_native_expert_cache_request_report& report) const {
  // Allocate final MLX-owned storage first, then submit every source interval as
  // one range batch. There is no intermediate host payload or concatenation.
  auto slot_storage =
      std::make_shared<mx::PagedBufferSlot>(layer_profile.slot_byte_count);
  std::unordered_set<std::string> source_files_read;
  std::vector<mx::PagedBufferReadRange> source_read_ranges;
  source_read_ranges.reserve(layer_profile.tensor_sources.size());
  uint64_t payload_byte_count = 0;
  for (const auto& tensor_source : layer_profile.tensor_sources) {
    const uint64_t source_offset =
        tensor_source.tensor_payload_offset +
        expert_id * tensor_source.bytes_per_expert;
    source_read_ranges.push_back(mx::PagedBufferReadRange{
        tensor_source.opened_source_file->reader,
        source_offset,
        slot_storage,
        tensor_source.slot_byte_offset,
        tensor_source.bytes_per_expert});
    source_files_read.insert(tensor_source.source_file_path);
    report.successful_source_read_count += 1;
    report.successful_source_read_byte_count += tensor_source.bytes_per_expert;
    payload_byte_count += tensor_source.bytes_per_expert;
  }
  const auto source_read_started_at = std::chrono::steady_clock::now();
  mx::read_paged_buffer_ranges(source_read_ranges);
  if (collect_performance_metrics) {
    report.successful_source_read_elapsed_nanoseconds +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - source_read_started_at)
            .count();
  }
  // Commit only after every exact range completed. A short read throws before
  // this boundary, so a partial candidate can never enter cache policy.
  slot_storage->commit();
  report.disk_page_load_count += 1;
  report.disk_batch_load_count += source_files_read.size();

  std::array<std::array<std::optional<mx::array>, kParameterCount>,
             kProjectionCount>
      tensor_views;
  // Views retain the shared slot buffer and add no payload copy. The ExpertSlot
  // owner keeps both the storage and semantic projection grouping together.
  for (const auto& tensor_source : layer_profile.tensor_sources) {
    tensor_views[tensor_source.projection_index][tensor_source.parameter_index] =
        slot_storage->view(
            tensor_source.expert_shape,
            tensor_source.dtype,
            tensor_source.slot_byte_offset,
            tensor_source.bytes_per_expert);
  }
  if (layer_profile.storage_mode ==
      paged::ExpertPageStorageMode::NativeBfloat16) {
    std::array<mx::array, kProjectionCount> projection_weights = {
        *tensor_views[0][0], *tensor_views[1][0], *tensor_views[2][0]};
    return std::make_shared<ExpertSlot>(ExpertSlot{
        std::move(slot_storage),
        std::nullopt,
        paged::NativeBfloat16ExpertPageArrays{
            std::move(projection_weights)},
        static_cast<size_t>(payload_byte_count)});
  }
  std::array<paged::QuantizedProjectionArrays, kProjectionCount> projections = {
      paged::QuantizedProjectionArrays{
          *tensor_views[0][0],
           *tensor_views[0][1],
           *tensor_views[0][2],
           layer_profile.projection_quantization_group_sizes[0],
           layer_profile.projection_quantization_bits[0]},
      paged::QuantizedProjectionArrays{
          *tensor_views[1][0],
          *tensor_views[1][1],
          *tensor_views[1][2],
           layer_profile.projection_quantization_group_sizes[1],
           layer_profile.projection_quantization_bits[1]},
      paged::QuantizedProjectionArrays{
          *tensor_views[2][0],
          *tensor_views[2][1],
          *tensor_views[2][2],
           layer_profile.projection_quantization_group_sizes[2],
           layer_profile.projection_quantization_bits[2]}};
  return std::make_shared<ExpertSlot>(ExpertSlot{
      std::move(slot_storage),
      paged::ExpertPageArrays{std::move(projections)},
      std::nullopt,
      static_cast<size_t>(payload_byte_count)});
}

PreparedExpertRoute NativeExpertCache::prepare_layer(
    size_t layer_index,
    const mx::array& selected_expert_indices,
    mx::Stream stream,
    bool collect_performance_metrics,
    astronomical_native_expert_cache_request_report& report) {
  if (layer_index >= layer_profiles_.size()) {
    throw std::out_of_range("native expert cache layer index is out of range");
  }
  if (layer_is_fully_resident(layer_index)) {
    // Every possible address already exists, so projection can use the current
    // snapshot immediately. Preserve the lazy route bitmap and postpone its host
    // wait until recency can affect an eviction decision.
    pending_route_evidence_.push_back(PendingRouteEvidence{
        layer_index,
        build_expert_selection_bitmap(
            selected_expert_indices,
            layer_profiles_[layer_index].expert_capacity,
            stream)});
    report.complete_layer_route_synchronization_elision_count = 1;
    if (!layer_profiles_[layer_index].current_snapshot) {
      throw std::runtime_error("complete native expert layer has no snapshot");
    }
    return PreparedExpertRoute{
        layer_profiles_[layer_index].current_snapshot, std::nullopt};
  }
  // Before changing global policy, account for every complete-layer route whose
  // synchronization was previously elided; otherwise a recently used page could
  // be selected as the least-recently-used victim.
  reconcile_pending_route_evidence(collect_performance_metrics, &report);
  auto& layer_profile = layer_profiles_[layer_index];
  auto selected_expert_ids = copy_sorted_unique_expert_ids_from_bitmap(
      build_expert_selection_bitmap(
          selected_expert_indices, layer_profile.expert_capacity, stream),
      layer_profile.expert_capacity,
      collect_performance_metrics,
      &report);
  if (selected_expert_ids.back() >= layer_profile.expert_capacity) {
    throw std::invalid_argument("selected expert ID exceeds layer capacity");
  }

  std::vector<size_t> missing_expert_ids;
  for (const auto expert_id : selected_expert_ids) {
    next_access_sequence_number_ += 1;
    const CacheKey cache_key{layer_index, expert_id};
    const auto cache_entry = cache_entries_.find(cache_key);
    if (cache_entry == cache_entries_.end()) {
      missing_expert_ids.push_back(expert_id);
      report.cache_miss_count += 1;
      cumulative_statistics_.cache_miss_count += 1;
    } else {
      cache_entry->second.last_access_sequence_number =
          next_access_sequence_number_;
      report.cache_hit_count += 1;
      cumulative_statistics_.cache_hit_count += 1;
    }
  }

  const uint64_t payload_bytes_per_expert =
      layer_profile.payload_byte_count_per_expert;
  if (selected_expert_ids.size() >
      std::numeric_limits<uint64_t>::max() / payload_bytes_per_expert) {
    throw std::overflow_error("selected expert route payload exceeds u64");
  }
  const uint64_t selected_route_payload_byte_count =
      payload_bytes_per_expert * selected_expert_ids.size();
  const uint64_t pending_payload_byte_count =
      payload_bytes_per_expert * missing_expert_ids.size();
  if (selected_route_payload_byte_count >
      maximum_retained_payload_byte_count_for_layer(layer_index)) {
    // A route larger than the retention ceiling still has to execute exactly.
    // Build a request-owned generation from retained and temporary slots, but do
    // not admit any new slot into global recency or resident-byte accounting.
    std::vector<std::pair<size_t, std::shared_ptr<ExpertSlot>>>
        ephemeral_expert_slots;
    ephemeral_expert_slots.reserve(selected_expert_ids.size());
    for (const auto expert_id : selected_expert_ids) {
      const auto cache_entry =
          cache_entries_.find(CacheKey{layer_index, expert_id});
      if (cache_entry != cache_entries_.end()) {
        ephemeral_expert_slots.emplace_back(expert_id, cache_entry->second.slot);
      } else {
        ephemeral_expert_slots.emplace_back(
            expert_id,
            load_expert_slot(
                layer_profile,
                expert_id,
                collect_performance_metrics,
                report));
      }
    }
    layer_profile.publication_generation += 1;
    report.page_table_publication_count = 1;
    report.payload_copy_byte_count = 0;
    cumulative_statistics_.disk_page_load_count += report.disk_page_load_count;
    cumulative_statistics_.disk_batch_load_count += report.disk_batch_load_count;
    return PreparedExpertRoute{
        build_page_table_snapshot(
            layer_profile.expert_capacity,
            layer_profile.publication_generation,
            layer_profile.storage_mode,
            ephemeral_expert_slots),
        std::move(selected_expert_ids)};
  }

  std::vector<size_t> changed_layer_indices;
  evict_layer_to_fit(
      layer_index,
      selected_expert_ids,
      pending_payload_byte_count,
      changed_layer_indices);
  // Protect the complete current route while reclaiming globally oldest entries
  // from any layer. Changed layers receive new immutable page-table generations
  // before removed cache owners can release their slots.
  evict_to_fit(
      layer_index,
      selected_expert_ids,
      pending_payload_byte_count,
      changed_layer_indices);
  if (resident_payload_byte_count_ + pending_payload_byte_count >
      maximum_resident_payload_byte_count_) {
    throw std::runtime_error(
        "native expert cache cannot fit the protected routed selection");
  }
  for (const auto expert_id : missing_expert_ids) {
    auto expert_slot =
        load_expert_slot(
            layer_profile,
            expert_id,
            collect_performance_metrics,
            report);
    resident_payload_byte_count_ += expert_slot->payload_byte_count;
    layer_profile.resident_payload_byte_count += expert_slot->payload_byte_count;
    layer_profile.resident_expert_count += 1;
    next_access_sequence_number_ += 1;
    cache_entries_.emplace(
        CacheKey{layer_index, expert_id},
        CacheEntry{std::move(expert_slot), next_access_sequence_number_});
    changed_layer_indices.push_back(layer_index);
  }
  report.page_table_publication_count =
      publish_changed_layers(changed_layer_indices);
  if (!layer_profiles_[layer_index].current_snapshot) {
    throw std::runtime_error("native expert layer has no published snapshot");
  }
  report.payload_copy_byte_count = 0;
  cumulative_statistics_.disk_page_load_count += report.disk_page_load_count;
  cumulative_statistics_.disk_batch_load_count += report.disk_batch_load_count;
  cumulative_statistics_.resident_expert_count = cache_entries_.size();
  cumulative_statistics_.resident_payload_byte_count =
      resident_payload_byte_count_;
  return PreparedExpertRoute{
      layer_profiles_[layer_index].current_snapshot,
      std::move(selected_expert_ids)};
}

}  // namespace astronomical::native_expert_cache
