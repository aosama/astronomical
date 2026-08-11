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
  const auto source_read_started_at = collect_performance_metrics
      ? std::optional{std::chrono::steady_clock::now()}
      : std::nullopt;
  mx::read_paged_buffer_ranges(source_read_ranges);
  if (source_read_started_at.has_value()) {
    report.successful_source_read_elapsed_nanoseconds +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - *source_read_started_at)
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
  // Tests and simple callers still use this one-call convenience path. Production
  // calls the same two phases separately so Rust can measure memory between them.
  auto route_analysis = analyze_layer(
      layer_index,
      selected_expert_indices,
      stream,
      collect_performance_metrics,
      report);
  return commit_layer(
      std::move(route_analysis),
      maximum_resident_payload_byte_count_,
      stream,
      collect_performance_metrics,
      report);
}

NativeExpertRouteAnalysis NativeExpertCache::analyze_layer(
    size_t layer_index,
    const mx::array& selected_expert_indices,
    mx::Stream stream,
    bool collect_performance_metrics,
    astronomical_native_expert_cache_request_report& report) {
  if (layer_index >= layer_profiles_.size()) {
    throw std::out_of_range("native expert cache layer index is out of range");
  }
  // Router output can contain thousands of assignments but only a small number
  // of distinct experts. Example: [7, 7, 7, 7] means one expert page, not four.
  // The bitmap performs that deduplication before any cache page is removed.
  report.selected_expert_assignment_count = selected_expert_indices.size();
  report.retention_ceiling_before_byte_count =
      maximum_resident_payload_byte_count_;
  auto selected_expert_bitmap = build_expert_selection_bitmap(
      selected_expert_indices,
      layer_profiles_[layer_index].expert_capacity,
      stream);
  if (layer_is_fully_resident(layer_index)) {
    // Every possible address already exists, so route preparation can preserve
    // the lazy bitmap and postpone its host wait until recency can affect an
    // eviction decision.
    report.complete_layer_route_synchronization_elision_count = 1;
    if (!layer_profiles_[layer_index].current_snapshot) {
      throw std::runtime_error("complete native expert layer has no snapshot");
    }
    return NativeExpertRouteAnalysis{
        layer_index,
        std::nullopt,
        std::move(selected_expert_bitmap),
        layer_profiles_[layer_index].current_snapshot};
  }
  // Before changing global policy, account for every complete-layer route whose
  // synchronization was previously elided; otherwise a recently used page could
  // be selected as the least-recently-used victim.
  reconcile_pending_route_evidence(collect_performance_metrics, &report);
  auto selected_expert_ids = copy_sorted_unique_expert_ids_from_bitmap(
      std::move(selected_expert_bitmap),
      layer_profiles_[layer_index].expert_capacity,
      collect_performance_metrics,
      &report);
  // These are the exact numbers Rust needs. "Selected" means used by this
  // forward. "Missing" means selected but not already retained in the cache.
  const auto missing_expert_ids =
      missing_expert_ids_for_route(layer_index, selected_expert_ids);
  report.distinct_route_expert_count = selected_expert_ids.size();
  report.missing_route_expert_count = missing_expert_ids.size();
  report.selected_route_payload_byte_count =
      payload_byte_count_for_expert_ids(layer_index, selected_expert_ids);
  report.missing_route_payload_byte_count =
      payload_byte_count_for_expert_ids(layer_index, missing_expert_ids);
  return NativeExpertRouteAnalysis{
      layer_index, std::move(selected_expert_ids), std::nullopt, nullptr};
}

PreparedExpertRoute NativeExpertCache::commit_layer(
    NativeExpertRouteAnalysis route_analysis,
    uint64_t maximum_resident_payload_byte_count,
    mx::Stream /*stream*/,
    bool collect_performance_metrics,
    astronomical_native_expert_cache_request_report& report) {
  const auto layer_index = route_analysis.layer_index;
  if (layer_index >= layer_profiles_.size()) {
    throw std::out_of_range("native expert cache route layer index is out of range");
  }
  // A complete layer normally skips bitmap synchronization. It cannot keep that
  // shortcut if the new ceiling will evict pages: policy must know the exact
  // current route first so it never removes an expert this forward will use.
  const bool route_ceiling_can_evict =
      (!retention_growth_frozen_ ||
       maximum_resident_payload_byte_count <
           maximum_resident_payload_byte_count_) &&
      resident_payload_byte_count_ > maximum_resident_payload_byte_count;
  if (!route_analysis.selected_expert_ids.has_value() &&
      route_ceiling_can_evict) {
    if (!route_analysis.selected_expert_bitmap.has_value()) {
      throw std::runtime_error("native expert route analysis has no route bitmap");
    }
    reconcile_pending_route_evidence(collect_performance_metrics, &report);
    auto selected_expert_ids = copy_sorted_unique_expert_ids_from_bitmap(
        std::move(*route_analysis.selected_expert_bitmap),
        layer_profiles_[layer_index].expert_capacity,
        collect_performance_metrics,
        &report);
    report.complete_layer_route_synchronization_elision_count = 0;
    report.distinct_route_expert_count = selected_expert_ids.size();
    report.selected_route_payload_byte_count =
        payload_byte_count_for_expert_ids(layer_index, selected_expert_ids);
    route_analysis.selected_expert_ids = std::move(selected_expert_ids);
    route_analysis.selected_expert_bitmap.reset();
  }
  // Install the new ceiling and evict in the same native transaction. Doing
  // these as separate calls would create a dangerous gap where a route hit could
  // be evicted after analysis but before its snapshot is published.
  configured_maximum_resident_payload_byte_count_ =
      maximum_resident_payload_byte_count;
  if (!retention_growth_frozen_ ||
      maximum_resident_payload_byte_count <
          maximum_resident_payload_byte_count_) {
    maximum_resident_payload_byte_count_ =
        maximum_resident_payload_byte_count;
    std::vector<size_t> ceiling_changed_layer_indices;
    uint64_t ceiling_evicted_payload_byte_count = 0;
    const std::vector<size_t> no_selected_expert_ids;
    const auto& protected_expert_ids =
        route_analysis.selected_expert_ids.has_value()
        ? *route_analysis.selected_expert_ids
        : no_selected_expert_ids;
    evict_to_fit(
        layer_index,
        protected_expert_ids,
        0,
        ceiling_changed_layer_indices,
        &ceiling_evicted_payload_byte_count);
    report.evicted_payload_byte_count +=
        ceiling_evicted_payload_byte_count;
    publish_changed_layers(ceiling_changed_layer_indices);
  }
  if (!route_analysis.selected_expert_ids.has_value()) {
    if (!route_analysis.selected_expert_bitmap.has_value()) {
      throw std::runtime_error("native expert route analysis has no route evidence");
    }
    // No eviction was needed, so every address still exists. Preserve the lazy
    // bitmap and postpone its host wait until recency can affect a later eviction.
    pending_route_evidence_.push_back(PendingRouteEvidence{
        layer_index,
        std::move(*route_analysis.selected_expert_bitmap)});
    if (!route_analysis.complete_layer_snapshot) {
      throw std::runtime_error("complete native expert layer has no snapshot");
    }
    report.retention_ceiling_after_byte_count =
        maximum_resident_payload_byte_count_;
    return PreparedExpertRoute{
        std::move(route_analysis.complete_layer_snapshot), std::nullopt};
  }

  const auto& selected_expert_ids = *route_analysis.selected_expert_ids;
  auto missing_expert_ids =
      missing_expert_ids_for_route(layer_index, selected_expert_ids);
  report.cache_hit_count = selected_expert_ids.size() - missing_expert_ids.size();
  report.cache_miss_count = missing_expert_ids.size();
  cumulative_statistics_.cache_hit_count += report.cache_hit_count;
  cumulative_statistics_.cache_miss_count += report.cache_miss_count;
  for (const auto expert_id : selected_expert_ids) {
    next_access_sequence_number_ += 1;
    const auto cache_entry = cache_entries_.find(CacheKey{layer_index, expert_id});
    if (cache_entry != cache_entries_.end()) {
      cache_entry->second.last_access_sequence_number =
          next_access_sequence_number_;
    }
  }

  auto& layer_profile = layer_profiles_[layer_index];
  const uint64_t selected_route_payload_byte_count =
      report.selected_route_payload_byte_count;
  const uint64_t pending_payload_byte_count =
      report.missing_route_payload_byte_count;
  if (selected_route_payload_byte_count >
      maximum_borrowable_retained_payload_byte_count_for_route(layer_index)) {
    // The route is larger than the bytes this layer may safely retain after
    // protecting other layers' floors. Correctness still wins: build a temporary
    // request-owned snapshot from hits plus exact misses. Missing pages serve
    // this forward but do not enter the long-lived cache.
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
    // Publish the temporary snapshot before logical eviction. Shared ownership
    // keeps every address used by this forward alive even after the cache map
    // stops counting the page as retained.
    auto ephemeral_snapshot = build_page_table_snapshot(
        layer_profile.expert_capacity,
        layer_profile.publication_generation,
        layer_profile.storage_mode,
        ephemeral_expert_slots);
    std::vector<size_t> changed_layer_indices_after_ephemeral_publication;
    uint64_t evicted_payload_byte_count_after_ephemeral_publication = 0;
    evict_to_fit(
        layer_index,
        {},
        0,
        changed_layer_indices_after_ephemeral_publication,
        &evicted_payload_byte_count_after_ephemeral_publication);
    report.evicted_payload_byte_count +=
        evicted_payload_byte_count_after_ephemeral_publication;
    report.page_table_publication_count += publish_changed_layers(
        changed_layer_indices_after_ephemeral_publication);
    report.retention_ceiling_after_byte_count =
        maximum_resident_payload_byte_count_;
    return PreparedExpertRoute{
        std::move(ephemeral_snapshot),
        std::move(selected_expert_ids)};
  }

  std::vector<size_t> changed_layer_indices;
  uint64_t evicted_payload_byte_count = 0;
  // This route fits retention. Protect it while reclaiming globally oldest entries
  // from extras above proportional layer floors first, then global LRU entries.
  // Unused capacity is therefore borrowable rather than a hard per-layer cap.
  // Changed layers receive new immutable page-table generations before removed
  // cache owners can release their slots.
  evict_to_fit(
      layer_index,
      selected_expert_ids,
      pending_payload_byte_count,
      changed_layer_indices,
      &evicted_payload_byte_count);
  report.evicted_payload_byte_count += evicted_payload_byte_count;
  if (combined_payload_exceeds_byte_ceiling(
      resident_payload_byte_count_,
      pending_payload_byte_count,
      maximum_resident_payload_byte_count_)) {
    throw std::runtime_error(
        "native expert cache cannot fit the protected routed selection");
  }

  // Publish logical evictions before allocating replacement pages. This drops
  // cache-owned references to removed slots as early as snapshot safety allows
  // and leaves policy and current snapshots consistent even if a later source
  // read fails. External in-flight snapshots keep only the pages they still use.
  report.page_table_publication_count +=
      publish_changed_layers(changed_layer_indices);
  changed_layer_indices.clear();

  // Load every missing page into transaction-local ownership first. If any
  // later range read fails, this vector drops all earlier candidates and the
  // cache map, byte accounting, and published snapshot remain mutually
  // consistent. Existing evictions may remain, but no partial new route enters
  // long-lived retention.
  std::vector<std::pair<size_t, std::shared_ptr<ExpertSlot>>>
      loaded_missing_expert_slots;
  loaded_missing_expert_slots.reserve(missing_expert_ids.size());
  for (const auto expert_id : missing_expert_ids) {
    loaded_missing_expert_slots.emplace_back(
        expert_id,
        load_expert_slot(
            layer_profile,
            expert_id,
            collect_performance_metrics,
            report));
  }
  for (auto& [expert_id, expert_slot] : loaded_missing_expert_slots) {
    resident_payload_byte_count_ += expert_slot->payload_byte_count;
    layer_profile.resident_payload_byte_count += expert_slot->payload_byte_count;
    layer_profile.resident_expert_count += 1;
    next_access_sequence_number_ += 1;
    cache_entries_.emplace(
        CacheKey{layer_index, expert_id},
        CacheEntry{std::move(expert_slot), next_access_sequence_number_});
    changed_layer_indices.push_back(layer_index);
  }
  report.page_table_publication_count +=
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
  report.retention_ceiling_after_byte_count =
      maximum_resident_payload_byte_count_;
  return PreparedExpertRoute{
      layer_profiles_[layer_index].current_snapshot,
      std::move(selected_expert_ids)};
}

uint64_t NativeExpertCache::payload_byte_count_for_expert_ids(
    size_t layer_index,
    const std::vector<size_t>& expert_ids) const {
  const auto payload_bytes_per_expert =
      layer_profiles_.at(layer_index).payload_byte_count_per_expert;
  if (expert_ids.size() >
      std::numeric_limits<uint64_t>::max() / payload_bytes_per_expert) {
    throw std::overflow_error("selected expert route payload exceeds u64");
  }
  return payload_bytes_per_expert * expert_ids.size();
}

std::vector<size_t> NativeExpertCache::missing_expert_ids_for_route(
    size_t layer_index,
    const std::vector<size_t>& selected_expert_ids) const {
  std::vector<size_t> missing_expert_ids;
  missing_expert_ids.reserve(selected_expert_ids.size());
  for (const auto expert_id : selected_expert_ids) {
    if (cache_entries_.find(CacheKey{layer_index, expert_id}) ==
        cache_entries_.end()) {
      missing_expert_ids.push_back(expert_id);
    }
  }
  return missing_expert_ids;
}

}  // namespace astronomical::native_expert_cache
