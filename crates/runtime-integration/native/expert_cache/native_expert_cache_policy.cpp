#include "native_expert_cache_internal.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace astronomical::native_expert_cache {

static_assert(combined_payload_exceeds_byte_ceiling(
    std::numeric_limits<uint64_t>::max(),
    1,
    std::numeric_limits<uint64_t>::max()));

// Cache policy is separate from input/output. It owns layer-balanced recency,
// byte ceilings, eviction, and immutable page-table publication; no function
// here reads model payloads from disk.

uint64_t NativeExpertCache::proportional_retention_share_byte_count(
    size_t layer_index) const {
  const uint64_t layer_count = layer_profiles_.size();
  const uint64_t base_share =
      maximum_resident_payload_byte_count_ / layer_count;
  const uint64_t remainder_byte_count =
      maximum_resident_payload_byte_count_ % layer_count;
  return base_share + (layer_index < remainder_byte_count ? 1 : 0);
}

uint64_t NativeExpertCache::protected_retention_floor_byte_count_for_layer(
    size_t layer_index) const {
  // A page is indivisible. Round the fair byte share up to a complete expert so
  // a small layer share can still protect one useful decode page.
  const uint64_t proportional_share =
      proportional_retention_share_byte_count(layer_index);
  const uint64_t payload_bytes_per_expert =
      layer_profiles_.at(layer_index).payload_byte_count_per_expert;
  if (proportional_share == 0 || payload_bytes_per_expert == 0) {
    return 0;
  }
  const uint64_t complete_expert_count =
      proportional_share / payload_bytes_per_expert;
  const uint64_t rounded_expert_count = complete_expert_count +
      uint64_t{proportional_share % payload_bytes_per_expert != 0};
  if (rounded_expert_count >
      maximum_resident_payload_byte_count_ / payload_bytes_per_expert) {
    return maximum_resident_payload_byte_count_;
  }
  return rounded_expert_count * payload_bytes_per_expert;
}

uint64_t NativeExpertCache::maximum_borrowable_retained_payload_byte_count_for_route(
    size_t layer_index) const {
  // Imagine every other populated layer reserving a small parking space. This
  // route may use everything left over. Empty layers reserve nothing, so a hot
  // layer can borrow their unused bytes instead of wasting global capacity.
  uint64_t protected_other_layer_payload_byte_count = 0;
  for (size_t candidate_layer_index = 0;
       candidate_layer_index < layer_profiles_.size();
       ++candidate_layer_index) {
    if (candidate_layer_index == layer_index) {
      continue;
    }
    const uint64_t layer_floor_payload_byte_count =
        protected_retention_floor_byte_count_for_layer(candidate_layer_index);
    const uint64_t protected_layer_payload_byte_count = std::min(
        layer_profiles_[candidate_layer_index].resident_payload_byte_count,
        layer_floor_payload_byte_count);
    if (protected_layer_payload_byte_count >
        std::numeric_limits<uint64_t>::max() -
            protected_other_layer_payload_byte_count) {
      return 0;
    }
    protected_other_layer_payload_byte_count +=
        protected_layer_payload_byte_count;
  }
  return maximum_resident_payload_byte_count_ >=
          protected_other_layer_payload_byte_count
      ? maximum_resident_payload_byte_count_ -
            protected_other_layer_payload_byte_count
      : 0;
}

std::shared_ptr<const paged::PageTableSnapshot> build_page_table_snapshot(
    size_t expert_capacity,
    uint64_t generation,
    paged::ExpertPageStorageMode storage_mode,
    const std::vector<std::pair<size_t, std::shared_ptr<ExpertSlot>>>&
        expert_slots) {
  // Copy semantic array handles into an immutable generation. Each MLX array is
  // a shared owner of its paged slot, so snapshots remain valid after the cache
  // map releases an evicted entry.
  if (storage_mode == paged::ExpertPageStorageMode::NativeBfloat16) {
    std::vector<std::pair<size_t, paged::NativeBfloat16ExpertPageArrays>>
        native_source_pages;
    native_source_pages.reserve(expert_slots.size());
    for (const auto& [expert_id, expert_slot] : expert_slots) {
      if (!expert_slot->native_bfloat16_page_arrays.has_value()) {
        throw std::invalid_argument(
            "native bfloat16 cache slot has no native projection weights");
      }
      native_source_pages.emplace_back(
          expert_id, *expert_slot->native_bfloat16_page_arrays);
    }
    return paged::publish_native_bfloat16_snapshot(
        expert_capacity, generation, native_source_pages);
  }
  std::vector<std::pair<size_t, paged::ExpertPageArrays>> source_pages;
  source_pages.reserve(expert_slots.size());
  for (const auto& [expert_id, expert_slot] : expert_slots) {
    if (!expert_slot->quantized_page_arrays.has_value()) {
      throw std::invalid_argument(
          "affine cache slot has no quantized projection arrays");
    }
    const auto& page_arrays = *expert_slot->quantized_page_arrays;
    source_pages.emplace_back(expert_id, page_arrays);
  }
  return paged::publish_snapshot(expert_capacity, generation, source_pages);
}

void NativeExpertCache::evict_to_fit(
    size_t protected_layer_index,
    const std::vector<size_t>& protected_expert_ids,
    uint64_t pending_payload_byte_count,
    std::vector<size_t>& changed_layer_indices,
    uint64_t* evicted_payload_byte_count_output) {
  // There is one hard rule: retained bytes plus incoming bytes must fit the one
  // global ceiling. Layer floors only choose a sensible victim. First remove the
  // oldest page from a layer using more than its floor; if no such page exists,
  // remove the globally oldest unprotected page. Never remove this route's page.
  // Layer and expert IDs make equal-recency choices deterministic.
  while (combined_payload_exceeds_byte_ceiling(
      resident_payload_byte_count_,
      pending_payload_byte_count,
      maximum_resident_payload_byte_count_)) {
    auto overrepresented_layer_eviction_candidate = cache_entries_.end();
    auto fallback_eviction_candidate = cache_entries_.end();
    for (auto iterator = cache_entries_.begin();
         iterator != cache_entries_.end();
         ++iterator) {
      if (iterator->first.layer_index == protected_layer_index &&
          std::binary_search(
              protected_expert_ids.begin(),
              protected_expert_ids.end(),
              iterator->first.expert_id)) {
        continue;
      }
      if (fallback_eviction_candidate == cache_entries_.end() ||
          std::tie(
              iterator->second.last_access_sequence_number,
              iterator->first.layer_index,
              iterator->first.expert_id) <
              std::tie(
                  fallback_eviction_candidate->second.last_access_sequence_number,
                  fallback_eviction_candidate->first.layer_index,
                  fallback_eviction_candidate->first.expert_id)) {
        fallback_eviction_candidate = iterator;
      }
      const auto candidate_layer_index = iterator->first.layer_index;
      const uint64_t projected_layer_resident_payload_byte_count =
          layer_profiles_[candidate_layer_index].resident_payload_byte_count +
          (candidate_layer_index == protected_layer_index
               ? pending_payload_byte_count
               : 0);
      if (projected_layer_resident_payload_byte_count <=
          protected_retention_floor_byte_count_for_layer(candidate_layer_index)) {
        continue;
      }
      if (overrepresented_layer_eviction_candidate == cache_entries_.end() ||
          std::tie(
              iterator->second.last_access_sequence_number,
              iterator->first.layer_index,
              iterator->first.expert_id) <
              std::tie(
                  overrepresented_layer_eviction_candidate->second
                      .last_access_sequence_number,
                  overrepresented_layer_eviction_candidate->first.layer_index,
                  overrepresented_layer_eviction_candidate->first.expert_id)) {
        overrepresented_layer_eviction_candidate = iterator;
      }
    }
    const auto eviction_candidate =
        overrepresented_layer_eviction_candidate != cache_entries_.end()
        ? overrepresented_layer_eviction_candidate
        : fallback_eviction_candidate;
    if (eviction_candidate == cache_entries_.end()) {
      break;
    }
    const auto evicted_layer_index = eviction_candidate->first.layer_index;
    const auto evicted_payload_byte_count =
        eviction_candidate->second.slot->payload_byte_count;
    resident_payload_byte_count_ -= evicted_payload_byte_count;
    layer_profiles_[evicted_layer_index].resident_payload_byte_count -=
        evicted_payload_byte_count;
    layer_profiles_[evicted_layer_index].resident_expert_count -= 1;
    changed_layer_indices.push_back(evicted_layer_index);
    cache_entries_.erase(eviction_candidate);
    if (evicted_payload_byte_count_output != nullptr) {
      *evicted_payload_byte_count_output += evicted_payload_byte_count;
    }
    cumulative_statistics_.eviction_count += 1;
  }
}

void NativeExpertCache::publish_layer(size_t layer_index) {
  auto& layer_profile = layer_profiles_.at(layer_index);
  std::vector<std::pair<size_t, std::shared_ptr<ExpertSlot>>> expert_slots;
  for (const auto& [cache_key, cache_entry] : cache_entries_) {
    if (cache_key.layer_index == layer_index) {
      expert_slots.emplace_back(cache_key.expert_id, cache_entry.slot);
    }
  }
  std::sort(
      expert_slots.begin(),
      expert_slots.end(),
      [](const auto& left, const auto& right) {
        return left.first < right.first;
      });
  // Publish the replacement generation before local shared owners leave scope.
  // Older lazy graphs retain their previous generation independently.
  layer_profile.publication_generation += 1;
  layer_profile.current_snapshot = build_page_table_snapshot(
      layer_profile.expert_capacity,
      layer_profile.publication_generation,
      layer_profile.storage_mode,
      expert_slots);
}

uint64_t NativeExpertCache::publish_changed_layers(
    const std::vector<size_t>& changed_layer_indices) {
  auto unique_layer_indices = changed_layer_indices;
  std::sort(unique_layer_indices.begin(), unique_layer_indices.end());
  unique_layer_indices.erase(
      std::unique(unique_layer_indices.begin(), unique_layer_indices.end()),
      unique_layer_indices.end());
  for (const auto layer_index : unique_layer_indices) {
    publish_layer(layer_index);
  }
  return unique_layer_indices.size();
}

astronomical_native_expert_cache_statistics NativeExpertCache::statistics()
    const {
  auto statistics = cumulative_statistics_;
  statistics.resident_expert_count = cache_entries_.size();
  statistics.resident_payload_byte_count = resident_payload_byte_count_;
  statistics.maximum_resident_payload_byte_count =
      maximum_resident_payload_byte_count_;
  return statistics;
}

void NativeExpertCache::enforce_maximum_resident_payload_byte_count(
    uint64_t maximum_resident_payload_byte_count) {
  if (maximum_resident_payload_byte_count <
      maximum_resident_payload_byte_count_) {
    reconcile_pending_route_evidence();
  }
  maximum_resident_payload_byte_count_ =
      maximum_resident_payload_byte_count;
  std::vector<size_t> changed_layer_indices;
  // Example: the new ceiling is 100 bytes and the cache already holds 90 bytes.
  // Nothing must be evicted, even if one layer owns most of those 90 bytes. The
  // old per-layer pre-pass could evict in that situation. One global loop cannot.
  evict_to_fit(0, {}, 0, changed_layer_indices);
  publish_changed_layers(changed_layer_indices);
}

void NativeExpertCache::update_maximum_resident_payload_byte_count(
    uint64_t maximum_resident_payload_byte_count) {
  // Remember the newest model-derived policy while a request-scoped freeze is
  // active. A higher ceiling remains deferred, but a lower ceiling must evict
  // immediately so route admission never observes resident bytes above policy.
  configured_maximum_resident_payload_byte_count_ =
      maximum_resident_payload_byte_count;
  if (retention_growth_frozen_ &&
      maximum_resident_payload_byte_count >=
          maximum_resident_payload_byte_count_) {
    return;
  }
  enforce_maximum_resident_payload_byte_count(
      maximum_resident_payload_byte_count);
}

bool NativeExpertCache::freeze_retention_growth() noexcept {
  if (retention_growth_frozen_) {
    return false;
  }
  // Freeze is deliberately non-destructive: request admission uses it when only
  // a soft recovery reserve is short and preserving the hot set is still safe.
  retention_growth_frozen_ = true;
  maximum_resident_payload_byte_count_ = resident_payload_byte_count_;
  return true;
}

bool NativeExpertCache::reclaim_retained_payload_bytes(
    uint64_t reclamation_target_byte_count) {
  retention_growth_frozen_ = true;
  const uint64_t previous_resident_payload_byte_count =
      resident_payload_byte_count_;
  const uint64_t reclaimed_maximum_resident_payload_byte_count =
      resident_payload_byte_count_ > reclamation_target_byte_count
      ? resident_payload_byte_count_ - reclamation_target_byte_count
      : 0;
  enforce_maximum_resident_payload_byte_count(
      reclaimed_maximum_resident_payload_byte_count);
  return resident_payload_byte_count_ < previous_resident_payload_byte_count;
}

bool NativeExpertCache::resume_retention_growth() noexcept {
  if (!retention_growth_frozen_) {
    return false;
  }
  retention_growth_frozen_ = false;
  // Every lower configured ceiling is enforced even while frozen, so resume can
  // only preserve or increase the active ceiling and cannot strand excess pages.
  maximum_resident_payload_byte_count_ =
      configured_maximum_resident_payload_byte_count_;
  return true;
}

}  // namespace astronomical::native_expert_cache
