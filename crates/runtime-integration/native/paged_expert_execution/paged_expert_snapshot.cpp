#include "paged_expert_execution_internal.h"

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/dtype.h"

namespace mx = mlx::core;

namespace astronomical::paged_expert_execution {

// Snapshot publication converts retained MLX arrays into a fixed expert-ID to
// graphics-processor-address table while preserving those arrays as lifetime
// owners. It performs no route selection or matrix work.

namespace {

struct alignas(8) MetalExpertPageEntry {
  std::array<uint64_t, 9> resource_addresses{};
  uint32_t presence{0};
  uint32_t generation{0};
};

// This host layout is re-declared in every Metal kernel. Changing it requires
// updating all consumers as one contract.
static_assert(sizeof(MetalExpertPageEntry) == 80);

bool is_supported_affine_parameter_dtype(mx::Dtype dtype) {
  return dtype == mx::float16 || dtype == mx::bfloat16 ||
      dtype == mx::float32;
}

uint64_t array_gpu_address(const mx::array& array) {
  if (array.buffer().ptr() == nullptr) {
    throw std::runtime_error(
        "expert page must be evaluated before page-table publication");
  }
  // The raw address carries no ownership. PageTableSnapshot must retain this
  // array and the command encoder must separately register its Metal resource.
  auto* metal_buffer = static_cast<const MTL::Buffer*>(array.buffer().ptr());
  return metal_buffer->gpuAddress() + array.offset();
}

void validate_projection_arrays(
    const QuantizedProjectionArrays& projection,
    const char* projection_name) {
  const auto& weight_shape = projection.packed_weight.shape();
  const auto& scale_shape = projection.scales.shape();
  if (projection.packed_weight.dtype() != mx::uint32 ||
      weight_shape.size() != 3 || weight_shape.front() != 1 ||
      scale_shape.size() != 3 || scale_shape.front() != 1 ||
      scale_shape != projection.biases.shape() ||
      !is_supported_affine_parameter_dtype(projection.scales.dtype()) ||
      !is_supported_affine_parameter_dtype(projection.biases.dtype()) ||
      weight_shape[1] != scale_shape[1] ||
      (projection.group_size != 32 && projection.group_size != 64 &&
       projection.group_size != 128) ||
      (projection.bits != 2 && projection.bits != 3 &&
       projection.bits != 4 && projection.bits != 5 &&
       projection.bits != 6 && projection.bits != 8)) {
    throw std::invalid_argument(
        std::string(projection_name) +
        " expert page has incompatible affine tensor geometry");
  }
  const uint64_t scale_covered_bit_count =
      static_cast<uint64_t>(scale_shape[2]) * projection.group_size *
      projection.bits;
  const uint64_t packed_bit_count =
      static_cast<uint64_t>(weight_shape[2]) * 32;
  if (scale_covered_bit_count != packed_bit_count) {
    throw std::invalid_argument(
        std::string(projection_name) +
        " expert page has incompatible affine tensor geometry");
  }
}

void validate_projection_compatibility(
    const QuantizedProjectionArrays& expected_projection,
    const QuantizedProjectionArrays& candidate_projection,
    const char* projection_name) {
  if (candidate_projection.packed_weight.shape() !=
          expected_projection.packed_weight.shape() ||
      candidate_projection.scales.shape() != expected_projection.scales.shape() ||
      candidate_projection.scales.dtype() != expected_projection.scales.dtype() ||
      candidate_projection.biases.dtype() != expected_projection.biases.dtype() ||
      candidate_projection.group_size != expected_projection.group_size ||
      candidate_projection.bits != expected_projection.bits) {
    throw std::invalid_argument(
        std::string(projection_name) +
        " expert pages do not share one executable affine profile");
  }
}

MetalExpertPageEntry metal_entry_for_page(
    const ExpertPageArrays& page,
    uint64_t generation) {
  MetalExpertPageEntry metal_entry;
  size_t resource_address_index = 0;
  for (const auto& projection : page.projections) {
    metal_entry.resource_addresses[resource_address_index++] =
        array_gpu_address(projection.packed_weight);
    metal_entry.resource_addresses[resource_address_index++] =
        array_gpu_address(projection.scales);
    metal_entry.resource_addresses[resource_address_index++] =
        array_gpu_address(projection.biases);
  }
  metal_entry.presence = 1;
  metal_entry.generation = static_cast<uint32_t>(generation);
  return metal_entry;
}

mx::array make_metal_page_table(
    const std::vector<MetalExpertPageEntry>& metal_entries) {
  const size_t byte_count = page_table_metadata_byte_count(metal_entries.size());
  mx::array metal_page_table(
      {static_cast<int>(byte_count)}, mx::uint8, nullptr, {});
  metal_page_table.set_data(mx::allocator::malloc(byte_count));
  std::memcpy(
      metal_page_table.data<uint8_t>(), metal_entries.data(), byte_count);
  metal_page_table.set_status(mx::array::Status::available);
  return metal_page_table;
}

}  // namespace

size_t page_table_metadata_byte_count(size_t expert_capacity) {
  return expert_capacity * sizeof(MetalExpertPageEntry);
}

std::shared_ptr<const PageTableSnapshot> publish_snapshot(
    size_t expert_capacity,
    uint64_t generation,
    const std::vector<std::pair<size_t, ExpertPageArrays>>& source_pages) {
  // Keep a capacity-indexed host owner table aligned with the fixed-capacity
  // Metal address table. Holes have presence=0 and zero addresses.
  std::vector<std::optional<ExpertPageArrays>> retained_pages(expert_capacity);
  std::vector<MetalExpertPageEntry> metal_entries(expert_capacity);
  std::unordered_set<size_t> published_expert_ids;
  std::optional<ExpertPageArrays> first_retained_page;
  for (const auto& [expert_id, source_page] : source_pages) {
    if (expert_id >= expert_capacity ||
        !published_expert_ids.insert(expert_id).second) {
      throw std::invalid_argument(
          "expert page ID is duplicated or exceeds table capacity");
    }
    validate_projection_arrays(source_page.projections[0], "gate");
    validate_projection_arrays(source_page.projections[1], "up");
    validate_projection_arrays(source_page.projections[2], "down");
    retained_pages[expert_id] = source_page;
    if (first_retained_page.has_value()) {
      validate_projection_compatibility(
          first_retained_page->projections[0],
          retained_pages[expert_id]->projections[0],
          "gate");
      validate_projection_compatibility(
          first_retained_page->projections[1],
          retained_pages[expert_id]->projections[1],
          "up");
      validate_projection_compatibility(
          first_retained_page->projections[2],
          retained_pages[expert_id]->projections[2],
          "down");
    } else {
      first_retained_page = retained_pages[expert_id];
    }
    metal_entries[expert_id] = metal_entry_for_page(
        *retained_pages[expert_id], generation);
  }
  return std::make_shared<PageTableSnapshot>(PageTableSnapshot{
      generation,
      expert_capacity,
      source_pages.size(),
      ExpertPageStorageMode::QuantizedAffine,
      std::move(retained_pages),
      {},
      make_metal_page_table(metal_entries)});
}

}  // namespace astronomical::paged_expert_execution
