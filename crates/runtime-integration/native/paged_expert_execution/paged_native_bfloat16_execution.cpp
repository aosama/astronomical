#include "paged_expert_execution_internal.h"

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "mlx/allocator.h"
#include "mlx/backend/common/broadcasting.h"
#include "mlx/backend/common/compiled.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"
#include "paged_native_bfloat16_kernel_source.h"

namespace mx = mlx::core;

namespace astronomical::paged_expert_execution {

// Native BF16 has a dedicated page publisher and gathered matrix primitive so
// the artifact's precision survives unchanged from positional read to output.

namespace {

struct alignas(8) MetalNativeExpertPageEntry {
  // Preserve the nine-address layout used by affine snapshots. Native BF16 uses
  // only indices 0, 3, and 6 so both storage modes retain one metadata contract.
  std::array<uint64_t, 9> resource_addresses{};
  uint32_t presence{0};
  uint32_t generation{0};
};

static_assert(sizeof(MetalNativeExpertPageEntry) == 80);

uint64_t array_gpu_address(const mx::array& array) {
  if (array.buffer().ptr() == nullptr) {
    throw std::runtime_error(
        "native bfloat16 expert must be materialized before publication");
  }
  auto* metal_buffer = static_cast<const MTL::Buffer*>(array.buffer().ptr());
  return metal_buffer->gpuAddress() + array.offset();
}

mx::array make_metal_page_table(
    const std::vector<MetalNativeExpertPageEntry>& metal_entries) {
  const size_t byte_count = metal_entries.size() * sizeof(MetalNativeExpertPageEntry);
  mx::array metal_page_table(
      {static_cast<int>(byte_count)}, mx::uint8, nullptr, {});
  metal_page_table.set_data(mx::allocator::malloc(byte_count));
  std::memcpy(metal_page_table.data<uint8_t>(), metal_entries.data(), byte_count);
  metal_page_table.set_status(mx::array::Status::available);
  return metal_page_table;
}

void validate_native_page(const NativeBfloat16ExpertPageArrays& page) {
  for (const auto& projection_weight : page.projection_weights) {
    if (projection_weight.dtype() != mx::bfloat16 ||
        projection_weight.shape().size() != 3 ||
        projection_weight.shape().front() != 1) {
      throw std::invalid_argument(
          "native bfloat16 expert weight must have shape [1, output, input]");
    }
  }
}

void validate_native_page_compatibility(
    const NativeBfloat16ExpertPageArrays& expected_page,
    const NativeBfloat16ExpertPageArrays& candidate_page) {
  for (int projection_index = 0; projection_index < kProjectionCount;
       ++projection_index) {
    if (expected_page.projection_weights[projection_index].shape() !=
        candidate_page.projection_weights[projection_index].shape()) {
      throw std::invalid_argument(
          "native bfloat16 expert pages do not share one projection geometry");
    }
  }
}

mx::array row_contiguous_array(
    const mx::array& source_array,
    const mx::Stream& stream) {
  if (source_array.flags().row_contiguous) {
    return source_array;
  }
  auto contiguous_array = mx::contiguous_copy_gpu(source_array, stream);
  mx::metal::get_command_encoder(stream).add_temporary(contiguous_array);
  return contiguous_array;
}

const mx::array& first_native_projection(
    const PageTableSnapshot& snapshot,
    int projection_index) {
  for (const auto& expert_page : snapshot.native_bfloat16_expert_pages) {
    if (expert_page.has_value()) {
      return expert_page->projection_weights[projection_index];
    }
  }
  throw std::invalid_argument(
      "paged native bfloat16 snapshot has no resident experts");
}

class PagedGatherNativeBfloat16Matrix final : public mx::UnaryPrimitive {
 public:
  PagedGatherNativeBfloat16Matrix(
      mx::Stream stream,
      std::shared_ptr<const PageTableSnapshot> snapshot,
      std::optional<std::vector<size_t>> selected_expert_ids,
      int projection_index)
      : UnaryPrimitive(stream),
        snapshot_(std::move(snapshot)),
        selected_expert_ids_(std::move(selected_expert_ids)),
        projection_index_(projection_index) {}

  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error(
        "paged native bfloat16 matrix multiplication requires Metal");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      mx::array& output) override {
    auto& stream = this->stream();
    auto activations = row_contiguous_array(inputs[0], stream);
    auto selected_indices = row_contiguous_array(inputs[1], stream);
    const int input_dimension = activations.shape(-1);
    const int output_dimension = output.shape(-1);
    const int routed_assignment_count = selected_indices.size();
    // Broadcast one token row across its routed experts without materializing a
    // complete stacked expert-weight tensor. The custom kernel reads each
    // selected matrix directly from its independently retained page.
    if (activations.size() / input_dimension != routed_assignment_count) {
      auto broadcast_shape = selected_indices.shape();
      broadcast_shape.push_back(activations.shape(-2));
      broadcast_shape.push_back(input_dimension);
      mx::array broadcast_activations(
          std::move(broadcast_shape), activations.dtype(), nullptr, {});
      mx::broadcast(activations, broadcast_activations);
      activations = row_contiguous_array(broadcast_activations, stream);
    }
    if (activations.size() / input_dimension != routed_assignment_count ||
        activations.shape(-2) != 1) {
      throw std::invalid_argument(
          "paged native bfloat16 execution requires one matrix row per assignment");
    }
    output.set_data(mx::allocator::malloc(output.nbytes()));
    std::string kernel_name;
    mx::concatenate(
        kernel_name,
        "astronomical_paged_gather_native_bfloat16_matrix_",
        mx::get_type_string(activations.dtype()),
        "_p_",
        projection_index_);
    std::string template_definition = kPagedNativeBfloat16KernelSource;
    template_definition += mx::get_template_definition(
        kernel_name,
        "astronomical_paged_gather_native_bfloat16_matrix",
        mx::get_type_string(activations.dtype()),
        projection_index_);
    auto& metal_device = mx::metal::device(stream.device);
    auto* kernel_library = metal_device.get_library(kernel_name, [&]() {
      std::string kernel_source;
      mx::concatenate(
          kernel_source,
          mx::metal::utils(),
          template_definition);
      return kernel_source;
    });
    auto* kernel = metal_device.get_kernel(
        kernel_name, kernel_library, kernel_name);
    auto& command_encoder = mx::metal::get_command_encoder(stream);
    command_encoder.set_compute_pipeline_state(kernel);
    command_encoder.set_input_array(snapshot_->metal_page_table, 0);
    command_encoder.set_input_array(activations, 1);
    command_encoder.set_input_array(selected_indices, 2);
    command_encoder.set_output_array(output, 3);
    command_encoder.set_bytes(input_dimension, 4);
    command_encoder.set_bytes(output_dimension, 5);
    // GPU addresses in the table do not establish Metal resource residency or
    // hazard tracking. Register every possible backing array before dispatch.
    std::vector<const mx::array*> indirect_projection_resources;
    indirect_projection_resources.reserve(
        selected_expert_ids_.has_value()
            ? selected_expert_ids_->size()
            : snapshot_->resident_expert_count);
    const auto append_projection_resource = [&](size_t expert_id) {
      if (expert_id >= snapshot_->native_bfloat16_expert_pages.size() ||
          !snapshot_->native_bfloat16_expert_pages[expert_id].has_value()) {
        throw std::invalid_argument(
            "paged native bfloat16 route references a nonresident expert");
      }
      indirect_projection_resources.push_back(
          &snapshot_->native_bfloat16_expert_pages[expert_id]
               ->projection_weights[projection_index_]);
    };
    if (selected_expert_ids_.has_value()) {
      for (const auto expert_id : *selected_expert_ids_) {
        append_projection_resource(expert_id);
      }
    } else {
      for (size_t expert_id = 0;
           expert_id < snapshot_->native_bfloat16_expert_pages.size();
           ++expert_id) {
        if (snapshot_->native_bfloat16_expert_pages[expert_id].has_value()) {
          append_projection_resource(expert_id);
        }
      }
    }
    command_encoder.register_indirect_input_arrays(indirect_projection_resources);
    command_encoder.dispatch_threadgroups(
        MTL::Size(output_dimension, routed_assignment_count, 1),
        MTL::Size(32, 1, 1));
  }

  const char* name() const override {
    return "AstronomicalPagedGatherNativeBfloat16Matrix";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto* paged_other =
        dynamic_cast<const PagedGatherNativeBfloat16Matrix*>(&other);
    return paged_other != nullptr &&
        paged_other->snapshot_.get() == snapshot_.get() &&
        paged_other->selected_expert_ids_ == selected_expert_ids_ &&
        paged_other->projection_index_ == projection_index_;
  }

 private:
  std::shared_ptr<const PageTableSnapshot> snapshot_;
  std::optional<std::vector<size_t>> selected_expert_ids_;
  int projection_index_;
};

}  // namespace

std::shared_ptr<const PageTableSnapshot> publish_native_bfloat16_snapshot(
    size_t expert_capacity,
    uint64_t generation,
    const std::vector<
        std::pair<size_t, NativeBfloat16ExpertPageArrays>>& source_pages) {
  // Host array owners and Metal entries share one fixed expert-ID index. Missing
  // entries remain absent rather than receiving fabricated affine parameters.
  std::vector<std::optional<NativeBfloat16ExpertPageArrays>> retained_pages(
      expert_capacity);
  std::vector<MetalNativeExpertPageEntry> metal_entries(expert_capacity);
  std::unordered_set<size_t> published_expert_ids;
  std::optional<NativeBfloat16ExpertPageArrays> first_retained_page;
  for (const auto& [expert_id, source_page] : source_pages) {
    if (expert_id >= expert_capacity ||
        !published_expert_ids.insert(expert_id).second) {
      throw std::invalid_argument(
          "native bfloat16 expert ID is duplicated or exceeds table capacity");
    }
    validate_native_page(source_page);
    if (first_retained_page.has_value()) {
      validate_native_page_compatibility(*first_retained_page, source_page);
    } else {
      first_retained_page = source_page;
    }
    retained_pages[expert_id] = source_page;
    auto& metal_entry = metal_entries[expert_id];
    for (size_t projection_index = 0; projection_index < kProjectionCount;
         ++projection_index) {
      metal_entry.resource_addresses[projection_index * 3] =
          array_gpu_address(source_page.projection_weights[projection_index]);
    }
    metal_entry.presence = 1;
    metal_entry.generation = static_cast<uint32_t>(generation);
  }
  return std::make_shared<PageTableSnapshot>(PageTableSnapshot{
      generation,
      expert_capacity,
      source_pages.size(),
      ExpertPageStorageMode::NativeBfloat16,
      {},
      std::move(retained_pages),
      make_metal_page_table(metal_entries)});
}

mx::array build_paged_native_bfloat16_product(
    const std::shared_ptr<const PageTableSnapshot>& snapshot,
    const std::optional<std::vector<size_t>>& selected_expert_ids,
    int projection_index,
    const mx::array& activations,
    const mx::array& selected_indices,
    mx::Stream stream) {
  if (snapshot->storage_mode != ExpertPageStorageMode::NativeBfloat16) {
    throw std::invalid_argument(
        "paged native bfloat16 product requires a native page-table snapshot");
  }
  const auto& projection_weight =
      first_native_projection(*snapshot, projection_index);
  if (activations.dtype() != mx::bfloat16) {
    throw std::invalid_argument(
        "paged native bfloat16 activations must use bfloat16");
  }
  if (activations.shape(-1) != projection_weight.shape(-1)) {
    throw std::invalid_argument(
        "paged native bfloat16 activation and weight dimensions differ");
  }
  auto output_shape = selected_indices.shape();
  output_shape.push_back(activations.shape(-2));
  output_shape.push_back(projection_weight.shape(-2));
  std::vector<mx::array> primitive_inputs{activations, selected_indices};
  // Keep native BF16 from disk through multiplication and output. The custom
  // primitive accumulates in float internally but stores the model's declared
  // BF16 activation dtype, matching the non-paged arithmetic contract.
  return mx::array(
      std::move(output_shape),
      mx::bfloat16,
      std::make_shared<PagedGatherNativeBfloat16Matrix>(
          stream,
          snapshot,
          selected_expert_ids,
          projection_index),
      std::move(primitive_inputs));
}

}  // namespace astronomical::paged_expert_execution
