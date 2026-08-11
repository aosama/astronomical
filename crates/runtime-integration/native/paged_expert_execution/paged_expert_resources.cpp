#include "paged_expert_execution_internal.h"

#include <stdexcept>

namespace astronomical::paged_expert_execution {

// Resource enumeration is separate from page ownership: snapshots keep buffers
// alive, while this file tells Metal which indirectly addressed buffers the
// current command encoder may read.

std::vector<const mlx::core::array*> routed_quantized_projection_resources(
    const PageTableSnapshot& snapshot,
    const std::optional<std::vector<size_t>>& selected_expert_ids,
    int projection_index) {
  // Graphics-processor addresses embedded in the page table do not tell Metal
  // which buffers a command may access. Return every indirectly referenced
  // packed weight, scale, and bias so the encoder can declare correct resource
  // usage before dispatch.
  std::vector<const mlx::core::array*> projection_resources;
  projection_resources.reserve(
      (selected_expert_ids.has_value()
           ? selected_expert_ids->size()
           : snapshot.resident_expert_count) *
      3);
  const auto append_expert_resources =
      [&](const std::optional<ExpertPageArrays>& expert_page) {
        if (!expert_page.has_value()) {
          throw std::invalid_argument(
              "paged expert route references a nonresident expert");
        }
        const auto& projection = expert_page->projections[projection_index];
        projection_resources.push_back(&projection.packed_weight);
        projection_resources.push_back(&projection.scales);
        projection_resources.push_back(&projection.biases);
      };
  if (selected_expert_ids.has_value()) {
    // Ordinary routes were synchronized on the host, so declaring only selected
    // pages is both exact and cheaper than walking the complete snapshot.
    for (const auto expert_id : *selected_expert_ids) {
      if (expert_id >= snapshot.expert_pages.size()) {
        throw std::invalid_argument(
            "paged expert route resource ID exceeds table capacity");
      }
      append_expert_resources(snapshot.expert_pages[expert_id]);
    }
  } else {
    // Only a fully resident layer omits a bounded resource list.
    // Encoder-lifetime declaration deduplication was measured and retained no
    // serving gain; keep this direct enumeration until end-to-end evidence
    // supports a different resource contract.
    for (const auto& expert_page : snapshot.expert_pages) {
      if (expert_page.has_value()) {
        append_expert_resources(expert_page);
      }
    }
  }
  return projection_resources;
}

}  // namespace astronomical::paged_expert_execution
