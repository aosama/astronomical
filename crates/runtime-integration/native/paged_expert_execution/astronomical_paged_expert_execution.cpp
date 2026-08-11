#include "paged_expert_execution_internal.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/common/broadcasting.h"
#include "mlx/backend/common/compiled.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/dtype.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#include "paged_expert_kernel_source.h"
#include "paged_expert_mixed_dtype_kernel_support.h"

namespace mx = mlx::core;

namespace astronomical::paged_expert_execution {

// This translation unit owns lazy affine projection primitives and generic
// Metal dispatch. Snapshot publication, NAX dispatch, and resource enumeration
// remain separate so each boundary has one reason to change.

mx::array row_contiguous_array(
    const mx::array& source_array,
    const mx::Stream& stream) {
  if (source_array.flags().row_contiguous) {
    return source_array;
  }
  auto contiguous_array = mx::contiguous_copy_gpu(source_array, stream);
  // The command encoder, rather than this stack frame, owns the temporary until
  // the encoded product has finished reading it.
  mx::metal::get_command_encoder(stream).add_temporary(contiguous_array);
  return contiguous_array;
}

const QuantizedProjectionArrays& first_projection(
    const PageTableSnapshot& snapshot,
    int projection_index) {
  for (const auto& expert_page : snapshot.expert_pages) {
    if (expert_page.has_value()) {
      return expert_page->projections[projection_index];
    }
  }
  throw std::invalid_argument("paged expert snapshot has no resident experts");
}

class PagedGatherQuantizedMatrix final : public mx::UnaryPrimitive {
 public:
  PagedGatherQuantizedMatrix(
      mx::Stream stream,
      std::shared_ptr<const PageTableSnapshot> snapshot,
      std::optional<std::vector<size_t>> selected_expert_ids,
      int projection_index,
      int group_size,
      int bits,
      bool sorted_indices)
      : UnaryPrimitive(stream),
        // shared_ptr capture is the lifetime barrier between mutable cache
        // policy and this lazily evaluated primitive.
        snapshot_(std::move(snapshot)),
        selected_expert_ids_(std::move(selected_expert_ids)),
        projection_index_(projection_index),
        group_size_(group_size),
        bits_(bits),
        sorted_indices_(sorted_indices) {}

  void eval_cpu(
      const std::vector<mx::array>&,
      mx::array&) override {
    throw std::runtime_error(
        "paged expert quantized matrix multiplication requires Metal");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      mx::array& output) override;

  const char* name() const override {
    return "AstronomicalPagedGatherQuantizedMatrix";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto* paged_other =
        dynamic_cast<const PagedGatherQuantizedMatrix*>(&other);
    // Snapshot identity is part of primitive equivalence because two
    // generations can map the same expert IDs to different page owners.
    return paged_other != nullptr &&
        paged_other->snapshot_.get() == snapshot_.get() &&
        paged_other->selected_expert_ids_ == selected_expert_ids_ &&
        paged_other->projection_index_ == projection_index_ &&
        paged_other->group_size_ == group_size_ &&
        paged_other->bits_ == bits_ &&
        paged_other->sorted_indices_ == sorted_indices_;
  }

 private:
  std::shared_ptr<const PageTableSnapshot> snapshot_;
  std::optional<std::vector<size_t>> selected_expert_ids_;
  int projection_index_;
  int group_size_;
  int bits_;
  bool sorted_indices_;
};

void PagedGatherQuantizedMatrix::eval_gpu(
    const std::vector<mx::array>& inputs,
    mx::array& output) {
  auto& stream = this->stream();
  auto& metal_device = mx::metal::device(stream.device);
  auto activations = row_contiguous_array(inputs[0], stream);
  auto selected_indices = row_contiguous_array(inputs[1], stream);
  const int input_dimension = activations.shape(-1);
  const int output_dimension = output.shape(-1);
  const int routed_assignment_count = selected_indices.size();
  const int activation_row_count = activations.size() / input_dimension;
  const auto& projection = first_projection(*snapshot_, projection_index_);
  const auto scale_dtype = projection.scales.dtype();
  const auto bias_dtype = projection.biases.dtype();
  if (activation_row_count <= 0 ||
      routed_assignment_count % activation_row_count != 0) {
    throw std::invalid_argument(
        "paged expert assignments do not map evenly to activation rows");
  }
  output.set_data(mx::allocator::malloc(output.nbytes()));

  // Decode and small routes use one-row gathered matrix-vector work. Large,
  // sorted routes switch to matrix tiles only when each resident expert has
  // enough contiguous assignments to amortize tile setup.
  const bool should_use_sorted_matrix_kernel =
      sorted_indices_ && routed_assignment_count >= 16 &&
      routed_assignment_count /
              static_cast<int>(snapshot_->resident_expert_count) >=
          4;
  if (!should_use_sorted_matrix_kernel) {
    if (activations.shape(-2) != 1) {
      throw std::invalid_argument(
          "paged expert unsorted execution currently requires one row per routed assignment");
    }
    constexpr int output_columns_per_simdgroup = 4;
    constexpr int simdgroups_per_threadgroup = 2;
    constexpr int output_columns_per_threadgroup =
        output_columns_per_simdgroup * simdgroups_per_threadgroup;
    const bool fast = output_dimension % 8 == 0 && input_dimension % 512 == 0;
    std::string vector_kernel_name;
    mx::concatenate(
        vector_kernel_name,
        "astronomical_paged_gather_qmv_",
        mx::get_type_string(activations.dtype()),
        "_scales_",
        mx::get_type_string(scale_dtype),
        "_biases_",
        mx::get_type_string(bias_dtype),
        "_gs_",
        group_size_,
        "_b_",
        bits_,
        "_p_",
        projection_index_,
        fast ? "_fast" : "_regular");
    std::string vector_template_definition = kPagedGatherQuantizedMatrixKernelSource;
    vector_template_definition += mx::get_template_definition(
        vector_kernel_name,
        "astronomical_paged_gather_qmv",
        mx::get_type_string(activations.dtype()),
        mx::get_type_string(scale_dtype),
        mx::get_type_string(bias_dtype),
        group_size_,
        bits_,
        projection_index_,
        fast);
    auto* vector_kernel_library = metal_device.get_library(vector_kernel_name, [&]() {
      std::string kernel_source;
      mx::concatenate(
          kernel_source,
          mx::metal::utils(),
          mx::metal::gemm(),
          mx::metal::quantized_utils(),
          mx::metal::quantized(),
          kPagedExpertMixedDtypeKernelSupportSource,
          vector_template_definition);
      return kernel_source;
    });
    auto* vector_kernel = metal_device.get_kernel(
        vector_kernel_name, vector_kernel_library, vector_kernel_name);
    auto& command_encoder = mx::metal::get_command_encoder(stream);
    command_encoder.set_compute_pipeline_state(vector_kernel);
    command_encoder.set_input_array(snapshot_->metal_page_table, 0);
    command_encoder.set_input_array(activations, 1);
    command_encoder.set_input_array(selected_indices, 2);
    command_encoder.set_output_array(output, 3);
    command_encoder.set_bytes(input_dimension, 4);
    command_encoder.set_bytes(output_dimension, 5);
    const int assignments_per_activation_row =
        routed_assignment_count / activation_row_count;
    command_encoder.set_bytes(assignments_per_activation_row, 6);
    // The page table contains addresses, but Metal hazard tracking still needs
    // explicit useResource declarations for every reachable backing buffer.
    auto indirect_projection_resources =
        routed_quantized_projection_resources(
            *snapshot_, selected_expert_ids_, projection_index_);
    command_encoder.register_indirect_input_arrays(indirect_projection_resources);
    command_encoder.dispatch_threadgroups(
        MTL::Size(
            1,
            (output_dimension + output_columns_per_threadgroup - 1) /
                output_columns_per_threadgroup,
            routed_assignment_count),
        MTL::Size(32, simdgroups_per_threadgroup, 1));
    return;
  }

  if (activation_row_count != routed_assignment_count) {
    auto broadcast_shape = selected_indices.shape();
    broadcast_shape.push_back(activations.shape(-2));
    broadcast_shape.push_back(input_dimension);
    mx::array broadcast_activations(
        std::move(broadcast_shape), activations.dtype(), nullptr, {});
    mx::broadcast(activations, broadcast_activations);
    activations = row_contiguous_array(broadcast_activations, stream);
  }
  const int matrix_row_count = activations.size() / input_dimension;

  if (dispatch_paged_quantized_matrix_nax(
          *snapshot_,
          selected_expert_ids_,
          projection_index_,
          group_size_,
          bits_,
          activations,
          scale_dtype,
          bias_dtype,
          selected_indices,
          output,
          matrix_row_count,
          output_dimension,
          input_dimension,
          stream)) {
    return;
  }

  constexpr int block_rows = 16;
  constexpr int block_columns = 32;
  constexpr int block_depth = 32;
  constexpr int warp_rows = 1;
  constexpr int warp_columns = 2;
  const bool align_rows = matrix_row_count % block_rows == 0;
  const bool align_columns = output_dimension % block_columns == 0;
  const bool align_depth = input_dimension % block_depth == 0;
  std::string kernel_name;
  mx::concatenate(
      kernel_name,
      "astronomical_paged_gather_qmm_rhs_",
      mx::get_type_string(activations.dtype()),
      "_scales_",
      mx::get_type_string(scale_dtype),
      "_biases_",
      mx::get_type_string(bias_dtype),
      "_gs_",
      group_size_,
      "_b_",
      bits_,
      "_p_",
      projection_index_);
  std::string template_definition = kPagedGatherQuantizedMatrixKernelSource;
  template_definition += mx::get_template_definition(
      kernel_name,
      "astronomical_paged_gather_qmm_rhs",
      mx::get_type_string(activations.dtype()),
      mx::get_type_string(scale_dtype),
      mx::get_type_string(bias_dtype),
      group_size_,
      bits_,
      projection_index_,
      block_rows,
      block_columns,
      block_depth,
      warp_rows,
      warp_columns,
      true);
  mx::metal::MTLFCList function_constants = {
      {&align_rows, MTL::DataType::DataTypeBool, 200},
      {&align_columns, MTL::DataType::DataTypeBool, 201},
      {&align_depth, MTL::DataType::DataTypeBool, 202},
  };
  std::string specialized_kernel_name;
  mx::concatenate(
      specialized_kernel_name,
      kernel_name,
      "_m_",
      align_rows,
      "_n_",
      align_columns,
      "_k_",
      align_depth);
  auto* kernel_library = metal_device.get_library(kernel_name, [&]() {
    std::string kernel_source;
    mx::concatenate(
        kernel_source,
        mx::metal::utils(),
        mx::metal::gemm(),
        mx::metal::quantized_utils(),
        mx::metal::quantized(),
        kPagedExpertMixedDtypeKernelSupportSource,
        template_definition);
    return kernel_source;
  });
  auto* kernel = metal_device.get_kernel(
      kernel_name,
      kernel_library,
      specialized_kernel_name,
      function_constants);

  auto& command_encoder = mx::metal::get_command_encoder(stream);
  command_encoder.set_compute_pipeline_state(kernel);
  command_encoder.set_input_array(snapshot_->metal_page_table, 0);
  command_encoder.set_input_array(activations, 1);
  command_encoder.set_input_array(selected_indices, 2);
  command_encoder.set_output_array(output, 3);
  command_encoder.set_bytes(matrix_row_count, 4);
  command_encoder.set_bytes(output_dimension, 5);
  command_encoder.set_bytes(input_dimension, 6);
  auto indirect_projection_resources = routed_quantized_projection_resources(
      *snapshot_, selected_expert_ids_, projection_index_);
  command_encoder.register_indirect_input_arrays(indirect_projection_resources);
  command_encoder.dispatch_threadgroups(
      MTL::Size(
          (output_dimension + block_columns - 1) / block_columns,
          (matrix_row_count + block_rows - 1) / block_rows,
          1),
      MTL::Size(32, warp_columns, warp_rows));
}

mx::array build_paged_quantized_product(
    const std::shared_ptr<const PageTableSnapshot>& snapshot,
    const std::optional<std::vector<size_t>>& selected_expert_ids,
    int projection_index,
    const mx::array& activations,
    const mx::array& selected_indices,
    bool sorted_indices,
    mx::Stream stream) {
  if (snapshot->storage_mode != ExpertPageStorageMode::QuantizedAffine) {
    throw std::invalid_argument(
        "paged quantized product requires an affine page-table snapshot");
  }
  const auto& projection = first_projection(*snapshot, projection_index);
  if (!mx::issubdtype(activations.dtype(), mx::floating) ||
      !mx::issubdtype(projection.scales.dtype(), mx::floating) ||
      !mx::issubdtype(projection.biases.dtype(), mx::floating)) {
    throw std::invalid_argument(
        "paged expert activations, scales, and biases must be floating types");
  }
  const auto affine_parameter_dtype =
      mx::promote_types(projection.scales.dtype(), projection.biases.dtype());
  const auto output_dtype =
      mx::promote_types(activations.dtype(), affine_parameter_dtype);
  auto promoted_activations = activations.dtype() == output_dtype
      ? activations
      : mx::astype(activations, output_dtype, stream);
  const int output_dimension = projection.packed_weight.shape(-2);
  auto output_shape = selected_indices.shape();
  output_shape.push_back(activations.shape(-2));
  output_shape.push_back(output_dimension);
  std::vector<mx::array> primitive_inputs{
      std::move(promoted_activations), selected_indices};
  // Construct a lazy MLX array. Evaluation may occur after cache eviction, so
  // the primitive captures the immutable snapshot rather than consulting live
  // cache state again.
  return mx::array(
      std::move(output_shape),
      output_dtype,
      std::make_shared<PagedGatherQuantizedMatrix>(
          stream,
          snapshot,
          selected_expert_ids,
          projection_index,
          projection.group_size,
          projection.bits,
          sorted_indices),
      std::move(primitive_inputs));
}

}  // namespace astronomical::paged_expert_execution
