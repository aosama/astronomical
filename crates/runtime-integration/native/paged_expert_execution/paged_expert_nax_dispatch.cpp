#include "paged_expert_execution_internal.h"

#include <string>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/utils.h"
#include "paged_expert_nax_kernel_source.h"

namespace mx = mlx::core;

namespace astronomical::paged_expert_execution {

// Optional Metal 4 matrix dispatch for large sorted affine routes. Returning
// false is a normal capability decision and lets the caller use its generic
// exact-arithmetic fallback.

bool dispatch_paged_quantized_matrix_nax(
    const PageTableSnapshot& snapshot,
    const std::optional<std::vector<size_t>>& selected_expert_ids,
    int projection_index,
    int group_size,
    int bits,
    const mx::array& activations,
    mx::Dtype scale_dtype,
    mx::Dtype bias_dtype,
    const mx::array& selected_indices,
    mx::array& output,
    int matrix_row_count,
    int output_dimension,
    int input_dimension,
    mx::Stream stream) {
  // NAX is MLX's internal name for the Metal 4 matrix path. Float32 and older
  // graphics processors retain the generic matrix fallback owned by the caller;
  // precision is never changed merely to enter this faster kernel family.
  if (!mx::metal::is_nax_available() || activations.dtype() == mx::float32 ||
      activations.dtype() != scale_dtype || activations.dtype() != bias_dtype) {
    return false;
  }

  constexpr int block_rows = 64;
  constexpr int block_columns = 64;
  constexpr int block_depth = 64;
  constexpr int warp_rows = 2;
  constexpr int warp_columns = 2;
  const bool align_rows = matrix_row_count % block_rows == 0;
  const bool align_columns = output_dimension % block_columns == 0;
  const bool align_depth = input_dimension % block_depth == 0;
  std::string kernel_name;
  mx::concatenate(
      kernel_name,
      "astronomical_paged_gather_qmm_rhs_nax_",
      mx::get_type_string(activations.dtype()),
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_p_",
      projection_index);
  std::string template_definition = kPagedGatherQuantizedMatrixNaxKernelSource;
  template_definition += mx::get_template_definition(
      kernel_name,
      "astronomical_paged_gather_qmm_rhs_nax",
      mx::get_type_string(activations.dtype()),
      group_size,
      bits,
      projection_index,
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

  auto& metal_device = mx::metal::device(stream.device);
  auto* kernel_library = metal_device.get_library(kernel_name, [&]() {
    std::string kernel_source;
    mx::concatenate(
        kernel_source,
        mx::metal::utils(),
        mx::metal::gemm_nax(),
        mx::metal::quantized_utils(),
        mx::metal::quantized_nax(),
        template_definition);
    return kernel_source;
  });
  auto* kernel = metal_device.get_kernel(
      kernel_name,
      kernel_library,
      specialized_kernel_name,
      function_constants);
  // This dispatch reads expert arrays through addresses in the page table. Bind
  // the table normally, then declare every routed backing resource explicitly
  // before encoding the matrix work.
  auto& command_encoder = mx::metal::get_command_encoder(stream);
  command_encoder.set_compute_pipeline_state(kernel);
  command_encoder.set_input_array(snapshot.metal_page_table, 0);
  command_encoder.set_input_array(activations, 1);
  command_encoder.set_input_array(selected_indices, 2);
  command_encoder.set_output_array(output, 3);
  command_encoder.set_bytes(matrix_row_count, 4);
  command_encoder.set_bytes(output_dimension, 5);
  command_encoder.set_bytes(input_dimension, 6);
  auto indirect_projection_resources = routed_quantized_projection_resources(
      snapshot, selected_expert_ids, projection_index);
  command_encoder.register_indirect_input_arrays(indirect_projection_resources);
  command_encoder.dispatch_threadgroups(
      MTL::Size(
          (output_dimension + block_columns - 1) / block_columns,
          (matrix_row_count + block_rows - 1) / block_rows,
          1),
      MTL::Size(32, warp_columns, warp_rows));
  return true;
}

}  // namespace astronomical::paged_expert_execution
