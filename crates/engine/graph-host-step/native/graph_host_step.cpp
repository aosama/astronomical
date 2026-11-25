#include "graph_host_step.h"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mlx/backend/cpu/encoder.h"
#include "mlx/c/error.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/enums.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace {

using mlx::core::array;

// The caller's state and the two functions that use it.
//
// Shared between the primitive and the run it queues, and not owned by the
// primitive alone, because MLX detaches an array from its primitive as soon as
// the evaluator has walked past it. That is before the queued run executes on
// the CPU stream thread, so the run has to keep the state alive itself.
class HostState {
 public:
  HostState(
      astro_graph_host_step_run run,
      astro_graph_host_step_release release,
      void* host_state)
      : run_(run), release_(release), host_state_(host_state) {}

  ~HostState() {
    if (release_ != nullptr) {
      release_(host_state_);
    }
  }

  HostState(const HostState&) = delete;
  HostState& operator=(const HostState&) = delete;

  void run(
      const std::vector<astro_graph_host_step_input>& inputs,
      void* output_bytes,
      size_t output_byte_count) const {
    // MLX's CPU stream thread records a thrown exception as the stream's
    // error, which fails the evaluation that is waiting on this step.
    if (run_(
            host_state_,
            inputs.data(),
            inputs.size(),
            output_bytes,
            output_byte_count) != 0) {
      throw std::runtime_error(
          "[graph_host_step] the host step reported a failure");
    }
  }

 private:
  astro_graph_host_step_run run_;
  astro_graph_host_step_release release_;
  void* host_state_;
};

class GraphHostStep : public mlx::core::Primitive {
 public:
  GraphHostStep(mlx::core::Stream cpu_stream, std::shared_ptr<HostState> host_state)
      : mlx::core::Primitive(cpu_stream), host_state_(std::move(host_state)) {}

  // Called on the thread that evaluates, while it walks the graph: the output
  // is allocated here so the GPU steps after this one can be encoded against
  // its address before it holds anything. The run itself is queued behind
  // whatever the evaluator already queued on the CPU stream, which is the wait
  // for the GPU steps this one reads.
  void eval_cpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& output = outputs[0];
    output.set_data(mlx::core::allocator::malloc(output.nbytes()));

    auto& encoder = mlx::core::cpu::get_command_encoder(stream());
    std::vector<astro_graph_host_step_input> input_views;
    input_views.reserve(inputs.size());
    for (const auto& input : inputs) {
      if (!input.flags().row_contiguous) {
        throw std::invalid_argument(
            "[graph_host_step] an input is not in row-major order");
      }
      encoder.set_input_array(input);
      input_views.push_back({input.data<char>(), input.nbytes()});
    }
    encoder.set_output_array(output);

    encoder.dispatch(
        [host_state = host_state_,
         input_views = std::move(input_views),
         output_bytes = output.data<char>(),
         output_byte_count = output.nbytes()]() {
          host_state->run(input_views, output_bytes, output_byte_count);
        });
  }

  void eval_gpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "[graph_host_step] a host step runs on the CPU stream only");
  }

  const char* name() const override {
    return "AstronomicalGraphHostStep";
  }

 private:
  std::shared_ptr<HostState> host_state_;
};

array append_host_step(
    const astro_graph_host_step_request& request,
    std::shared_ptr<HostState> host_state) {
  std::vector<array> inputs;
  inputs.reserve(request.input_count);
  for (size_t input_index = 0; input_index < request.input_count; ++input_index) {
    // Routes usually reach here as a slice of a wider selection, which is a
    // strided view. `contiguous` shares the buffer of an array already in
    // row-major order and copies only a view, on the stream that made it.
    inputs.push_back(
        mlx::core::contiguous(mlx_array_get_(request.input_arrays[input_index])));
  }
  mlx::core::Shape output_shape;
  for (size_t dimension = 0; dimension < request.output_dimension_count; ++dimension) {
    if (request.output_shape[dimension] < 0) {
      throw std::invalid_argument(
          "[graph_host_step] an output dimension is negative");
    }
    output_shape.push_back(request.output_shape[dimension]);
  }
  return array(
      std::move(output_shape),
      mlx_dtype_to_cpp(request.output_dtype),
      std::make_shared<GraphHostStep>(
          mlx::core::default_stream(mlx::core::Device::cpu),
          std::move(host_state)),
      std::move(inputs));
}

} // namespace

extern "C" int astro_graph_host_step_append(
    const astro_graph_host_step_request* request,
    mlx_array* output_array) {
  // Owned before anything can throw, so every failure path releases the
  // caller's state exactly once.
  auto host_state = std::make_shared<HostState>(
      request->run, request->release, request->host_state);
  try {
    mlx_array_set_(*output_array, append_host_step(*request, std::move(host_state)));
  } catch (const std::exception& failure) {
    mlx_error("%s", failure.what());
    return 1;
  }
  return 0;
}
