// A step of an MLX graph that runs on the host, between two GPU steps, without
// ending the evaluation.
//
// A streamed model has to look at what the GPU just computed (which experts a
// layer routed to) before it can say what the GPU reads next (where those
// experts are). Done from the caller, that is one evaluation a layer: the
// graph is cut at the router, submitted, waited for, read back, and a new
// graph is built from the answer. This step keeps the look inside the graph.
// MLX's own evaluator orders it after the GPU work it reads and before the
// GPU work that reads it, so a whole reply token is one graph and one
// evaluation however many layers stop to look.
//
// The C interface exists for the Rust owner in `src/host_step.rs`.

#pragma once

#include <stddef.h>

#include "mlx/c/array.h"

#ifdef __cplusplus
extern "C" {
#endif

// One input as the host step sees it: an evaluated array's elements in
// row-major order.
typedef struct astro_graph_host_step_input {
  const void* bytes;
  size_t byte_count;
} astro_graph_host_step_input;

// Fills `output_bytes` from the inputs. Runs on MLX's CPU stream thread, never
// on the thread that built the graph. Returns 0 when the output is written;
// any other value fails the evaluation that reached this step.
typedef int (*astro_graph_host_step_run)(
    void* host_state,
    const astro_graph_host_step_input* inputs,
    size_t input_count,
    void* output_bytes,
    size_t output_byte_count);

// Called exactly once, when neither the graph nor a queued run can reach
// `host_state` any more. May run on MLX's CPU stream thread.
typedef void (*astro_graph_host_step_release)(void* host_state);

typedef struct astro_graph_host_step_request {
  const mlx_array* input_arrays;
  size_t input_count;
  const int* output_shape;
  size_t output_dimension_count;
  mlx_dtype output_dtype;
  astro_graph_host_step_run run;
  astro_graph_host_step_release release;
  void* host_state;
} astro_graph_host_step_request;

// Adds the step to the graph and sets `output_array` to its lazy result. The
// step owns `host_state` from the moment this is called, on failure too: a
// failed call has already released it. Returns 0 on success; on failure the
// description is reported through MLX-C's error handler.
int astro_graph_host_step_append(
    const astro_graph_host_step_request* request,
    mlx_array* output_array);

#ifdef __cplusplus
}
#endif
