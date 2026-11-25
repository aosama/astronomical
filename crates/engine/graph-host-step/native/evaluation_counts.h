// How often MLX handed work between the caller, the GPU and its CPU stream.
//
// Read from MLX's own evaluator (the `evaluation-counts` patch), because a
// host readback evaluates inside MLX where no caller can count it. The C
// interface exists for the Rust owner in `src/evaluation_counts.rs`.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Totals since the process started; see `mlx::core::EvaluationCounts`.
typedef struct astro_evaluation_counts {
  uint64_t blocking_evaluations;
  uint64_t asynchronous_evaluations;
  uint64_t waits_for_submitted_arrays;
  uint64_t cross_stream_waits;
  uint64_t gpu_primitives;
  uint64_t cpu_primitives;
  uint64_t gpu_command_buffers;
  uint64_t gpu_busy_nanoseconds;
} astro_evaluation_counts;

void astro_evaluation_counts_read(astro_evaluation_counts* counts);

#ifdef __cplusplus
}
#endif
