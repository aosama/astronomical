#include "evaluation_counts.h"

#include "mlx/transforms.h"

extern "C" void astro_evaluation_counts_read(astro_evaluation_counts* counts) {
  const mlx::core::EvaluationCounts mlx_counts = mlx::core::evaluation_counts();
  counts->blocking_evaluations = mlx_counts.blocking_evaluations;
  counts->asynchronous_evaluations = mlx_counts.asynchronous_evaluations;
  counts->waits_for_submitted_arrays = mlx_counts.waits_for_submitted_arrays;
  counts->cross_stream_waits = mlx_counts.cross_stream_waits;
}
