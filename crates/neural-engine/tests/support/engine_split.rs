//! One projection split across the Neural Engine and the GPU, started at the
//! same instant, shared by the journeys that time both engines together.

use std::time::{Duration, Instant};

use astronomical_neural_engine::core_ml::{ComputePlacement, CoreMlComputeUnits, CoreMlProjection};
use astronomical_neural_engine::numerical_comparison::compare_values;
use astronomical_neural_engine::timing_summary::TimingSummary;
use astronomical_runtime_integration::MlxRuntime;

use super::measured_projection::{
    MEASURED_CALL_COUNT, MINIMUM_WARM_UP_CALL_COUNT, MINIMUM_WARM_UP_DURATION, report_progress,
};
use super::mlx_projection::MlxProjectionForm;
use super::projection_fixture::ProjectionFixture;
use crate::support::journey_constants::NEURAL_ENGINE_FINISH_TIMEOUT;
use crate::support::mlx_projection::build_projection;
use crate::support::program_loading::load_neural_network;

pub(crate) enum MeasuredEngines {
    GpuAlone,
    NeuralEngineAlone,
    Together,
}

pub(crate) fn time_split(
    runtime: &MlxRuntime,
    gpu_form: &MlxProjectionForm<'_>,
    neural_engine_projection: &mut CoreMlProjection,
    measured_engines: &MeasuredEngines,
    engine_label: &str,
) -> [TimingSummary; 3] {
    let warm_up_started_at = Instant::now();
    let mut warm_up_call_count = 0;
    while warm_up_call_count < MINIMUM_WARM_UP_CALL_COUNT
        || warm_up_started_at.elapsed() < MINIMUM_WARM_UP_DURATION
    {
        run_split_call(
            runtime,
            gpu_form,
            neural_engine_projection,
            measured_engines,
        );
        warm_up_call_count += 1;
    }
    let mut totals = Vec::with_capacity(MEASURED_CALL_COUNT);
    let mut gpu_parts = Vec::with_capacity(MEASURED_CALL_COUNT);
    let mut neural_engine_parts = Vec::with_capacity(MEASURED_CALL_COUNT);
    let measuring_started_at = Instant::now();
    let mut last_progress_at = measuring_started_at;
    for call_index in 0..MEASURED_CALL_COUNT {
        let (total, gpu_part, neural_engine_part) = run_split_call(
            runtime,
            gpu_form,
            neural_engine_projection,
            measured_engines,
        );
        totals.push(total);
        gpu_parts.push(gpu_part);
        neural_engine_parts.push(neural_engine_part);
        if last_progress_at.elapsed() >= Duration::from_secs(5) {
            report_progress(engine_label, call_index + 1, measuring_started_at.elapsed());
            last_progress_at = Instant::now();
        }
    }
    [
        summarize(&totals),
        summarize(&gpu_parts),
        summarize(&neural_engine_parts),
    ]
}

pub(crate) fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

pub(crate) fn assert_slice_matches_reference(
    slice_fixture: &ProjectionFixture,
    candidate: &[f32],
    label: &str,
) {
    let comparison =
        compare_values(&slice_fixture.reference_projected, candidate).expect("same length");
    eprintln!(
        "    {label} slice vs fp32 reference: max abs {:.3e}, cosine {:.8}, non-finite {}",
        comparison.maximum_absolute_error,
        comparison.cosine_similarity,
        comparison.non_finite_candidate_count
    );
    assert_eq!(comparison.non_finite_candidate_count, 0);
    assert!(comparison.cosine_similarity >= 0.9999, "{label} cosine");
}

pub(crate) fn load_neural_engine_slice(
    neural_engine_fixture: &ProjectionFixture,
    contraction_part_count: usize,
    model_directory: &std::path::Path,
) -> CoreMlProjection {
    let mut projection = load_neural_network(
        neural_engine_fixture.shape,
        &neural_engine_fixture.fp16_weights,
        CoreMlComputeUnits::CpuAndNeuralEngine,
        contraction_part_count,
        model_directory,
    )
    .expect("the Neural Engine slice loads");
    assert_eq!(
        projection.placement().preferred,
        ComputePlacement::NeuralEngine
    );
    projection
        .write_activations(&neural_engine_fixture.fp16_activations)
        .expect("the activations fit");
    projection
}

/// One call of the chosen engines; returns (total, GPU part, Neural Engine part).
fn run_split_call(
    runtime: &MlxRuntime,
    gpu_form: &MlxProjectionForm<'_>,
    neural_engine_projection: &mut CoreMlProjection,
    measured_engines: &MeasuredEngines,
) -> (Duration, Duration, Duration) {
    let call_started_at = Instant::now();
    match measured_engines {
        MeasuredEngines::GpuAlone => {
            evaluate_gpu_slice(runtime, gpu_form);
            let gpu_duration = call_started_at.elapsed();
            (gpu_duration, gpu_duration, Duration::ZERO)
        }
        MeasuredEngines::NeuralEngineAlone => {
            let neural_engine_duration = neural_engine_projection
                .predict()
                .expect("the Neural Engine slice runs");
            (
                call_started_at.elapsed(),
                Duration::ZERO,
                neural_engine_duration,
            )
        }
        MeasuredEngines::Together => {
            let pending = neural_engine_projection
                .begin_prediction()
                .expect("the Neural Engine slice starts");
            evaluate_gpu_slice(runtime, gpu_form);
            let gpu_duration = call_started_at.elapsed();
            let neural_engine_duration = pending
                .finish(NEURAL_ENGINE_FINISH_TIMEOUT)
                .expect("the Neural Engine slice finishes");
            (
                call_started_at.elapsed(),
                gpu_duration,
                neural_engine_duration,
            )
        }
    }
}

pub(crate) fn summarize(durations: &[Duration]) -> TimingSummary {
    TimingSummary::from_samples(durations).expect("calls were timed")
}

pub(crate) fn evaluate_gpu_slice(runtime: &MlxRuntime, gpu_form: &MlxProjectionForm<'_>) {
    let projected = build_projection(runtime, gpu_form);
    runtime
        .evaluate_arrays(&[&projected])
        .expect("the GPU slice evaluates");
}
