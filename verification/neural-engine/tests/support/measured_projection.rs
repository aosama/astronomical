//! Loads the fixture for one set of compute units and gathers the evidence.

use std::path::Path;
use std::time::{Duration, Instant};

use astronomical_neural_engine::core_ml::{CoreMlComputeUnits, CoreMlProjection, LayerPlacement};
use astronomical_neural_engine::documents::projection::MlProgramWeights;
use astronomical_neural_engine::weights::half_precision::f16_bits_slice_to_f32;
use astronomical_neural_engine_research::numerical_comparison::{
    NumericalComparison, compare_values,
};
use astronomical_neural_engine_research::process_memory::{
    MemoryGrowth, MemorySnapshot, mebibytes,
};
use astronomical_neural_engine_research::timing_summary::TimingSummary;

use super::projection_fixture::ProjectionFixture;
use crate::support::program_loading::{load_ml_program, load_neural_network};

// The first predictions pay for program loading and clock ramp-up. Warm-up
// runs for at least three calls and at least 1.2 seconds, the GPU clock-ramp
// window the existing quantized-matmul bench established, and the MLX
// baseline uses the identical rule so neither engine is measured cold.
pub const MINIMUM_WARM_UP_CALL_COUNT: usize = 3;
pub const MINIMUM_WARM_UP_DURATION: Duration = Duration::from_millis(1200);
pub const MEASURED_CALL_COUNT: usize = 20;
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

pub struct ProjectionEvidence {
    pub placement: LayerPlacement,
    pub comparison: NumericalComparison,
    pub timing: TimingSummary,
    pub load_duration: Duration,
    pub memory: ProjectionMemoryEvidence,
}

/// Growth from just before the load. Loading may defer work to the first
/// prediction. The release reading is kept because it is itself a finding:
/// measured on 100-million-weight loads, releasing a projection returned
/// none of its footprint to the process.
pub struct ProjectionMemoryEvidence {
    pub after_load: MemoryGrowth,
    pub after_predictions: MemoryGrowth,
    pub after_release: MemoryGrowth,
}

/// Which Core ML document carries the projection's weights.
#[derive(Clone, Copy)]
pub enum ProjectionModel<'weights> {
    /// The legacy `NeuralNetwork` document with the fixture's fp16 weights.
    NeuralNetworkFp16,
    MlProgram(MlProgramWeights<'weights>),
}

pub fn measure_projection(
    fixture: &ProjectionFixture,
    compute_units: CoreMlComputeUnits,
) -> ProjectionEvidence {
    measure_projection_with_contraction_split(fixture, compute_units, 1)
}

pub fn measure_projection_with_contraction_split(
    fixture: &ProjectionFixture,
    compute_units: CoreMlComputeUnits,
    contraction_part_count: usize,
) -> ProjectionEvidence {
    measure_projection_model(
        fixture,
        compute_units,
        contraction_part_count,
        ProjectionModel::NeuralNetworkFp16,
    )
}

pub fn measure_projection_model(
    fixture: &ProjectionFixture,
    compute_units: CoreMlComputeUnits,
    contraction_part_count: usize,
    projection_model: ProjectionModel<'_>,
) -> ProjectionEvidence {
    let model_directory = tempfile::tempdir().expect("a scratch directory for the model");
    let memory_before_load = MemorySnapshot::capture();
    let load_started_at = Instant::now();
    let mut projection = load_projection_model(
        fixture,
        compute_units,
        contraction_part_count,
        projection_model,
        model_directory.path(),
    );
    let load_duration = load_started_at.elapsed();
    let after_load = MemoryGrowth::between(memory_before_load, MemorySnapshot::capture());
    projection
        .write_activations(&fixture.fp16_activations)
        .expect("the activations fit the input surface");
    let timing = time_predictions(&mut projection, compute_units);
    let after_predictions = MemoryGrowth::between(memory_before_load, MemorySnapshot::capture());
    let comparison = compare_with_reference(fixture, &projection);
    let placement = projection.placement();
    drop(projection);
    ProjectionEvidence {
        placement,
        comparison,
        timing,
        load_duration,
        memory: ProjectionMemoryEvidence {
            after_load,
            after_predictions,
            after_release: MemoryGrowth::between(memory_before_load, MemorySnapshot::capture()),
        },
    }
}

fn compare_with_reference(
    fixture: &ProjectionFixture,
    projection: &CoreMlProjection,
) -> NumericalComparison {
    let candidate_projected = f16_bits_slice_to_f32(
        &projection
            .read_projected()
            .expect("the output surface is readable"),
    );
    compare_values(&fixture.reference_projected, &candidate_projected)
        .expect("reference and candidate have the same length")
}

fn load_projection_model(
    fixture: &ProjectionFixture,
    compute_units: CoreMlComputeUnits,
    contraction_part_count: usize,
    projection_model: ProjectionModel<'_>,
    model_directory: &Path,
) -> CoreMlProjection {
    match projection_model {
        ProjectionModel::NeuralNetworkFp16 => load_neural_network(
            fixture.shape,
            &fixture.fp16_weights,
            compute_units,
            contraction_part_count,
            model_directory,
        ),
        ProjectionModel::MlProgram(weights) => load_ml_program(
            fixture.shape,
            weights,
            compute_units,
            contraction_part_count,
            model_directory,
        ),
    }
    .expect("Core ML loads the projection")
}

fn time_predictions(
    projection: &mut CoreMlProjection,
    compute_units: CoreMlComputeUnits,
) -> TimingSummary {
    let warm_up_started_at = Instant::now();
    let mut warm_up_call_count = 0;
    while warm_up_call_count < MINIMUM_WARM_UP_CALL_COUNT
        || warm_up_started_at.elapsed() < MINIMUM_WARM_UP_DURATION
    {
        projection.predict().expect("the warm-up prediction runs");
        warm_up_call_count += 1;
    }
    let mut prediction_durations = Vec::with_capacity(MEASURED_CALL_COUNT);
    let measuring_started_at = Instant::now();
    let mut last_progress_at = measuring_started_at;
    for prediction_index in 0..MEASURED_CALL_COUNT {
        prediction_durations.push(projection.predict().expect("the measured prediction runs"));
        if last_progress_at.elapsed() >= PROGRESS_INTERVAL {
            report_progress(
                &format!("{compute_units:?}"),
                prediction_index + 1,
                measuring_started_at.elapsed(),
            );
            last_progress_at = Instant::now();
        }
    }
    TimingSummary::from_samples(&prediction_durations).expect("predictions were timed")
}

pub fn report_progress(engine_label: &str, completed_count: usize, elapsed: Duration) {
    let seconds_per_call = elapsed.as_secs_f64() / completed_count as f64;
    let remaining_seconds = seconds_per_call * (MEASURED_CALL_COUNT - completed_count) as f64;
    eprintln!(
        "[neural-engine-journey] {engine_label}: {completed_count}/{MEASURED_CALL_COUNT} \
         calls ({:.0}%), ETA {remaining_seconds:.1} s",
        100.0 * completed_count as f64 / MEASURED_CALL_COUNT as f64
    );
}

pub fn print_evidence(
    compute_units: CoreMlComputeUnits,
    fixture: &ProjectionFixture,
    evidence: &ProjectionEvidence,
) {
    let timing = &evidence.timing;
    let comparison = &evidence.comparison;
    eprintln!(
        "[neural-engine-journey] {compute_units:?}\n  placement: preferred {:?} \
         (supported: neural engine {}, gpu {}, cpu {})\n  load+compile: {:.0} ms\n  \
         prediction: median {:.2} ms, p90 {:.2} ms, min {:.2} ms, max {:.2} ms over {} → {:.2} TFLOP/s\n  \
         vs fp32 reference: max abs {:.3e} ({:.2e} of reference RMS), mean abs {:.3e}, cosine {:.8}, non-finite {}",
        evidence.placement.preferred,
        evidence.placement.neural_engine_supported,
        evidence.placement.gpu_supported,
        evidence.placement.cpu_supported,
        evidence.load_duration.as_secs_f64() * 1000.0,
        timing.median.as_secs_f64() * 1000.0,
        timing.ninetieth_percentile.as_secs_f64() * 1000.0,
        timing.minimum.as_secs_f64() * 1000.0,
        timing.maximum.as_secs_f64() * 1000.0,
        timing.sample_count,
        timing.median_teraflops(fixture.shape.floating_point_operations()),
        comparison.maximum_absolute_error,
        comparison.maximum_error_relative_to_reference_rms,
        comparison.mean_absolute_error,
        comparison.cosine_similarity,
        comparison.non_finite_candidate_count,
    );
    print_memory_evidence(&evidence.memory);
}

fn print_memory_evidence(memory: &ProjectionMemoryEvidence) {
    eprintln!(
        "  process footprint growth: {:+.0} MiB after load, {:+.0} MiB after predictions, \
         {:+.0} MiB after release\n  system wired growth: {:+.0} MiB after load, {:+.0} MiB \
         after predictions, {:+.0} MiB after release",
        mebibytes(memory.after_load.process_physical_footprint_bytes),
        mebibytes(memory.after_predictions.process_physical_footprint_bytes),
        mebibytes(memory.after_release.process_physical_footprint_bytes),
        mebibytes(memory.after_load.system_wired_bytes),
        mebibytes(memory.after_predictions.system_wired_bytes),
        mebibytes(memory.after_release.system_wired_bytes),
    );
}
