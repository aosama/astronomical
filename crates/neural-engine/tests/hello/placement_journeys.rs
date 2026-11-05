//! One journey per allowed-device set, so each placement verdict and timing
//! is recorded on its own.

use astronomical_neural_engine::core_ml::compute_device_report::report_compute_devices;
use astronomical_neural_engine::core_ml::{ComputePlacement, CoreMlComputeUnits};
use serial_test::serial;

use crate::support::journey_constants::MINIMUM_COSINE_SIMILARITY;
use crate::support::measured_projection::{ProjectionEvidence, measure_projection, print_evidence};
use crate::support::projection_fixture::ProjectionFixture;

fn assert_numerically_sound(evidence: &ProjectionEvidence) {
    assert_eq!(evidence.comparison.non_finite_candidate_count, 0);
    assert!(
        evidence.comparison.cosine_similarity >= MINIMUM_COSINE_SIMILARITY,
        "cosine similarity {} is below {MINIMUM_COSINE_SIMILARITY}",
        evidence.comparison.cosine_similarity
    );
}

fn measure_and_print(compute_units: CoreMlComputeUnits) -> ProjectionEvidence {
    let fixture = ProjectionFixture::mixer_output_projection();
    let evidence = measure_projection(&fixture, compute_units);
    print_evidence(compute_units, &fixture, &evidence);
    evidence
}

#[test]
#[serial(neural_engine)]
#[ignore = "talks to the real Neural Engine; run through cargo test-neural-engine-hello"]
fn should_see_a_neural_engine_through_core_ml() {
    let report = report_compute_devices();

    eprintln!("[neural-engine-hello] compute devices: {report:?}");
    assert!(
        report.has_neural_engine(),
        "Core ML reports no Neural Engine: {report:?}"
    );
}

#[test]
#[serial(neural_engine)]
#[ignore = "runs a 34-GFLOP projection on the real Neural Engine"]
fn should_place_the_mixer_output_projection_on_the_neural_engine() {
    let evidence = measure_and_print(CoreMlComputeUnits::CpuAndNeuralEngine);

    assert_eq!(evidence.placement.preferred, ComputePlacement::NeuralEngine);
    assert_numerically_sound(&evidence);
}

#[test]
#[serial(neural_engine)]
#[ignore = "runs a 34-GFLOP projection on the real CPU as the negative control"]
fn should_keep_the_projection_on_the_cpu_when_the_neural_engine_is_excluded() {
    let evidence = measure_and_print(CoreMlComputeUnits::CpuOnly);

    // Only the preferred device reflects the allowed compute units: the
    // compute plan's supported-device list names every device the layer
    // could run on, and measured on the first run it still listed the
    // Neural Engine and GPU under CPU-only compute units.
    assert_eq!(evidence.placement.preferred, ComputePlacement::Cpu);
    assert_numerically_sound(&evidence);
}

#[test]
#[serial(neural_engine)]
#[ignore = "runs a 34-GFLOP projection on the real GPU through Core ML"]
fn should_place_the_projection_on_the_gpu_when_the_neural_engine_is_excluded() {
    let evidence = measure_and_print(CoreMlComputeUnits::CpuAndGpu);

    assert_eq!(evidence.placement.preferred, ComputePlacement::Gpu);
    assert_numerically_sound(&evidence);
}
