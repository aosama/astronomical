//! Does splitting a deep contraction into summed 2,048-deep parts inside one
//! Core ML program recover the Neural Engine's full rate?
//!
//! The shape sweep measured 18.6 TFLOP/s at a 2,048-deep contraction and 9.7
//! at 4,096. Dense models contract over 4,096 (hidden size) and 12,288 (the
//! MLP intermediate size of Ornith-1.5-9B), so this decides whether they can
//! use the Neural Engine at all.

use astronomical_neural_engine::core_ml_projection::{ComputePlacement, CoreMlComputeUnits};
use serial_test::serial;

use super::measured_projection::{measure_projection_with_contraction_split, print_evidence};
use super::projection_fixture::ProjectionFixture;

const PREFILL_CHUNK_TOKEN_COUNT: usize = 2048;

fn sweep_contraction_splits(input_channels: usize, output_channels: usize, part_counts: &[usize]) {
    let fixture =
        ProjectionFixture::with_shape(input_channels, output_channels, PREFILL_CHUNK_TOKEN_COUNT);
    for contraction_part_count in part_counts {
        eprintln!(
            "[neural-engine-hello] ── {input_channels} → {output_channels}, contraction in \
             {contraction_part_count} part(s) of {}",
            input_channels / contraction_part_count
        );
        let evidence = measure_projection_with_contraction_split(
            &fixture,
            CoreMlComputeUnits::CpuAndNeuralEngine,
            *contraction_part_count,
        );
        print_evidence(CoreMlComputeUnits::CpuAndNeuralEngine, &fixture, &evidence);
        assert_eq!(evidence.placement.preferred, ComputePlacement::NeuralEngine);
        assert!(evidence.comparison.cosine_similarity >= 0.9999);
    }
}

#[test]
#[serial(neural_engine)]
#[ignore = "runs 4,096-deep projections on the real Neural Engine"]
fn should_measure_a_4096_deep_contraction_split_into_parts() {
    sweep_contraction_splits(4096, 4096, &[1, 2, 4]);
}

#[test]
#[serial(neural_engine)]
#[ignore = "runs 12,288-deep projections on the real Neural Engine"]
fn should_measure_a_12288_deep_contraction_split_into_parts() {
    sweep_contraction_splits(12_288, 4096, &[1, 3, 6]);
}
