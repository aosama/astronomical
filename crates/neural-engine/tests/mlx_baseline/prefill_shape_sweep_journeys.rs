//! Every candidate prefill projection from issue #161, on the Neural Engine
//! alone and on MLX alone, so the split experiment can be planned per shape.
//!
//! The ideal two-engine speedup for a shape is `1 + t_gpu / t_ne`: with the
//! output channels divided in proportion to each engine's speed, both finish
//! together. The mixer output split measured 1.27× against an ideal of 1.43×,
//! so the ideal is an upper bound, not a forecast.

use astronomical_neural_engine::core_ml::CoreMlComputeUnits;
use mlx_c_rust::MlxDtype;
use serial_test::serial;

use crate::support::measured_projection::{measure_projection, print_evidence};
use crate::support::mlx_projection::{initialize_runtime, measure_dense};
use crate::support::projection_fixture::ProjectionFixture;

struct CandidateShape {
    label: &'static str,
    input_channels: usize,
    output_channels: usize,
    token_count: usize,
}

const CANDIDATE_SHAPES: [CandidateShape; 8] = [
    CandidateShape {
        label: "DeltaNet fused input (qkv+z+a+b)",
        input_channels: 2048,
        output_channels: 12_352,
        token_count: 2048,
    },
    CandidateShape {
        label: "attention fused input (q+gate, k, v)",
        input_channels: 2048,
        output_channels: 9216,
        token_count: 2048,
    },
    CandidateShape {
        label: "mixer output (out_proj / o_proj)",
        input_channels: 4096,
        output_channels: 2048,
        token_count: 2048,
    },
    // One half of the mixer output's 4,096-deep contraction. The first sweep
    // measured 18.6 TFLOP/s at a 2,048-deep contraction against 9.6 at
    // 4,096, so two half-depth predictions summed may beat one full one.
    CandidateShape {
        label: "mixer output, half contraction (2,048 of 4,096 inputs)",
        input_channels: 2048,
        output_channels: 2048,
        token_count: 2048,
    },
    CandidateShape {
        label: "post-mixer fused input (shared gate+up, router)",
        input_channels: 2048,
        output_channels: 1281,
        token_count: 2048,
    },
    CandidateShape {
        label: "shared-expert down",
        input_channels: 512,
        output_channels: 2048,
        token_count: 2048,
    },
    CandidateShape {
        label: "mixer output, 1,024-token chunk",
        input_channels: 4096,
        output_channels: 2048,
        token_count: 1024,
    },
    CandidateShape {
        label: "mixer output, 512-token chunk",
        input_channels: 4096,
        output_channels: 2048,
        token_count: 512,
    },
];

#[test]
#[serial(neural_engine)]
#[ignore = "runs every candidate prefill projection on the real Neural Engine and GPU"]
fn should_measure_every_candidate_prefill_shape_on_both_engines() {
    let runtime = initialize_runtime();
    for candidate in &CANDIDATE_SHAPES {
        let fixture = ProjectionFixture::with_shape(
            candidate.input_channels,
            candidate.output_channels,
            candidate.token_count,
        );
        eprintln!(
            "[neural-engine-hello] ── {} ({} → {}, {} tokens, {:.1} GFLOP)",
            candidate.label,
            candidate.input_channels,
            candidate.output_channels,
            candidate.token_count,
            fixture.shape.floating_point_operations() / 1.0e9
        );
        let neural_engine_evidence =
            measure_projection(&fixture, CoreMlComputeUnits::CpuAndNeuralEngine);
        print_evidence(
            CoreMlComputeUnits::CpuAndNeuralEngine,
            &fixture,
            &neural_engine_evidence,
        );
        let (gpu_timing, gpu_cosine) = measure_dense(&runtime, &fixture, MlxDtype::Float16);
        let neural_engine_seconds = neural_engine_evidence.timing.median.as_secs_f64();
        let gpu_seconds = gpu_timing.median.as_secs_f64();
        eprintln!(
            "  → NE/GPU time ratio {:.2}, ideal split: {:.0}% of channels on the NE for {:.2}×",
            neural_engine_seconds / gpu_seconds,
            100.0 * gpu_seconds / (gpu_seconds + neural_engine_seconds),
            1.0 + gpu_seconds / neural_engine_seconds
        );
        assert!(neural_engine_evidence.comparison.cosine_similarity >= 0.9999);
        assert!(gpu_cosine >= 0.9999);
    }
}
