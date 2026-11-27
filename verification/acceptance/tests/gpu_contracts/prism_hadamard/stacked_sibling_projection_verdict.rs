//! Whether one 2-bit matmul over stacked sibling projections beats one
//! matmul each, at a folded pack's decode shape (#2653).
//!
//! Matrix multiplies are 90% of a reply token of a folded pack on the shared
//! core (`shared_core_reply_token_profile`); its rotations are 2%. The
//! folded engine stacks the MLP's gate and up projections into one launch,
//! where MLX's 2-bit matrix-vector kernel streams weights faster than in two
//! launches of half the rows. This compares the two on layer 0's real gate
//! and up weights and one real activation row, and the profiler checks that
//! both give the same outputs.
//!
//! `cargo measure-prism-stacked-sibling-projections --configuration PATH`,
//! where the configuration names a folded pack. Under a minute.

use mlx_bridge::MlxArray;
use mlx_performance_profiler::{MeasuredJob, ReportLocation, Verdict, compare_jobs};
use model_serving::qwen3_5_core::model_math::decoder_layer_weights::{
    Qwen3_5AffineWeights, Qwen3_5DecoderFeedForwardWeights,
};
use runtime_integration::MlxRuntime;
use serial_test::serial;

use super::shared_core::loaded_shared_core;
use super::support::pack_directory;

const REPORT_SUBDIRECTORY: &str = "prism-stacked-sibling-projections";

/// One affine-packed projection's arrays and packing.
struct PackedProjection {
    packed_weight: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
    group_size: i32,
    bits: i32,
}

impl PackedProjection {
    fn of(affine_weights: &Qwen3_5AffineWeights) -> Self {
        let Qwen3_5AffineWeights::Quantized {
            packed_weight,
            quantization_scales,
            quantization_biases,
            quantization_group_size,
            quantization_bits,
        } = affine_weights
        else {
            panic!("a folded pack's MLP projections are affine-packed");
        };
        Self {
            packed_weight: packed_weight.retain().expect("the weight retains"),
            scales: quantization_scales.retain().expect("the scales retain"),
            biases: quantization_biases.retain().expect("the biases retain"),
            group_size: *quantization_group_size as i32,
            bits: *quantization_bits as i32,
        }
    }

    fn multiplied(&self, runtime: &MlxRuntime, activations: &MlxArray) -> Result<MlxArray, String> {
        runtime
            .quantized_matmul_affine_on_platform(
                activations,
                &self.packed_weight,
                &self.scales,
                &self.biases,
                true,
                self.group_size,
                self.bits,
            )
            .map_err(|matmul_error| matmul_error.to_string())
    }

    /// Both projections' rows in one packed matrix, gate rows first.
    fn stacked(runtime: &MlxRuntime, first: &Self, second: &Self) -> Self {
        let stacked_along_rows = |first_part: &MlxArray, second_part: &MlxArray| {
            let stacked = runtime
                .concatenate_axis(&[first_part, second_part], 0)
                .expect("the rows stack");
            runtime
                .evaluate_arrays(&[&stacked])
                .expect("the stack evaluates");
            stacked
        };
        Self {
            packed_weight: stacked_along_rows(&first.packed_weight, &second.packed_weight),
            scales: stacked_along_rows(&first.scales, &second.scales),
            biases: stacked_along_rows(&first.biases, &second.biases),
            group_size: first.group_size,
            bits: first.bits,
        }
    }
}

/// The gate and up projections as two launches.
struct SeparateLaunches<'job> {
    runtime: &'job MlxRuntime,
    gate: &'job PackedProjection,
    up: &'job PackedProjection,
    activations: &'job MlxArray,
}

impl MeasuredJob for SeparateLaunches<'_> {
    fn name(&self) -> &str {
        "separate_gate_and_up"
    }

    fn fresh_outputs(&mut self) -> Result<Vec<MlxArray>, String> {
        Ok(vec![
            self.gate.multiplied(self.runtime, self.activations)?,
            self.up.multiplied(self.runtime, self.activations)?,
        ])
    }
}

/// The gate and up projections as one stacked launch, split after.
struct StackedLaunch<'job> {
    runtime: &'job MlxRuntime,
    stacked: &'job PackedProjection,
    gate_row_count: i32,
    activations: &'job MlxArray,
}

impl MeasuredJob for StackedLaunch<'_> {
    fn name(&self) -> &str {
        "stacked_gate_and_up"
    }

    fn fresh_outputs(&mut self) -> Result<Vec<MlxArray>, String> {
        let both = self.stacked.multiplied(self.runtime, self.activations)?;
        let total_row_count = both.shape()[2];
        let part = |first_row: i32, end_row: i32| {
            self.runtime
                .slice(&both, &[0, 0, first_row], &[1, 1, end_row], &[1, 1, 1])
                .map_err(|slice_error| slice_error.to_string())
        };
        Ok(vec![
            part(0, self.gate_row_count)?,
            part(self.gate_row_count, total_row_count)?,
        ])
    }
}

#[test]
#[serial]
#[ignore = "loads the real folded pack and compares 2-bit matmuls on the graphics processor"]
fn should_compare_stacked_sibling_projections_with_one_launch_each() {
    let shared_core = loaded_shared_core(&pack_directory());
    let runtime = shared_core.runtime();
    let Qwen3_5DecoderFeedForwardWeights::Dense(dense_mlp_weights) =
        &shared_core.weights.decoder_layer_weights[0].mlp_weights
    else {
        panic!("a folded pack's layers are dense");
    };
    let stored = |projection: &model_serving::qwen3_5_core::model_math::dense_projection::Qwen3_5DenseProjection| {
        PackedProjection::of(projection.stored_affine_weights().expect("the projection is stored packed"))
    };
    let gate = stored(&dense_mlp_weights.gate_projection);
    let up = stored(&dense_mlp_weights.up_projection);
    let stacked = PackedProjection::stacked(runtime, &gate, &up);
    let hidden_size = shared_core.config.hidden_size() as i32;
    let activations = one_real_activation_row(&shared_core, hidden_size);
    let report_directory = std::env::temp_dir().join(REPORT_SUBDIRECTORY);
    std::fs::create_dir_all(&report_directory).expect("the report folder is writable");
    let comparison = compare_jobs(
        &mut SeparateLaunches {
            runtime,
            gate: &gate,
            up: &up,
            activations: &activations,
        },
        &mut StackedLaunch {
            runtime,
            stacked: &stacked,
            gate_row_count: gate.packed_weight.shape()[0],
            activations: &activations,
        },
        ReportLocation {
            title: "Gate and up: one launch each against one stacked launch",
            directory: &report_directory,
            file_base: "stacked_gate_and_up",
        },
    )
    .expect("the profiler compares the launches");
    println!(
        "[prism-stacked-sibling-projections] separate_us={:.1} stacked_us={:.1} difference={:+.1}% verdict={:?} largest_output_difference={}",
        comparison.gpu_span.baseline_median_microseconds,
        comparison.gpu_span.candidate_median_microseconds,
        100.0 * comparison.gpu_span.relative_difference,
        comparison.gpu_span.verdict,
        comparison.largest_output_difference,
    );
    assert_eq!(
        comparison.largest_output_difference, 0.0,
        "stacking must not change a single output value"
    );
    assert_ne!(comparison.gpu_span.verdict, Verdict::CandidateSlower);
}

/// A real decode activation: the embedding row of a Shakespeare token,
/// rotated as the gate and up projections read it.
fn one_real_activation_row(
    shared_core: &model_serving::Qwen3_5TextModel,
    hidden_size: i32,
) -> MlxArray {
    let runtime = shared_core.runtime();
    let token_ids = runtime
        .array_from_i32(&[1_000], &[1, 1])
        .expect("the token array builds");
    let embedded = shared_core
        .embedding_lookup(&token_ids)
        .expect("the embedding row is read");
    let row = runtime
        .reshape(&embedded, &[1, 1, hidden_size])
        .expect("the row reshapes");
    runtime.evaluate_arrays(&[&row]).expect("the row evaluates");
    row
}
