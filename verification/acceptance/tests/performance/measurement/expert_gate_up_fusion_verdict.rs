//! Whether fusing a routed expert's gate and up projections into one
//! gathered multiply is faster than two, with a verdict (issue #2585).
//!
//! The removed in-memory engine concatenated each expert layer's gate and up
//! weights at load and ran one gathered projection where the engine runs two. Before the engine's `astro_expert_cache` tables are laid
//! out the fused way, this prices what the fusion buys at the kernel: the
//! two ways as two jobs of the MLX performance profiler, twelve rounds with
//! the order flipped every round, "faster" or "slower" only when the
//! interval for the median difference excludes zero.
//!
//! The shapes are a 35B-A3B expert layer's: 256 experts, 2,048 in, 512 out
//! for gate and for up, 4-bit in groups of 64. Four distinct layers a job,
//! so a pass reads DRAM as a forward does. Cells are the assignments a
//! forward gathers: one reply token's 8, a four-row draft round's 32, and a
//! 2,048-token prompt chunk's 16,384. Asserts nothing; prints each cell's
//! verdict and writes its comparison page and JSON.
//!
//! Read on an M5 Pro: at a prompt chunk the fused multiply's GPU time is 25%
//! shorter (29.9 against 40.0 ms over four layers), because it reads the
//! chunk's activations once; at a reply token and a draft round the GPU
//! time is level and only the host's graph building is shorter.
//!
//! Run with `cargo measure-expert-gate-up-fusion-verdict`. No model; about a
//! minute.

use std::path::Path;

use mlx_bridge::{MlxArray, MlxDtype};
use mlx_performance_profiler::{JobComparison, MeasuredJob, ReportLocation, compare_jobs};
use runtime_integration::MlxRuntime;
use serial_test::serial;

use super::verdict_text::metric_text;

use super::k_row_verify_cost::initialize_runtime;

const EXPERT_COUNT: i32 = 256;
const EXPERTS_A_TOKEN: i32 = 8;
const INPUT_WIDTH: i32 = 2_048;
const PROJECTION_WIDTH: i32 = 512;
const QUANTIZATION_BITS: i32 = 4;
const QUANTIZATION_GROUP_SIZE: i32 = 64;
const LAYER_COUNT: usize = 4;
/// Tokens a forward carries: a reply token, a draft round, a prompt chunk.
const SERVED_TOKEN_COUNTS: [i32; 3] = [1, 4, 2_048];
const REPORT_SUBDIRECTORY: &str = "expert-gate-up-fusion-verdict";

/// One quantized stack of experts, `[experts, out, in]`: its packed
/// weights, scales and biases, as MLX's quantizer returns them.
type QuantizedExperts = (MlxArray, MlxArray, MlxArray);

/// One layer's gate and up as the engine holds them and as the
/// removed in-memory engine held them.
struct ExpertLayer {
    gate: QuantizedExperts,
    up: QuantizedExperts,
    fused_gate_up: QuantizedExperts,
}

#[derive(Clone, Copy)]
enum GateUpWay {
    TwoProjections,
    OneFusedProjection,
}

/// Every layer's gate and up outputs for one forward's assignments.
struct GateUpPass<'fixture> {
    name: &'static str,
    runtime: &'fixture MlxRuntime,
    gate_up_way: GateUpWay,
    expert_layers: &'fixture [ExpertLayer],
    activations: &'fixture MlxArray,
    sorted_expert_ids: &'fixture MlxArray,
}

impl MeasuredJob for GateUpPass<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn fresh_outputs(&mut self) -> Result<Vec<MlxArray>, String> {
        let mut gate_and_up_outputs = Vec::with_capacity(2 * self.expert_layers.len());
        for expert_layer in self.expert_layers {
            match self.gate_up_way {
                GateUpWay::TwoProjections => {
                    gate_and_up_outputs.push(self.gathered(&expert_layer.gate)?);
                    gate_and_up_outputs.push(self.gathered(&expert_layer.up)?);
                }
                GateUpWay::OneFusedProjection => {
                    let fused_output = self.gathered(&expert_layer.fused_gate_up)?;
                    gate_and_up_outputs.push(self.half_of(&fused_output, 0)?);
                    gate_and_up_outputs.push(self.half_of(&fused_output, PROJECTION_WIDTH)?);
                }
            }
        }
        Ok(gate_and_up_outputs)
    }
}

impl GateUpPass<'_> {
    fn gathered(&self, experts: &QuantizedExperts) -> Result<MlxArray, String> {
        self.runtime
            .gather_quantized_matmul_affine(
                self.activations,
                &experts.0,
                &experts.1,
                &experts.2,
                None,
                Some(self.sorted_expert_ids),
                true,
                QUANTIZATION_GROUP_SIZE,
                QUANTIZATION_BITS,
                true,
            )
            .map_err(|gather_error| gather_error.to_string())
    }

    /// The gate or the up half of a fused output, as the engine splits it.
    fn half_of(&self, fused_output: &MlxArray, first_column: i32) -> Result<MlxArray, String> {
        let output_shape = fused_output.shape();
        self.runtime
            .slice(
                fused_output,
                &[0, 0, first_column],
                &[
                    output_shape[0],
                    output_shape[1],
                    first_column + PROJECTION_WIDTH,
                ],
                &[1, 1, 1],
            )
            .map_err(|slice_error| slice_error.to_string())
    }
}

#[test]
#[ignore = "performance measurement on the GPU; run through cargo measure-expert-gate-up-fusion-verdict"]
#[serial]
fn should_give_a_verdict_on_one_fused_gate_and_up_projection_against_two() {
    let runtime = initialize_runtime();
    let random_key = runtime.random_key(2_585).expect("a key builds");
    let mut expert_layers = Vec::with_capacity(LAYER_COUNT);
    for _layer_index in 0..LAYER_COUNT {
        expert_layers.push(expert_layer(&runtime, &random_key));
    }
    let report_directory = Path::new(env!("CARGO_TARGET_TMPDIR")).join(REPORT_SUBDIRECTORY);
    std::fs::create_dir_all(&report_directory).expect("the report directory is made");
    for (cell_index, token_count) in SERVED_TOKEN_COUNTS.into_iter().enumerate() {
        let assignment_count = token_count * EXPERTS_A_TOKEN;
        let activations = runtime
            .random_normal(
                &[assignment_count, 1, INPUT_WIDTH],
                MlxDtype::BFloat16,
                0.0,
                1.0,
                &random_key,
            )
            .expect("the activations are drawn");
        let sorted_expert_ids = sorted_expert_ids(&runtime, assignment_count);
        runtime
            .evaluate_arrays(&[&activations, &sorted_expert_ids])
            .expect("the cell's inputs evaluate");
        let comparison = compare_gate_up_ways(
            (&runtime, &expert_layers),
            (&activations, &sorted_expert_ids),
            (&report_directory, &format!("tokens_{token_count}")),
        );
        eprintln!(
            "[expert-gate-up-fusion-verdict] tokens={token_count} assignments={assignment_count}: job {} | GPU {} | host submission {} | cells {}/{}",
            metric_text(&comparison.job_wall, "fused"),
            metric_text(&comparison.gpu_span, "fused"),
            metric_text(&comparison.host_submission, "fused"),
            cell_index + 1,
            SERVED_TOKEN_COUNTS.len()
        );
    }
    eprintln!(
        "[expert-gate-up-fusion-verdict] status=success pages={}",
        report_directory.display()
    );
}

/// Two projections as the baseline and the fused one as the candidate.
fn compare_gate_up_ways(
    (runtime, expert_layers): (&MlxRuntime, &[ExpertLayer]),
    (activations, sorted_expert_ids): (&MlxArray, &MlxArray),
    (report_directory, file_base): (&Path, &str),
) -> JobComparison {
    let mut two_projections = GateUpPass {
        name: "two_projections",
        runtime,
        gate_up_way: GateUpWay::TwoProjections,
        expert_layers,
        activations,
        sorted_expert_ids,
    };
    let mut one_fused_projection = GateUpPass {
        name: "one_fused_projection",
        runtime,
        gate_up_way: GateUpWay::OneFusedProjection,
        expert_layers,
        activations,
        sorted_expert_ids,
    };
    compare_jobs(
        &mut two_projections,
        &mut one_fused_projection,
        ReportLocation {
            title: "One fused gate and up projection against two",
            directory: report_directory,
            file_base,
        },
    )
    .expect("the two ways are compared")
}

/// One layer of random experts, with its gate and up joined on the output
/// axis the way the removed in-memory engine joined them at load.
fn expert_layer(runtime: &MlxRuntime, random_key: &MlxArray) -> ExpertLayer {
    let gate = quantized_experts(runtime, random_key);
    let up = quantized_experts(runtime, random_key);
    let fused_gate_up = (
        joined(runtime, &gate.0, &up.0),
        joined(runtime, &gate.1, &up.1),
        joined(runtime, &gate.2, &up.2),
    );
    runtime
        .evaluate_arrays(&[&fused_gate_up.0, &fused_gate_up.1, &fused_gate_up.2])
        .expect("the fused weights evaluate");
    ExpertLayer {
        gate,
        up,
        fused_gate_up,
    }
}

fn joined(runtime: &MlxRuntime, gate_part: &MlxArray, up_part: &MlxArray) -> MlxArray {
    runtime
        .concatenate_axis(&[gate_part, up_part], 1)
        .expect("gate and up join on the output axis")
}

fn quantized_experts(runtime: &MlxRuntime, random_key: &MlxArray) -> QuantizedExperts {
    let weights = runtime
        .random_normal(
            &[EXPERT_COUNT, PROJECTION_WIDTH, INPUT_WIDTH],
            MlxDtype::Float32,
            0.0,
            0.02,
            random_key,
        )
        .expect("the weights are drawn");
    let quantized_stack = runtime
        .quantize_affine(&weights, QUANTIZATION_GROUP_SIZE, QUANTIZATION_BITS)
        .expect("the weights quantize");
    runtime
        .evaluate_arrays(&[&quantized_stack.0, &quantized_stack.1, &quantized_stack.2])
        .expect("the quantized weights evaluate");
    quantized_stack
}

/// An expert for every assignment, spread over the layer's experts and
/// sorted, as the sorted route hands them to the gathered multiply.
fn sorted_expert_ids(runtime: &MlxRuntime, assignment_count: i32) -> MlxArray {
    let mut expert_ids = Vec::with_capacity(assignment_count as usize);
    for assignment_index in 0..assignment_count {
        expert_ids.push((assignment_index * 37) % EXPERT_COUNT);
    }
    expert_ids.sort_unstable();
    runtime
        .array_from_i32(&expert_ids, &[assignment_count])
        .expect("the expert ids upload")
}
