//! Whether a 35B-A3B expert layer's routed multiplies over a prompt chunk
//! take less GPU time through the tiled sorted-expert kernels than through
//! the route the engine takes without them, and whether they give the same
//! bits.
//!
//! Without them, 4-bit stacks take Astronomical's sorted-expert matmul for
//! gate and for up from the token rows, MLX's compiled SwiGLU, then MLX's
//! sorted `gather_qmm` for down; 6-bit stacks take MLX's own route: the
//! sorted copy of the rows, MLX's sorted `gather_qmm` for gate and for up,
//! its compiled SwiGLU and its sorted `gather_qmm` for down. Tiled: one
//! kernel for gate, up and their SwiGLU from the token rows (gate and up
//! never written), then the tiled kernel for down
//! (`model_primitives::sorted_expert_matmul::tiled_expert_projection`).
//!
//! Same shapes and routing as `sorted_expert_matmul_verdict`: 256 experts,
//! groups of 64, hidden 2,048, expert width 512, 8 distinct experts a token
//! over 2,048 tokens, four layers a job. The first layer's down output
//! is compared element by element, and a mismatch fails the run after the
//! timing is printed; the two routes are timed as two jobs of the MLX
//! performance profiler, twelve rounds with the order flipped every round.
//!
//! Run with `cargo measure-sorted-expert-tiled-layer-verdict` (both
//! widths). No model; under two minutes.

use std::path::Path;

use mlx_bridge::{MlxArray, MlxCompiledSwiGlu};
use mlx_performance_profiler::{MeasuredJob, ReportLocation, compare_jobs};
use model_serving::sorted_expert_matmul::{
    SortedExpertMatmulKernels, TiledExpertKernels, sorted_expert_quantized_matmul_from_token_rows,
    sorted_expert_swiglu_from_token_rows, tiled_sorted_expert_matmul,
};
use runtime_integration::MlxRuntime;
use serial_test::serial;

use super::k_row_verify_cost::initialize_runtime;
use super::sorted_expert_matmul_verdict::{EXPERTS_A_TOKEN, PROMPT_TOKENS, routed_expert_ids};
use super::verdict_text::metric_text;
use crate::sorted_expert_fixtures::{
    QuantizedStack, mismatched_values, mlx_sorted_gather_qmm, quantized_expert_stack, stack_of,
    standard_normal_rows, uploaded_stable_routing,
};

const EXPERT_COUNT: i32 = 256;
const HIDDEN_WIDTH: i32 = 2_048;
const EXPERT_WIDTH: i32 = 512;
const LAYER_COUNT: usize = 4;
const REPORT_SUBDIRECTORY: &str = "sorted-expert-tiled-layer-verdict";

/// One layer's routed expert stacks.
struct ExpertLayer {
    gate: QuantizedStack,
    up: QuantizedStack,
    down: QuantizedStack,
}

/// What both routes read.
struct LayerFixture<'fixture> {
    runtime: &'fixture MlxRuntime,
    todays_kernels: &'fixture SortedExpertMatmulKernels,
    tiled_kernels: &'fixture TiledExpertKernels,
    compiled_swiglu: MlxCompiledSwiGlu,
    layers: Vec<ExpertLayer>,
    token_rows: MlxArray,
    sorted_order: MlxArray,
    sorted_token_indices: MlxArray,
    sorted_expert_ids: MlxArray,
    weight_bits: i32,
}

#[derive(Clone, Copy)]
enum ExpertRoute {
    /// The 4-bit route before the tiled kernels.
    Today,
    /// MLX's own sorted route, which 6-bit stacks took before them.
    MlxSorted,
    Tiled,
}

struct ExpertLayerPass<'fixture> {
    name: &'static str,
    route: ExpertRoute,
    fixture: &'fixture LayerFixture<'fixture>,
}

impl MeasuredJob for ExpertLayerPass<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn fresh_outputs(&mut self) -> Result<Vec<MlxArray>, String> {
        let mut down_outputs = Vec::with_capacity(self.fixture.layers.len());
        for layer in &self.fixture.layers {
            down_outputs.push(self.fixture.routed_outputs(self.route, layer)?);
        }
        Ok(down_outputs)
    }
}

impl LayerFixture<'_> {
    /// One layer's down outputs, `[assignments, 1, 2048]` in sorted order.
    fn routed_outputs(&self, route: ExpertRoute, layer: &ExpertLayer) -> Result<MlxArray, String> {
        let routing = (
            &self.token_rows,
            &self.sorted_order,
            &self.sorted_expert_ids,
        );
        let mlx_routing = (&self.sorted_expert_ids, self.weight_bits);
        match route {
            ExpertRoute::MlxSorted => {
                let sorted_rows = self
                    .runtime
                    .take_axis(&self.token_rows, &self.sorted_token_indices, 0)
                    .map_err(projection_failure_text)?;
                let gate =
                    mlx_sorted_gather_qmm(self.runtime, &sorted_rows, &layer.gate, mlx_routing);
                let up = mlx_sorted_gather_qmm(self.runtime, &sorted_rows, &layer.up, mlx_routing);
                let activated = self
                    .runtime
                    .apply_compiled_swiglu(&self.compiled_swiglu, &gate, &up)
                    .map_err(projection_failure_text)?;
                Ok(mlx_sorted_gather_qmm(
                    self.runtime,
                    &activated,
                    &layer.down,
                    mlx_routing,
                ))
            }
            ExpertRoute::Today => {
                let gate = sorted_expert_quantized_matmul_from_token_rows(
                    self.runtime,
                    self.todays_kernels,
                    routing,
                    EXPERTS_A_TOKEN,
                    &stack_of(&layer.gate),
                )
                .map_err(projection_failure_text)?;
                let up = sorted_expert_quantized_matmul_from_token_rows(
                    self.runtime,
                    self.todays_kernels,
                    routing,
                    EXPERTS_A_TOKEN,
                    &stack_of(&layer.up),
                )
                .map_err(projection_failure_text)?;
                let activated = self
                    .runtime
                    .apply_compiled_swiglu(&self.compiled_swiglu, &gate, &up)
                    .map_err(projection_failure_text)?;
                Ok(mlx_sorted_gather_qmm(
                    self.runtime,
                    &activated,
                    &layer.down,
                    mlx_routing,
                ))
            }
            ExpertRoute::Tiled => {
                let activated = sorted_expert_swiglu_from_token_rows(
                    self.runtime,
                    self.tiled_kernels,
                    routing,
                    EXPERTS_A_TOKEN,
                    (&stack_of(&layer.gate), &stack_of(&layer.up)),
                )
                .map_err(projection_failure_text)?;
                tiled_sorted_expert_matmul(
                    self.runtime,
                    self.tiled_kernels,
                    &activated,
                    &stack_of(&layer.down),
                    &self.sorted_expert_ids,
                )
                .map_err(projection_failure_text)
            }
        }
    }
}

#[test]
#[ignore = "performance measurement on the GPU; run through cargo measure-sorted-expert-tiled-layer-verdict"]
#[serial]
fn should_give_a_verdict_on_the_tiled_routed_experts_against_todays_route() {
    run_tiled_verdict(
        (ExpertRoute::Today, "today"),
        4,
        2_721,
        "Routed experts: tiled kernels against today's route",
    );
}

#[test]
#[ignore = "performance measurement on the GPU; run through cargo measure-sorted-expert-tiled-layer-verdict"]
#[serial]
fn should_give_a_verdict_on_the_6bit_tiled_routed_experts_against_mlxs_route() {
    run_tiled_verdict(
        (ExpertRoute::MlxSorted, "mlx"),
        6,
        2_724,
        "6-bit routed experts: tiled kernels against MLX's sorted route",
    );
}

/// Times the tiled route against `baseline` on `weight_bits` stacks as two
/// profiler jobs, then fails if the first layer's down outputs differ in any
/// bit.
fn run_tiled_verdict(
    (baseline_route, baseline_name): (ExpertRoute, &'static str),
    weight_bits: i32,
    seed: u64,
    report_title: &str,
) {
    let runtime = initialize_runtime();
    let random_key = runtime.random_key(seed).expect("a key builds");
    let todays_kernels = SortedExpertMatmulKernels::declare().expect("today's kernels declare");
    let tiled_kernels = TiledExpertKernels::declare().expect("the tiled kernels declare");
    let fixture = layer_fixture(
        &runtime,
        (&todays_kernels, &tiled_kernels),
        (&random_key, weight_bits),
    );
    let mismatched_down_values = mismatched_values(
        &runtime,
        &fixture
            .routed_outputs(baseline_route, &fixture.layers[0])
            .expect("the baseline route runs"),
        &fixture
            .routed_outputs(ExpertRoute::Tiled, &fixture.layers[0])
            .expect("the tiled route runs"),
    );
    let report_directory = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(REPORT_SUBDIRECTORY)
        .join(format!("{weight_bits}bit"));
    std::fs::create_dir_all(&report_directory).expect("the report directory is made");
    let comparison = compare_jobs(
        &mut ExpertLayerPass {
            name: baseline_name,
            route: baseline_route,
            fixture: &fixture,
        },
        &mut ExpertLayerPass {
            name: "tiled",
            route: ExpertRoute::Tiled,
            fixture: &fixture,
        },
        ReportLocation {
            title: report_title,
            directory: &report_directory,
            file_base: "routed_experts",
        },
    )
    .expect("the two routes are compared");
    eprintln!(
        "[sorted-expert-tiled-layer-verdict] {weight_bits}-bit routed experts, {LAYER_COUNT} layers: mismatched down values {mismatched_down_values} | job {} | GPU {} | pages {}",
        metric_text(&comparison.job_wall, "tiled"),
        metric_text(&comparison.gpu_span, "tiled"),
        report_directory.display()
    );
    assert_eq!(
        mismatched_down_values, 0,
        "the tiled route gave different bits from the {baseline_name} route"
    );
}

fn layer_fixture<'fixture>(
    runtime: &'fixture MlxRuntime,
    (todays_kernels, tiled_kernels): (
        &'fixture SortedExpertMatmulKernels,
        &'fixture TiledExpertKernels,
    ),
    (random_key, weight_bits): (&MlxArray, i32),
) -> LayerFixture<'fixture> {
    let mut layers = Vec::with_capacity(LAYER_COUNT);
    for _ in 0..LAYER_COUNT {
        layers.push(ExpertLayer {
            gate: layer_stack(
                runtime,
                random_key,
                (EXPERT_WIDTH, HIDDEN_WIDTH),
                weight_bits,
            ),
            up: layer_stack(
                runtime,
                random_key,
                (EXPERT_WIDTH, HIDDEN_WIDTH),
                weight_bits,
            ),
            down: layer_stack(
                runtime,
                random_key,
                (HIDDEN_WIDTH, EXPERT_WIDTH),
                weight_bits,
            ),
        });
    }
    let token_rows = standard_normal_rows(runtime, random_key, (PROMPT_TOKENS, HIDDEN_WIDTH));
    let (sorted_order, sorted_token_indices, sorted_expert_ids) =
        uploaded_stable_routing(runtime, &routed_expert_ids(), EXPERTS_A_TOKEN);
    runtime
        .evaluate_arrays(&[
            &token_rows,
            &sorted_order,
            &sorted_token_indices,
            &sorted_expert_ids,
        ])
        .expect("the inputs evaluate");
    LayerFixture {
        runtime,
        todays_kernels,
        tiled_kernels,
        compiled_swiglu: MlxCompiledSwiGlu::new().expect("the SwiGLU compiles"),
        layers,
        token_rows,
        sorted_order,
        sorted_token_indices,
        sorted_expert_ids,
        weight_bits,
    }
}

/// One `(output, input)` expert stack of the layer at `weight_bits`.
fn layer_stack(
    runtime: &MlxRuntime,
    random_key: &MlxArray,
    (output_width, input_width): (i32, i32),
    weight_bits: i32,
) -> QuantizedStack {
    quantized_expert_stack(
        runtime,
        random_key,
        (EXPERT_COUNT, output_width, input_width),
        weight_bits,
    )
}

/// A failed multiply as the profiler's job reports it.
fn projection_failure_text(projection_failure: impl std::fmt::Display) -> String {
    projection_failure.to_string()
}
