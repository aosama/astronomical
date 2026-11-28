//! Whether a 35B-A3B expert layer's routed multiplies over a prompt chunk
//! take less GPU time through the tiled sorted-expert kernels than through
//! the route the engine takes today, and whether they give the same bits.
//!
//! Today: Astronomical's sorted-expert matmul for gate and for up from the
//! token rows, MLX's compiled SwiGLU, then MLX's sorted `gather_qmm` for
//! down. Tiled: one kernel for gate, up and their SwiGLU from the token rows
//! (gate and up never written), then the tiled kernel for down
//! (`model_primitives::sorted_expert_matmul::tiled_expert_projection`).
//!
//! Same shapes and routing as `sorted_expert_matmul_verdict`: 256 experts,
//! 4-bit groups of 64, hidden 2,048, expert width 512, 8 distinct experts a
//! token over 2,048 tokens, four layers a job. The first layer's down output
//! is compared element by element, and a mismatch fails the run after the
//! timing is printed; the two routes are timed as two jobs of the MLX
//! performance profiler, twelve rounds with the order flipped every round.
//!
//! Run with `cargo measure-sorted-expert-tiled-layer-verdict`. No model;
//! under a minute.

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
use super::sorted_expert_matmul_verdict::{
    EXPERTS_A_TOKEN, PROMPT_TOKENS, quantized_experts, routed_expert_ids,
};
use super::verdict_text::metric_text;
use crate::sorted_expert_fixtures::{
    QuantizedStack, mismatched_values, mlx_sorted_gather_qmm, stack_of, standard_normal_rows,
    uploaded_stable_routing,
};

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
    sorted_expert_ids: MlxArray,
}

#[derive(Clone, Copy)]
enum ExpertRoute {
    Today,
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
        match route {
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
                    &self.sorted_expert_ids,
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
    let runtime = initialize_runtime();
    let random_key = runtime.random_key(2_721).expect("a key builds");
    let todays_kernels = SortedExpertMatmulKernels::declare().expect("today's kernels declare");
    let tiled_kernels = TiledExpertKernels::declare().expect("the tiled kernels declare");
    let fixture = layer_fixture(&runtime, (&todays_kernels, &tiled_kernels), &random_key);
    let mismatched_down_values = mismatched_values(
        &runtime,
        &fixture
            .routed_outputs(ExpertRoute::Today, &fixture.layers[0])
            .expect("today's route runs"),
        &fixture
            .routed_outputs(ExpertRoute::Tiled, &fixture.layers[0])
            .expect("the tiled route runs"),
    );
    let report_directory = Path::new(env!("CARGO_TARGET_TMPDIR")).join(REPORT_SUBDIRECTORY);
    std::fs::create_dir_all(&report_directory).expect("the report directory is made");
    let comparison = compare_jobs(
        &mut ExpertLayerPass {
            name: "today",
            route: ExpertRoute::Today,
            fixture: &fixture,
        },
        &mut ExpertLayerPass {
            name: "tiled",
            route: ExpertRoute::Tiled,
            fixture: &fixture,
        },
        ReportLocation {
            title: "Routed experts: tiled kernels against today's route",
            directory: &report_directory,
            file_base: "routed_experts",
        },
    )
    .expect("the two routes are compared");
    eprintln!(
        "[sorted-expert-tiled-layer-verdict] routed experts, {LAYER_COUNT} layers: mismatched down values {mismatched_down_values} | job {} | GPU {} | pages {}",
        metric_text(&comparison.job_wall, "tiled"),
        metric_text(&comparison.gpu_span, "tiled"),
        report_directory.display()
    );
    assert_eq!(
        mismatched_down_values, 0,
        "the tiled route gave different bits from today's"
    );
}

fn layer_fixture<'fixture>(
    runtime: &'fixture MlxRuntime,
    (todays_kernels, tiled_kernels): (
        &'fixture SortedExpertMatmulKernels,
        &'fixture TiledExpertKernels,
    ),
    random_key: &MlxArray,
) -> LayerFixture<'fixture> {
    let mut layers = Vec::with_capacity(LAYER_COUNT);
    for _ in 0..LAYER_COUNT {
        layers.push(ExpertLayer {
            gate: quantized_experts(runtime, random_key, HIDDEN_WIDTH, EXPERT_WIDTH),
            up: quantized_experts(runtime, random_key, HIDDEN_WIDTH, EXPERT_WIDTH),
            down: quantized_experts(runtime, random_key, EXPERT_WIDTH, HIDDEN_WIDTH),
        });
    }
    let token_rows = standard_normal_rows(runtime, random_key, (PROMPT_TOKENS, HIDDEN_WIDTH));
    let (sorted_order, _, sorted_expert_ids) =
        uploaded_stable_routing(runtime, &routed_expert_ids(), EXPERTS_A_TOKEN);
    runtime
        .evaluate_arrays(&[&token_rows, &sorted_order, &sorted_expert_ids])
        .expect("the inputs evaluate");
    LayerFixture {
        runtime,
        todays_kernels,
        tiled_kernels,
        compiled_swiglu: MlxCompiledSwiGlu::new().expect("the SwiGLU compiles"),
        layers,
        token_rows,
        sorted_order,
        sorted_expert_ids,
    }
}

/// A failed multiply as the profiler's job reports it.
fn projection_failure_text(projection_failure: impl std::fmt::Display) -> String {
    projection_failure.to_string()
}
