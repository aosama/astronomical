//! Whether a 35B-A3B expert layer's gate and up projections over a prompt
//! chunk are faster read straight from the token rows than as the engine
//! makes them today, and whether they give the same bits.
//!
//! Today the engine takes every token's row once per assignment, in sorted
//! order (a 16,384 x 2,048 bf16 copy, 64 MiB, for a 2,048-token chunk), and
//! MLX's `gather_qmm` projects that copy for gate and for up. Astronomical's
//! sorted-expert matmul reads each sorted row straight from the token's own
//! row through the sort order, so the copy is never made, and the 2,048
//! token rows it reads (8 MiB) stay close to the GPU's caches.
//!
//! Same shapes and routing as `sorted_expert_matmul_verdict`: 256 experts,
//! 4-bit groups of 64, gate and up 2,048 in and 512 out, 8 distinct experts a
//! token, four layers a job. The first layer's gate outputs are compared
//! element by element with MLX's, and a mismatch fails the run after the
//! timing is printed; then the two ways are timed as two jobs of the MLX
//! performance profiler, twelve rounds with the order flipped every round.
//!
//! Read on an M5 Pro (2026-11-27), twice: the same bits, and 13.6% then 13.2%
//! less GPU time (18.0 against 20.8 ms for four layers), faster in 12 of 12
//! rounds both times.
//!
//! Run with `cargo measure-sorted-expert-token-rows-verdict`. No model; under
//! a minute.

use std::path::Path;

use mlx_bridge::{MlxArray, MlxDtype};
use mlx_performance_profiler::{MeasuredJob, ReportLocation, compare_jobs};
use model_serving::sorted_expert_matmul::{
    ExpertStack, SortedExpertMatmulKernels, sorted_expert_quantized_matmul_from_token_rows,
};
use runtime_integration::MlxRuntime;
use serial_test::serial;

use super::k_row_verify_cost::initialize_runtime;
use super::sorted_expert_matmul_verdict::{
    EXPERTS_A_TOKEN, PROMPT_TOKENS, QuantizedExperts, quantized_experts, routed_expert_ids,
};
use super::verdict_text::metric_text;
use crate::sorted_expert_fixtures::{
    mismatched_values, mlx_sorted_gather_qmm, uploaded_stable_routing,
};

const INPUT_WIDTH: i32 = 2_048;
const PROJECTION_WIDTH: i32 = 512;
const LAYER_COUNT: usize = 4;
const REPORT_SUBDIRECTORY: &str = "sorted-expert-token-rows-verdict";

/// One layer's gate and up stacks.
struct GateUpLayer {
    gate: QuantizedExperts,
    up: QuantizedExperts,
}

/// What both ways read: the layers, the token rows and the routing.
struct TokenRowsFixture<'fixture> {
    runtime: &'fixture MlxRuntime,
    kernels: &'fixture SortedExpertMatmulKernels,
    layers: Vec<GateUpLayer>,
    token_rows: MlxArray,
    sorted_order: MlxArray,
    sorted_token_indices: MlxArray,
    sorted_expert_ids: MlxArray,
}

#[derive(Clone, Copy)]
enum GateUpWay {
    MlxOnSortedCopy,
    AstronomicalFromTokenRows,
}

struct GateUpPass<'fixture> {
    name: &'static str,
    way: GateUpWay,
    fixture: &'fixture TokenRowsFixture<'fixture>,
}

impl MeasuredJob for GateUpPass<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn fresh_outputs(&mut self) -> Result<Vec<MlxArray>, String> {
        let mut gate_and_up_outputs = Vec::with_capacity(2 * self.fixture.layers.len());
        for layer in &self.fixture.layers {
            let (gate_output, up_output) = self.fixture.gate_and_up(self.way, layer)?;
            gate_and_up_outputs.push(gate_output);
            gate_and_up_outputs.push(up_output);
        }
        Ok(gate_and_up_outputs)
    }
}

impl TokenRowsFixture<'_> {
    /// One layer's gate and up outputs, `[assignments, 1, 512]` in sorted order.
    fn gate_and_up(
        &self,
        way: GateUpWay,
        layer: &GateUpLayer,
    ) -> Result<(MlxArray, MlxArray), String> {
        match way {
            GateUpWay::MlxOnSortedCopy => {
                let sorted_rows = self
                    .runtime
                    .take_axis(&self.token_rows, &self.sorted_token_indices, 0)
                    .map_err(|take_error| take_error.to_string())?;
                Ok((
                    self.mlx_projection(&sorted_rows, &layer.gate)?,
                    self.mlx_projection(&sorted_rows, &layer.up)?,
                ))
            }
            GateUpWay::AstronomicalFromTokenRows => Ok((
                self.token_rows_projection(&layer.gate)?,
                self.token_rows_projection(&layer.up)?,
            )),
        }
    }

    fn mlx_projection(
        &self,
        sorted_rows: &MlxArray,
        experts: &QuantizedExperts,
    ) -> Result<MlxArray, String> {
        Ok(mlx_sorted_gather_qmm(
            self.runtime,
            sorted_rows,
            experts,
            (&self.sorted_expert_ids, 4),
        ))
    }

    fn token_rows_projection(&self, experts: &QuantizedExperts) -> Result<MlxArray, String> {
        sorted_expert_quantized_matmul_from_token_rows(
            self.runtime,
            self.kernels,
            (
                &self.token_rows,
                &self.sorted_order,
                &self.sorted_expert_ids,
            ),
            EXPERTS_A_TOKEN,
            &ExpertStack {
                packed_weight: &experts.0,
                scales: &experts.1,
                biases: &experts.2,
            },
        )
        .map_err(|projection_error| projection_error.to_string())
    }
}

#[test]
#[ignore = "performance measurement on the GPU; run through cargo measure-sorted-expert-token-rows-verdict"]
#[serial]
fn should_give_a_verdict_on_gate_and_up_from_token_rows_against_the_sorted_copy() {
    let runtime = initialize_runtime();
    let random_key = runtime.random_key(2_711).expect("a key builds");
    let kernels = SortedExpertMatmulKernels::declare().expect("the kernels declare");
    let fixture = token_rows_fixture(&runtime, &kernels, &random_key);
    let mismatched_values = count_mismatched_gate_values(&fixture);
    let report_directory = Path::new(env!("CARGO_TARGET_TMPDIR")).join(REPORT_SUBDIRECTORY);
    std::fs::create_dir_all(&report_directory).expect("the report directory is made");
    let comparison = compare_jobs(
        &mut GateUpPass {
            name: "mlx_on_sorted_copy",
            way: GateUpWay::MlxOnSortedCopy,
            fixture: &fixture,
        },
        &mut GateUpPass {
            name: "astronomical_from_token_rows",
            way: GateUpWay::AstronomicalFromTokenRows,
            fixture: &fixture,
        },
        ReportLocation {
            title: "Gate and up from token rows against MLX on a sorted copy",
            directory: &report_directory,
            file_base: "gate_and_up",
        },
    )
    .expect("the two ways are compared");
    eprintln!(
        "[sorted-expert-token-rows-verdict] gate and up, {LAYER_COUNT} layers: mismatched values {mismatched_values} | job {} | GPU {} | pages {}",
        metric_text(&comparison.job_wall, "astronomical"),
        metric_text(&comparison.gpu_span, "astronomical"),
        report_directory.display()
    );
    assert_eq!(
        mismatched_values, 0,
        "the token-row multiply gave different bits from MLX"
    );
}

fn token_rows_fixture<'fixture>(
    runtime: &'fixture MlxRuntime,
    kernels: &'fixture SortedExpertMatmulKernels,
    random_key: &MlxArray,
) -> TokenRowsFixture<'fixture> {
    let layers = (0..LAYER_COUNT)
        .map(|_layer_index| GateUpLayer {
            gate: quantized_experts(runtime, random_key, INPUT_WIDTH, PROJECTION_WIDTH),
            up: quantized_experts(runtime, random_key, INPUT_WIDTH, PROJECTION_WIDTH),
        })
        .collect();
    let token_rows = runtime
        .random_normal(
            &[PROMPT_TOKENS, 1, INPUT_WIDTH],
            MlxDtype::BFloat16,
            0.0,
            1.0,
            random_key,
        )
        .expect("the token rows are drawn");
    let routing = sorted_routing(runtime);
    runtime
        .evaluate_arrays(&[&token_rows])
        .expect("the token rows evaluate");
    TokenRowsFixture {
        runtime,
        kernels,
        layers,
        token_rows,
        sorted_order: routing.0,
        sorted_token_indices: routing.1,
        sorted_expert_ids: routing.2,
    }
}

/// The routing as the engine's assignment sort makes it: the stable sort
/// order of the flat expert ids, each sorted assignment's token, and its
/// expert.
fn sorted_routing(runtime: &MlxRuntime) -> (MlxArray, MlxArray, MlxArray) {
    let routing = uploaded_stable_routing(runtime, &routed_expert_ids(), EXPERTS_A_TOKEN);
    runtime
        .evaluate_arrays(&[&routing.0, &routing.1, &routing.2])
        .expect("the routing evaluates");
    routing
}

/// Values of the first layer's gate output that differ from MLX's.
fn count_mismatched_gate_values(fixture: &TokenRowsFixture<'_>) -> usize {
    let (mlx_gate, _) = fixture
        .gate_and_up(GateUpWay::MlxOnSortedCopy, &fixture.layers[0])
        .expect("MLX projects");
    let (astronomical_gate, _) = fixture
        .gate_and_up(GateUpWay::AstronomicalFromTokenRows, &fixture.layers[0])
        .expect("the token-row multiply projects");
    mismatched_values(fixture.runtime, &mlx_gate, &astronomical_gate)
}
