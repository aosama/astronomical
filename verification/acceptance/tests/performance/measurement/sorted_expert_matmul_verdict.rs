//! Whether Astronomical's sorted-expert matmul is faster than MLX's
//! `gather_qmm` at a prompt chunk's expert multiplies, and whether it gives
//! the same bits.
//!
//! The shapes are a 35B-A3B expert layer's: 256 experts, 4-bit in groups of
//! 64, a gate (or up) projection 2,048 in and 512 out and a down projection
//! 512 in and 2,048 out, over a 2,048-token prompt chunk's 16,384 assignments.
//! Every token picks 8 distinct experts from a fixed pseudo-random stream, so
//! the experts' row counts vary as a real router's do (about 64 rows each,
//! some more, some fewer), which is what decides how many row tiles split at
//! an expert boundary. Four distinct layers a job, so a pass reads memory as
//! a forward does.
//!
//! First each projection's outputs are compared element by element with
//! MLX's: the kernel is built to give the same bits, and a mismatch fails the
//! run after the timing is printed. Then the two are timed as two jobs of the
//! MLX performance profiler, twelve rounds with the order flipped every
//! round, "faster" or "slower" only when the interval for the median
//! difference excludes zero.
//!
//! Read on an M5 Pro (2026-11-27): every layout gave MLX's bits, and none
//! beat MLX on rows that are already a sorted copy; the best (128 columns,
//! staged rows, 96-row passes) took about 8% more GPU time for gate and for
//! down. What MLX cannot do is read the rows without that copy, which is where
//! the kernel wins (`sorted_expert_token_rows_verdict`).
//!
//! Run with `cargo measure-sorted-expert-matmul-verdict`. No model; about a
//! minute.

use std::path::Path;

use mlx_bridge::{MlxArray, MlxDtype};
use mlx_performance_profiler::{JobComparison, MeasuredJob, ReportLocation, compare_jobs};
use model_serving::sorted_expert_matmul::{
    ExpertStack, SortedExpertLayout, SortedExpertMatmulKernels,
    sorted_expert_quantized_matmul_with_layout,
};
use runtime_integration::MlxRuntime;
use serial_test::serial;

use super::k_row_verify_cost::initialize_runtime;
use super::verdict_text::metric_text;
use crate::sorted_expert_fixtures::{
    QUANTIZATION_GROUP_SIZE, mismatched_values, quantized_expert_stack, tiled_layout,
};

pub(super) const EXPERT_COUNT: i32 = 256;
pub(super) const EXPERTS_A_TOKEN: i32 = 8;
pub(super) const PROMPT_TOKENS: i32 = 2_048;
pub(super) const QUANTIZATION_BITS: i32 = 4;
const LAYER_COUNT: usize = 4;
/// (name, input width, output width) of the two expert projection shapes.
const PROJECTIONS: [(&str, i32, i32); 2] = [("gate", 2_048, 512), ("down", 512, 2_048)];
const REPORT_SUBDIRECTORY: &str = "sorted-expert-matmul-verdict";

/// One quantized expert stack, `[experts, out, in]`, as MLX's quantizer
/// returns it: packed weights, scales and biases.
pub(super) type QuantizedExperts = (MlxArray, MlxArray, MlxArray);

/// The tile layouts compared, each against MLX: (name, layout).
const LAYOUTS: [(&str, SortedExpertLayout); 4] = [
    ("wide_staged_96", tiled_layout(4, true, 96)),
    ("wide_staged_64", tiled_layout(4, true, 64)),
    ("narrow_unstaged_96", tiled_layout(2, false, 96)),
    ("narrow_unstaged_64", tiled_layout(2, false, 64)),
];

#[derive(Clone, Copy)]
enum Implementation {
    MlxGatherQmm,
    AstronomicalSortedExpert(SortedExpertLayout),
}

/// Every layer's projection of one prompt chunk's sorted assignments.
struct ExpertProjectionPass<'fixture> {
    name: &'static str,
    implementation: Implementation,
    fixture: &'fixture ProjectionFixture<'fixture>,
}

/// What both implementations read: the layers, the sorted rows and ids.
struct ProjectionFixture<'fixture> {
    runtime: &'fixture MlxRuntime,
    kernels: &'fixture SortedExpertMatmulKernels,
    layers: Vec<QuantizedExperts>,
    sorted_rows: MlxArray,
    sorted_expert_ids: &'fixture MlxArray,
}

impl MeasuredJob for ExpertProjectionPass<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn fresh_outputs(&mut self) -> Result<Vec<MlxArray>, String> {
        self.fixture
            .layers
            .iter()
            .map(|experts| self.fixture.project(self.implementation, experts))
            .collect()
    }
}

impl ProjectionFixture<'_> {
    fn project(
        &self,
        implementation: Implementation,
        experts: &QuantizedExperts,
    ) -> Result<MlxArray, String> {
        match implementation {
            Implementation::MlxGatherQmm => self
                .runtime
                .gather_quantized_matmul_affine(
                    &self.sorted_rows,
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
                .map_err(|gather_error| gather_error.to_string()),
            Implementation::AstronomicalSortedExpert(layout) => {
                sorted_expert_quantized_matmul_with_layout(
                    self.runtime,
                    self.kernels,
                    (&self.sorted_rows, self.sorted_expert_ids),
                    &ExpertStack {
                        packed_weight: &experts.0,
                        scales: &experts.1,
                        biases: &experts.2,
                    },
                    layout,
                )
                .map_err(|projection_error| projection_error.to_string())
            }
        }
    }
}

#[test]
#[ignore = "performance measurement on the GPU; run through cargo measure-sorted-expert-matmul-verdict"]
#[serial]
fn should_give_a_verdict_on_the_sorted_expert_matmul_against_mlx_gather_qmm() {
    let runtime = initialize_runtime();
    let random_key = runtime.random_key(2_710).expect("a key builds");
    let kernels = SortedExpertMatmulKernels::declare().expect("the kernels declare");
    let sorted_expert_ids = routed_sorted_expert_ids(&runtime);
    let report_directory = Path::new(env!("CARGO_TARGET_TMPDIR")).join(REPORT_SUBDIRECTORY);
    std::fs::create_dir_all(&report_directory).expect("the report directory is made");
    let mut mismatched_projections = Vec::new();
    for (projection_name, input_width, output_width) in PROJECTIONS {
        let fixture = projection_fixture(
            (&runtime, &kernels, &random_key),
            (input_width, output_width),
            &sorted_expert_ids,
        );
        for (layout_name, layout) in LAYOUTS {
            let mismatched_values = count_mismatched_values(&fixture, layout);
            let comparison = compare_implementations(
                &fixture,
                layout,
                (
                    &report_directory,
                    &format!("{projection_name}_{layout_name}"),
                ),
            );
            eprintln!(
                "[sorted-expert-matmul-verdict] {projection_name} {input_width}->{output_width} {layout_name}: mismatched values {mismatched_values} | job {} | GPU {}",
                metric_text(&comparison.job_wall, "astronomical"),
                metric_text(&comparison.gpu_span, "astronomical"),
            );
            if mismatched_values > 0 {
                mismatched_projections.push(format!("{projection_name} {layout_name}"));
            }
        }
    }
    eprintln!(
        "[sorted-expert-matmul-verdict] status=done pages={}",
        report_directory.display()
    );
    assert!(
        mismatched_projections.is_empty(),
        "the sorted-expert matmul gave different bits from MLX for {mismatched_projections:?}"
    );
}

fn projection_fixture<'fixture>(
    (runtime, kernels, random_key): (
        &'fixture MlxRuntime,
        &'fixture SortedExpertMatmulKernels,
        &MlxArray,
    ),
    (input_width, output_width): (i32, i32),
    sorted_expert_ids: &'fixture MlxArray,
) -> ProjectionFixture<'fixture> {
    let layers = (0..LAYER_COUNT)
        .map(|_layer_index| quantized_experts(runtime, random_key, input_width, output_width))
        .collect();
    let sorted_rows = runtime
        .random_normal(
            &[PROMPT_TOKENS * EXPERTS_A_TOKEN, 1, input_width],
            MlxDtype::BFloat16,
            0.0,
            1.0,
            random_key,
        )
        .expect("the rows are drawn");
    runtime
        .evaluate_arrays(&[&sorted_rows])
        .expect("the rows evaluate");
    ProjectionFixture {
        runtime,
        kernels,
        layers,
        sorted_rows,
        sorted_expert_ids,
    }
}

/// Values of the first layer's output that differ from MLX's, compared as
/// numbers with NaN equal to NaN; 0 when the bits agree.
fn count_mismatched_values(fixture: &ProjectionFixture<'_>, layout: SortedExpertLayout) -> usize {
    let mlx_output = fixture
        .project(Implementation::MlxGatherQmm, &fixture.layers[0])
        .expect("MLX projects");
    let astronomical_output = fixture
        .project(
            Implementation::AstronomicalSortedExpert(layout),
            &fixture.layers[0],
        )
        .expect("the sorted-expert kernel projects");
    mismatched_values(fixture.runtime, &mlx_output, &astronomical_output)
}

fn compare_implementations(
    fixture: &ProjectionFixture<'_>,
    layout: SortedExpertLayout,
    (report_directory, file_base): (&Path, &str),
) -> JobComparison {
    let mut mlx_pass = ExpertProjectionPass {
        name: "mlx_gather_qmm",
        implementation: Implementation::MlxGatherQmm,
        fixture,
    };
    let mut astronomical_pass = ExpertProjectionPass {
        name: "astronomical_sorted_expert",
        implementation: Implementation::AstronomicalSortedExpert(layout),
        fixture,
    };
    compare_jobs(
        &mut mlx_pass,
        &mut astronomical_pass,
        ReportLocation {
            title: "Sorted-expert matmul against MLX's gather_qmm",
            directory: report_directory,
            file_base,
        },
    )
    .expect("the two implementations are compared")
}

/// One layer of random 4-bit experts with bf16 scales and biases.
pub(super) fn quantized_experts(
    runtime: &MlxRuntime,
    random_key: &MlxArray,
    input_width: i32,
    output_width: i32,
) -> QuantizedExperts {
    quantized_expert_stack(
        runtime,
        random_key,
        (EXPERT_COUNT, output_width, input_width),
        QUANTIZATION_BITS,
    )
}

/// Every token's 8 distinct experts over a `PROMPT_TOKENS` chunk.
pub(super) fn routed_expert_ids() -> Vec<i32> {
    routed_expert_ids_for(PROMPT_TOKENS)
}

/// Every token's 8 distinct experts from a fixed pseudo-random stream,
/// token by token over `token_count` tokens: the router's choices before
/// its sort. A longer chunk starts with the shorter one's tokens.
pub(super) fn routed_expert_ids_for(token_count: i32) -> Vec<i32> {
    let mut stream_state: u64 = 0x2710_5eed;
    let mut expert_ids = Vec::with_capacity((token_count * EXPERTS_A_TOKEN) as usize);
    for _token_index in 0..token_count {
        let mut token_experts: Vec<i32> = Vec::with_capacity(EXPERTS_A_TOKEN as usize);
        while token_experts.len() < EXPERTS_A_TOKEN as usize {
            stream_state = stream_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let expert = ((stream_state >> 33) % EXPERT_COUNT as u64) as i32;
            if !token_experts.contains(&expert) {
                token_experts.push(expert);
            }
        }
        expert_ids.extend(token_experts);
    }
    expert_ids
}

/// All assignments' experts sorted, as the router's sort hands them over.
fn routed_sorted_expert_ids(runtime: &MlxRuntime) -> MlxArray {
    let mut expert_ids = routed_expert_ids();
    expert_ids.sort_unstable();
    let ids = runtime
        .array_from_i32(&expert_ids, &[PROMPT_TOKENS * EXPERTS_A_TOKEN])
        .expect("the expert ids upload");
    runtime.evaluate_arrays(&[&ids]).expect("the ids evaluate");
    ids
}
