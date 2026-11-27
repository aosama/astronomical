//! How a 2-bit folded ternary projection over a prompt chunk compares with
//! the dense float16 multiply the GPU can do, with a verdict.
//!
//! On Ternary Bonsai 2.5 more than 90% of a prompt chunk's GPU time is
//! MLX's 2-bit quantized multiply (`affine_qmm_t_nax`, measured by
//! `cargo measure-prism-prefill-chunk-breakdown`), at about 25 to 27
//! TFLOP/s. This prices two other ways of the same projection against it,
//! as jobs of the MLX performance profiler:
//!
//! - expanding the weight to float16 inside the chunk's graph and running
//!   MLX's dense multiply, the way a prompt chunk could pay one expansion a
//!   projection for 2,048 rows of dense arithmetic;
//! - the dense multiply alone on a weight expanded beforehand: the ceiling
//!   any 2-bit kernel can approach on this GPU.
//!
//! Shapes are a Bonsai 2.5 layer's (5,120 wide, 17,408 MLP, 2-bit in groups
//! of 128, float16 activations), four distinct copies a job so a pass reads
//! DRAM as a forward does. Asserts nothing; prints a verdict a cell.
//!
//! Run with `cargo measure-ternary-prompt-projection-verdict`. No model;
//! about a minute.

use std::path::Path;

use mlx_bridge::{MlxArray, MlxDtype};
use mlx_performance_profiler::{MeasuredJob, ReportLocation, compare_jobs};
use runtime_integration::MlxRuntime;
use serial_test::serial;

use super::k_row_verify_cost::initialize_runtime;
use super::verdict_text::metric_text;

const PROMPT_CHUNK_ROWS: i32 = 2_048;
const QUANTIZATION_BITS: i32 = 2;
const QUANTIZATION_GROUP_SIZE: i32 = 128;
const COPIES_A_JOB: usize = 4;
/// (input width, output width, what it is in a Bonsai 2.5 layer).
const PROJECTION_SHAPES: [(i32, i32, &str); 4] = [
    (5_120, 17_408, "mlp_gate_or_up"),
    (17_408, 5_120, "mlp_down"),
    (5_120, 12_288, "full_attention_query_key_value"),
    (5_120, 10_240, "linear_attention_input"),
];
const REPORT_SUBDIRECTORY: &str = "ternary-prompt-projection-verdict";

/// One projection's weight as the pack stores it, and expanded to float16.
struct ProjectionCopy {
    packed_weight: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
    expanded_transposed_weight: MlxArray,
}

#[derive(Clone, Copy)]
enum ProjectionWay {
    QuantizedMultiply,
    ExpandThenDenseMultiply,
    DenseMultiplyOnExpandedWeight,
}

struct ProjectionPass<'fixture> {
    name: &'static str,
    runtime: &'fixture MlxRuntime,
    projection_way: ProjectionWay,
    projection_copies: &'fixture [ProjectionCopy],
    chunk_rows: &'fixture MlxArray,
}

impl MeasuredJob for ProjectionPass<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn fresh_outputs(&mut self) -> Result<Vec<MlxArray>, String> {
        self.projection_copies
            .iter()
            .map(|projection_copy| {
                self.projected(projection_copy)
                    .map_err(|projection_error| projection_error.to_string())
            })
            .collect()
    }
}

impl ProjectionPass<'_> {
    fn projected(
        &self,
        projection_copy: &ProjectionCopy,
    ) -> Result<MlxArray, mlx_bridge::MlxCError> {
        match self.projection_way {
            ProjectionWay::QuantizedMultiply => self.runtime.quantized_matmul_affine(
                self.chunk_rows,
                &projection_copy.packed_weight,
                &projection_copy.scales,
                &projection_copy.biases,
                true,
                QUANTIZATION_GROUP_SIZE,
                QUANTIZATION_BITS,
            ),
            ProjectionWay::ExpandThenDenseMultiply => {
                let expanded_weight = self.runtime.dequantize_affine(
                    &projection_copy.packed_weight,
                    &projection_copy.scales,
                    &projection_copy.biases,
                    QUANTIZATION_GROUP_SIZE,
                    QUANTIZATION_BITS,
                )?;
                let transposed_weight = self.runtime.transpose_axes(&expanded_weight, &[1, 0])?;
                self.runtime.matmul(self.chunk_rows, &transposed_weight)
            }
            ProjectionWay::DenseMultiplyOnExpandedWeight => self
                .runtime
                .matmul(self.chunk_rows, &projection_copy.expanded_transposed_weight),
        }
    }
}

#[test]
#[ignore = "performance measurement on the GPU; run through cargo measure-ternary-prompt-projection-verdict"]
#[serial]
fn should_give_a_verdict_on_ternary_prompt_projections_against_dense_arithmetic() {
    let runtime = initialize_runtime();
    let random_key = runtime.random_key(2_595).expect("a key builds");
    let report_directory = Path::new(env!("CARGO_TARGET_TMPDIR")).join(REPORT_SUBDIRECTORY);
    std::fs::create_dir_all(&report_directory).expect("the report directory is made");
    for (input_width, output_width, projection_name) in PROJECTION_SHAPES {
        let projection_copies: Vec<ProjectionCopy> = (0..COPIES_A_JOB)
            .map(|_copy_index| projection_copy(&runtime, &random_key, input_width, output_width))
            .collect();
        let chunk_rows = runtime
            .random_normal(
                &[1, PROMPT_CHUNK_ROWS, input_width],
                MlxDtype::Float16,
                0.0,
                1.0,
                &random_key,
            )
            .expect("the chunk rows are drawn");
        runtime
            .evaluate_arrays(&[&chunk_rows])
            .expect("the chunk rows evaluate");
        for candidate_way in [
            ProjectionWay::ExpandThenDenseMultiply,
            ProjectionWay::DenseMultiplyOnExpandedWeight,
        ] {
            print_verdict(
                (&runtime, &projection_copies, &chunk_rows),
                candidate_way,
                (projection_name, input_width, output_width),
                &report_directory,
            );
        }
    }
}

fn print_verdict(
    fixture: (&MlxRuntime, &[ProjectionCopy], &MlxArray),
    candidate_way: ProjectionWay,
    projection: (&str, i32, i32),
    report_directory: &Path,
) {
    let (runtime, projection_copies, chunk_rows) = fixture;
    let (projection_name, input_width, output_width) = projection;
    let candidate_name = match candidate_way {
        ProjectionWay::ExpandThenDenseMultiply => "expand-then-dense",
        _ => "dense-on-expanded",
    };
    let mut baseline = ProjectionPass {
        name: "quantized-2-bit",
        runtime,
        projection_way: ProjectionWay::QuantizedMultiply,
        projection_copies,
        chunk_rows,
    };
    let mut candidate = ProjectionPass {
        name: candidate_name,
        runtime,
        projection_way: candidate_way,
        projection_copies,
        chunk_rows,
    };
    let file_base = format!("{projection_name}_{candidate_name}");
    let comparison = compare_jobs(
        &mut baseline,
        &mut candidate,
        ReportLocation {
            title: "A 2-bit prompt projection against dense float16 arithmetic",
            directory: report_directory,
            file_base: &file_base,
        },
    )
    .expect("the profiler compares the two ways");
    let floating_point_operations = 2.0
        * f64::from(PROMPT_CHUNK_ROWS)
        * f64::from(input_width)
        * f64::from(output_width)
        * COPIES_A_JOB as f64;
    eprintln!(
        "[ternary-prompt-projection-verdict] {projection_name} {input_width}x{output_width}: job {} | GPU {} | baseline {:.1} TFLOP/s, candidate {:.1} TFLOP/s",
        metric_text(&comparison.job_wall, candidate_name),
        metric_text(&comparison.gpu_span, candidate_name),
        floating_point_operations / comparison.job_wall.baseline_median_microseconds / 1e6,
        floating_point_operations / comparison.job_wall.candidate_median_microseconds / 1e6,
    );
}

fn projection_copy(
    runtime: &MlxRuntime,
    random_key: &MlxArray,
    input_width: i32,
    output_width: i32,
) -> ProjectionCopy {
    let float32_weight = runtime
        .random_normal(
            &[output_width, input_width],
            MlxDtype::Float32,
            0.0,
            0.02,
            random_key,
        )
        .expect("the weight is drawn");
    let (packed_weight, float32_scales, float32_biases) = runtime
        .quantize_affine(&float32_weight, QUANTIZATION_GROUP_SIZE, QUANTIZATION_BITS)
        .expect("the weight quantizes");
    // A folded pack keeps its scales and biases in float16, so the multiply
    // returns float16 as serving's does.
    let scales = runtime
        .astype(&float32_scales, MlxDtype::Float16)
        .expect("the scales narrow");
    let biases = runtime
        .astype(&float32_biases, MlxDtype::Float16)
        .expect("the biases narrow");
    let expanded_weight = runtime
        .dequantize_affine(
            &packed_weight,
            &scales,
            &biases,
            QUANTIZATION_GROUP_SIZE,
            QUANTIZATION_BITS,
        )
        .expect("the weight expands");
    // MLX's dense multiply reads a transposed right operand natively, as it
    // reads a linear layer's weight, so the view needs no copy.
    let expanded_transposed_weight = runtime
        .transpose_axes(&expanded_weight, &[1, 0])
        .expect("the expanded weight transposes");
    runtime
        .evaluate_arrays(&[&packed_weight, &scales, &biases, &expanded_weight])
        .expect("the projection evaluates");
    ProjectionCopy {
        packed_weight,
        scales,
        biases,
        expanded_transposed_weight,
    }
}
