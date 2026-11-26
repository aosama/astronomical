//! MLX's layout against register order, with a verdict (issue #2601).
//!
//! `register_order_served_rows_cost` prints a median and a range for each
//! layout and leaves the reader to judge whether two medians a few percent
//! apart differ. This journey hands the same two passes to the MLX
//! performance profiler as two jobs: it runs them in twelve rounds with the
//! order flipped every round, takes the difference round by round so drift
//! on the machine lands on both, and says "faster" or "slower" only when the
//! interval for the median difference excludes zero.
//!
//! Each cell is one of the dense Qwen3.8 27B's projection shapes, sixteen
//! distinct copies so a pass reads DRAM, at a row count the engine serves: a
//! reply token's one row, a draft tree's sixteen, a prompt chunk's 512 and
//! 2,048 (a chunk restores MLX's layout for the pass, so its cells price
//! that restoration).
//! Asserts nothing; prints each cell's verdict and writes its comparison
//! page and JSON.
//!
//! Run with `cargo measure-register-order-verdict`. No model; about two
//! minutes.

use std::path::Path;

use mlx_bridge::MlxArray;
use mlx_performance_profiler::{JobComparison, ReportLocation, compare_jobs};
use model_serving::row_batched_matmul::RowBatchedMatmulKernels;
use runtime_integration::MlxRuntime;
use serial_test::serial;

use super::projection_verdict_support::{ProjectionVerdictFixture, ShapeCopiesPass};
use super::verdict_text::metric_text;

use super::k_row_verify_cost::{
    MatmulPath, PROJECTION_SHAPES, PackedProjection, distinct_shape_copies,
};

/// A reply token, a dense draft tree, and the smallest and a usual prompt
/// chunk that restore MLX's layout for the pass.
const SERVED_ROW_COUNTS: [i32; 4] = [1, 16, 512, 2_048];
const REPORT_SUBDIRECTORY: &str = "register-order-layout-verdict";

#[test]
#[ignore = "performance measurement on the GPU; run through cargo measure-register-order-verdict"]
#[serial]
fn should_give_a_verdict_on_register_order_against_mlx_layout_at_served_row_counts() {
    let fixture = ProjectionVerdictFixture::prepare(1234, REPORT_SUBDIRECTORY);
    let (runtime, kernels) = (&fixture.runtime, &fixture.kernels);
    let report_directory = &fixture.report_directory;
    let cell_count = PROJECTION_SHAPES.len() * SERVED_ROW_COUNTS.len();
    let mut finished_cell_count = 0;
    for projection_shape in PROJECTION_SHAPES {
        let shape_copies =
            distinct_shape_copies(runtime, kernels, &fixture.random_key, projection_shape);
        for row_count in SERVED_ROW_COUNTS {
            let input_rows = fixture.evaluated_rows(row_count, projection_shape.1);
            let file_base = format!(
                "{}x{}_rows_{row_count}",
                projection_shape.0, projection_shape.1
            );
            let comparison = compare_layouts(
                LayoutCell {
                    runtime,
                    kernels,
                    shape_copies: &shape_copies,
                    input_rows: &input_rows,
                },
                report_directory,
                &file_base,
            );
            finished_cell_count += 1;
            eprintln!(
                "[register-order-verdict] {file_base}: job {} | GPU {} | host submission {} | cells {finished_cell_count}/{cell_count}",
                metric_text(&comparison.job_wall, "register order"),
                metric_text(&comparison.gpu_span, "register order"),
                metric_text(&comparison.host_submission, "register order")
            );
        }
    }
    eprintln!(
        "[register-order-verdict] status=success pages={}",
        report_directory.display()
    );
}

/// What one cell compares: a shape's copies and the rows they multiply.
struct LayoutCell<'fixture> {
    runtime: &'fixture MlxRuntime,
    kernels: &'fixture RowBatchedMatmulKernels,
    shape_copies: &'fixture [PackedProjection],
    input_rows: &'fixture MlxArray,
}

/// MLX's layout as the baseline and register order as the candidate.
fn compare_layouts(
    cell: LayoutCell<'_>,
    report_directory: &Path,
    file_base: &str,
) -> JobComparison {
    let mut mlx_layout = ShapeCopiesPass {
        name: "mlx_layout",
        runtime: cell.runtime,
        matmul_path: MatmulPath::Mlx,
        shape_copies: cell.shape_copies,
        input_rows: cell.input_rows,
    };
    let mut register_order = ShapeCopiesPass {
        name: "register_order",
        runtime: cell.runtime,
        matmul_path: MatmulPath::RegisterOrdered(cell.kernels),
        shape_copies: cell.shape_copies,
        input_rows: cell.input_rows,
    };
    compare_jobs(
        &mut mlx_layout,
        &mut register_order,
        ReportLocation {
            title: "Register order against MLX's layout",
            directory: report_directory,
            file_base,
        },
    )
    .expect("the two layouts are compared")
}
