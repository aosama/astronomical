//! How many input slices the register-ordered kernel should split each of the
//! dense 27B's projection shapes into, with a profiler verdict (issue #2601).
//!
//! A slice is a simdgroup that takes a contiguous run of the input's groups;
//! the slices are added in a fixed order, so a row's bits depend on the
//! shape and its slice count only, never on the rows a call carries. The
//! counts were TensorFold's for one row; a verify pass multiplies 16, and
//! the feed-forward projections take 60% of its GPU time. Each cell compares,
//! for every projection shape, the slice count today (baseline) against
//! another (candidate), at 16 rows and at 1, over sixteen distinct copies so
//! a pass reads DRAM.
//!
//! Run with `cargo measure-register-order-slice-verdict`. No model; about
//! twenty seconds.

use mlx_bridge::MlxArray;
use mlx_performance_profiler::{ReportLocation, compare_jobs};
use serial_test::serial;

use super::k_row_verify_cost::{
    MatmulPath, PROJECTION_SHAPES, PackedProjection, distinct_shape_copies,
};
use super::projection_verdict_support::{ProjectionVerdictFixture, ShapeCopiesPass};
use super::verdict_text::metric_text;

const SWEPT_SLICE_COUNTS: [i32; 4] = [1, 2, 4, 8];
const SWEPT_ROW_COUNTS: [i32; 2] = [16, 1];
const REPORT_SUBDIRECTORY: &str = "register-order-slice-verdict";

#[test]
#[ignore = "performance measurement on the GPU; run through cargo measure-register-order-slice-verdict"]
#[serial]
fn should_give_a_verdict_on_each_slice_count_for_the_dense_27b_shapes() {
    let fixture = ProjectionVerdictFixture::prepare(2_601, REPORT_SUBDIRECTORY);
    for projection_shape in PROJECTION_SHAPES {
        let today_copies = shape_copies_split_into(&fixture, projection_shape, None);
        let today_slice_count = today_copies[0]
            .register_ordered
            .as_ref()
            .expect("the shape converts")
            .input_slice_count;
        for candidate_slice_count in SWEPT_SLICE_COUNTS {
            if candidate_slice_count == today_slice_count {
                continue;
            }
            let candidate_copies =
                shape_copies_split_into(&fixture, projection_shape, Some(candidate_slice_count));
            for row_count in SWEPT_ROW_COUNTS {
                let input_rows = fixture.evaluated_rows(row_count, projection_shape.1);
                let file_base = format!(
                    "{}x{}_rows_{row_count}_slices_{today_slice_count}_against_{candidate_slice_count}",
                    projection_shape.0, projection_shape.1
                );
                let gpu_verdict = compare_slice_counts(
                    &fixture,
                    (&today_copies, &candidate_copies),
                    &input_rows,
                    &file_base,
                );
                eprintln!("[register-order-slices] {file_base}: GPU {gpu_verdict}");
            }
        }
    }
}

/// Sixteen distinct copies of the shape, each split into `slice_count`
/// input slices, or into the count the shape takes today when `None`.
fn shape_copies_split_into(
    fixture: &ProjectionVerdictFixture,
    projection_shape: (i32, i32),
    slice_count: Option<i32>,
) -> Vec<PackedProjection> {
    let mut shape_copies = distinct_shape_copies(
        &fixture.runtime,
        &fixture.kernels,
        &fixture.random_key,
        projection_shape,
    );
    if let Some(slice_count) = slice_count {
        for shape_copy in &mut shape_copies {
            if let Some(register_ordered) = shape_copy.register_ordered.as_mut() {
                register_ordered.input_slice_count = slice_count;
            }
        }
    }
    shape_copies
}

/// Today's split as the baseline and the candidate's, over the same rows;
/// returns the GPU span's verdict as text.
fn compare_slice_counts(
    fixture: &ProjectionVerdictFixture,
    (today_copies, candidate_copies): (&[PackedProjection], &[PackedProjection]),
    input_rows: &MlxArray,
    file_base: &str,
) -> String {
    let mut today = ShapeCopiesPass {
        name: "slices_today",
        runtime: &fixture.runtime,
        matmul_path: MatmulPath::RegisterOrdered(&fixture.kernels),
        shape_copies: today_copies,
        input_rows,
    };
    let mut candidate = ShapeCopiesPass {
        name: "slices_candidate",
        shape_copies: candidate_copies,
        ..today
    };
    let comparison = compare_jobs(
        &mut today,
        &mut candidate,
        ReportLocation {
            title: "Register-ordered input slices",
            directory: &fixture.report_directory,
            file_base,
        },
    )
    .expect("the slice counts are compared");
    metric_text(&comparison.gpu_span, "the candidate split")
}
