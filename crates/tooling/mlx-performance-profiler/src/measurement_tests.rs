use std::path::Path;

use mlx_bridge::{MlxArray, MlxDtype};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::file_serial;

use crate::{MeasuredJob, ReportLocation, Verdict, compare_jobs, measure_job};

const MEMORY_LIMIT_BYTES: usize = 1 << 30;
const MATRIX_SIDE: i32 = 512;

/// `left` times `right`, a chosen number of times over: a job whose cost the
/// test controls.
struct RepeatedMatmul<'runtime> {
    name: &'static str,
    runtime: &'runtime MlxRuntime,
    left: &'runtime MlxArray,
    right: &'runtime MlxArray,
    repetition_count: usize,
}

impl MeasuredJob for RepeatedMatmul<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn fresh_outputs(&mut self) -> Result<Vec<MlxArray>, String> {
        let mut product = self
            .runtime
            .matmul(self.left, self.right)
            .map_err(|matmul_error| matmul_error.to_string())?;
        for _ in 1..self.repetition_count {
            product = self
                .runtime
                .matmul(&product, self.right)
                .map_err(|matmul_error| matmul_error.to_string())?;
        }
        Ok(vec![product])
    }
}

/// A job that cannot build its graph.
struct FailingJob;

impl MeasuredJob for FailingJob {
    fn name(&self) -> &str {
        "failing"
    }

    fn fresh_outputs(&mut self) -> Result<Vec<MlxArray>, String> {
        Err("this job has no graph".to_owned())
    }
}

fn skip_without_metal(test_name: &str) -> bool {
    test_harness::skip_without(test_harness::PlatformRequirement::MetalGpu, test_name)
}

fn runtime_and_matrices() -> (MlxRuntime, MlxArray, MlxArray) {
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES)
            .expect("the memory limits are valid"),
    )
    .expect("the MLX runtime initializes");
    let random_key = runtime.random_key(7).expect("a key builds");
    let left = runtime
        .random_normal(
            &[MATRIX_SIDE, MATRIX_SIDE],
            MlxDtype::Float32,
            0.0,
            1.0,
            &random_key,
        )
        .expect("a matrix builds");
    // Multiplying by an identity keeps a chain of products equal to the
    // first matrix, so the two jobs' outputs can be compared.
    let right = runtime
        .identity(MATRIX_SIDE, MlxDtype::Float32)
        .expect("an identity builds");
    runtime
        .evaluate_arrays(&[&left, &right])
        .expect("the matrices evaluate");
    (runtime, left, right)
}

fn report_in(directory: &Path) -> ReportLocation<'_> {
    ReportLocation {
        title: "Profiler contract",
        directory,
        file_base: "contract",
    }
}

#[test]
#[file_serial]
fn should_measure_a_job_and_write_its_page_and_node_records() {
    if skip_without_metal("should_measure_a_job_and_write_its_page_and_node_records") {
        return;
    }
    let (runtime, left, right) = runtime_and_matrices();
    let report_directory = tempfile::tempdir().expect("a directory is made");
    let mut job = RepeatedMatmul {
        name: "one_product",
        runtime: &runtime,
        left: &left,
        right: &right,
        repetition_count: 1,
    };

    let summary =
        measure_job(&mut job, report_in(report_directory.path())).expect("the job is measured");

    assert!(summary.job_wall_median_microseconds > 0.0);
    assert!(summary.executed_node_count >= 1);
    assert!(report_directory.path().join("contract.html").is_file());
    let node_records = std::fs::read_to_string(report_directory.path().join("contract.json"))
        .expect("the measurement file is written");
    assert!(node_records.contains("one_product"));
}

#[test]
#[file_serial]
fn should_call_a_job_that_does_eight_times_the_work_slower() {
    if skip_without_metal("should_call_a_job_that_does_eight_times_the_work_slower") {
        return;
    }
    let (runtime, left, right) = runtime_and_matrices();
    let report_directory = tempfile::tempdir().expect("a directory is made");
    let mut one_product = RepeatedMatmul {
        name: "one_product",
        runtime: &runtime,
        left: &left,
        right: &right,
        repetition_count: 1,
    };
    let mut eight_products = RepeatedMatmul {
        name: "eight_products",
        runtime: &runtime,
        left: &left,
        right: &right,
        repetition_count: 8,
    };

    let comparison = compare_jobs(
        &mut one_product,
        &mut eight_products,
        report_in(report_directory.path()),
    )
    .expect("the jobs are compared");

    assert_eq!(comparison.job_wall.verdict, Verdict::CandidateSlower);
    assert!(comparison.job_wall.median_difference_microseconds > 0.0);
    assert_eq!(comparison.job_wall.round_count, 12);
    assert!(comparison.largest_output_difference < 1e-3);
    assert!(report_directory.path().join("contract.html").is_file());
}

#[test]
#[file_serial]
fn should_report_a_jobs_own_reason_when_it_cannot_build_its_graph() {
    if skip_without_metal("should_report_a_jobs_own_reason_when_it_cannot_build_its_graph") {
        return;
    }
    let report_directory = tempfile::tempdir().expect("a directory is made");

    let failure = measure_job(&mut FailingJob, report_in(report_directory.path()))
        .expect_err("a job with no graph is not measured");

    assert_eq!(failure.description, "this job has no graph");
    assert!(!report_directory.path().join("contract.html").exists());
}

#[test]
#[file_serial]
fn should_refuse_two_jobs_with_one_name_and_a_name_holding_a_nul() {
    if skip_without_metal("should_refuse_two_jobs_with_one_name_and_a_name_holding_a_nul") {
        return;
    }
    let (runtime, left, right) = runtime_and_matrices();
    let report_directory = tempfile::tempdir().expect("a directory is made");
    let mut first = RepeatedMatmul {
        name: "same",
        runtime: &runtime,
        left: &left,
        right: &right,
        repetition_count: 1,
    };
    let mut second = RepeatedMatmul {
        name: "same",
        runtime: &runtime,
        left: &left,
        right: &right,
        repetition_count: 1,
    };
    assert!(compare_jobs(&mut first, &mut second, report_in(report_directory.path())).is_err());

    let nul_title = ReportLocation {
        title: "a\0b",
        directory: report_directory.path(),
        file_base: "contract",
    };
    let failure = measure_job(&mut first, nul_title).expect_err("a NUL in a title is refused");
    assert!(failure.description.contains("NUL"));
}
