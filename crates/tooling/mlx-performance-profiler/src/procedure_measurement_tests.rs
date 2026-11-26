use mlx_bridge::MlxArray;
use runtime_integration::MlxRuntime;
use serial_test::file_serial;

use crate::measurement_tests::{report_in, runtime_and_matrices, skip_without_metal};
use crate::{MeasuredProcedure, Verdict, compare_procedures};

const STEP_COUNT: usize = 32;

/// A chain of products, evaluated after every step or once at the end: the
/// same arithmetic with many hand-offs or with one.
struct ProductChain<'runtime> {
    name: &'static str,
    runtime: &'runtime MlxRuntime,
    left: &'runtime MlxArray,
    right: &'runtime MlxArray,
    evaluates_every_step: bool,
    prepare_count: usize,
}

impl MeasuredProcedure for ProductChain<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn prepare(&mut self) -> Result<(), String> {
        self.prepare_count += 1;
        Ok(())
    }

    fn run(&mut self) -> Result<(), String> {
        let mut product = self
            .runtime
            .matmul(self.left, self.right)
            .map_err(|matmul_error| matmul_error.to_string())?;
        for _ in 1..STEP_COUNT {
            if self.evaluates_every_step {
                self.runtime
                    .evaluate_arrays(&[&product])
                    .map_err(|evaluation_error| evaluation_error.to_string())?;
            }
            product = self
                .runtime
                .matmul(&product, self.right)
                .map_err(|matmul_error| matmul_error.to_string())?;
        }
        self.runtime
            .evaluate_arrays(&[&product])
            .map_err(|evaluation_error| evaluation_error.to_string())
    }
}

/// A procedure that fails at the step it is told to.
struct FailingProcedure {
    fails_in_prepare: bool,
}

impl MeasuredProcedure for FailingProcedure {
    fn name(&self) -> &str {
        "failing"
    }

    fn prepare(&mut self) -> Result<(), String> {
        if self.fails_in_prepare {
            return Err("this procedure cannot be prepared".to_owned());
        }
        Ok(())
    }

    fn run(&mut self) -> Result<(), String> {
        Err("this procedure cannot run".to_owned())
    }
}

fn product_chain<'runtime>(
    name: &'static str,
    (runtime, left, right): (&'runtime MlxRuntime, &'runtime MlxArray, &'runtime MlxArray),
    evaluates_every_step: bool,
) -> ProductChain<'runtime> {
    ProductChain {
        name,
        runtime,
        left,
        right,
        evaluates_every_step,
        prepare_count: 0,
    }
}

#[test]
#[file_serial]
fn should_count_a_wait_a_step_for_a_chain_evaluated_every_step_and_one_for_a_chain_evaluated_once()
{
    if skip_without_metal("should_count_a_wait_a_step_for_a_chain_evaluated_every_step") {
        return;
    }
    let (runtime, left, right) = runtime_and_matrices();
    let report_directory = tempfile::tempdir().expect("a directory is made");
    let mut every_step = product_chain("every_step", (&runtime, &left, &right), true);
    let mut once = product_chain("once", (&runtime, &left, &right), false);

    let comparison = compare_procedures(
        &mut every_step,
        &mut once,
        report_in(report_directory.path()),
    )
    .expect("the procedures are compared");

    assert_eq!(comparison.duration.round_count, 12);
    assert!(comparison.baseline.median_blocking_wait_count >= STEP_COUNT as f64);
    assert!(comparison.candidate.median_blocking_wait_count <= 2.0);
    assert!(
        comparison.baseline.median_command_buffer_count
            > comparison.candidate.median_command_buffer_count
    );
    assert_ne!(comparison.duration.verdict, Verdict::CandidateSlower);
    // A warm-up, an observed run and twelve rounds, each prepared first.
    assert_eq!(every_step.prepare_count, 14);
    assert!(report_directory.path().join("contract.html").is_file());
    assert!(report_directory.path().join("contract.json").is_file());
}

#[test]
#[file_serial]
fn should_report_a_procedures_own_reason_when_it_cannot_run_or_prepare() {
    if skip_without_metal("should_report_a_procedures_own_reason_when_it_cannot_run_or_prepare") {
        return;
    }
    let (runtime, left, right) = runtime_and_matrices();
    let report_directory = tempfile::tempdir().expect("a directory is made");
    for (fails_in_prepare, expected_reason) in [
        (false, "this procedure cannot run"),
        (true, "this procedure cannot be prepared"),
    ] {
        let mut working = product_chain("working", (&runtime, &left, &right), false);

        let failure = compare_procedures(
            &mut working,
            &mut FailingProcedure { fails_in_prepare },
            report_in(report_directory.path()),
        )
        .expect_err("a procedure that fails is not compared");

        assert_eq!(failure.description, expected_reason);
        assert!(!report_directory.path().join("contract.html").exists());
    }
}

#[test]
#[file_serial]
fn should_refuse_two_procedures_with_one_name() {
    if skip_without_metal("should_refuse_two_procedures_with_one_name") {
        return;
    }
    let (runtime, left, right) = runtime_and_matrices();
    let report_directory = tempfile::tempdir().expect("a directory is made");
    let mut first = product_chain("same", (&runtime, &left, &right), false);
    let mut second = product_chain("same", (&runtime, &left, &right), false);

    assert!(
        compare_procedures(&mut first, &mut second, report_in(report_directory.path())).is_err()
    );
    assert!(!report_directory.path().join("contract.html").exists());
}
