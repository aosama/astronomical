//! The counts follow what the caller did, read back from MLX's evaluator.

use mlx_bridge::MlxArray;
use runtime_integration::MlxRuntime;
use serial_test::file_serial;

use super::{EvaluationCounts, evaluation_counts};
use crate::host_step::host_step_tests::test_runtime;

fn lazy_sum(runtime: &MlxRuntime) -> MlxArray {
    let left = MlxArray::from_u32(&[1, 2, 3], &[3]).expect("the left operand uploads");
    let right = MlxArray::from_u32(&[4, 5, 6], &[3]).expect("the right operand uploads");
    runtime.add(&left, &right).expect("the sum joins the graph")
}

#[test]
#[file_serial]
fn should_count_one_blocking_evaluation_for_a_readback_of_a_lazy_array() {
    let runtime = test_runtime();
    let sum = lazy_sum(&runtime);
    let counts_before = evaluation_counts();

    let values = sum.to_vec_u32().expect("the sum reads back");

    let counts = evaluation_counts().since(counts_before);
    assert_eq!(values, vec![5, 7, 9]);
    assert_eq!(counts.blocking_evaluations, 1);
    assert_eq!(counts.asynchronous_evaluations, 0);
    assert_eq!(
        counts.cross_stream_waits, 0,
        "one stream, nothing to wait for"
    );
}

#[test]
#[file_serial]
fn should_count_no_new_evaluation_for_reading_an_evaluated_array_again() {
    let runtime = test_runtime();
    let sum = lazy_sum(&runtime);
    sum.to_vec_u32().expect("the sum reads back");
    let counts_before = evaluation_counts();

    sum.to_vec_u32().expect("the sum reads back again");

    let counts = evaluation_counts().since(counts_before);
    assert_eq!(counts.blocking_evaluations, 0);
    assert_eq!(counts.asynchronous_evaluations, 0);
}

#[test]
fn should_give_zeros_for_readings_taken_out_of_order() {
    let later = EvaluationCounts {
        blocking_evaluations: 9,
        asynchronous_evaluations: 4,
        waits_for_submitted_arrays: 2,
        cross_stream_waits: 80,
        gpu_primitives: 5_000,
        cpu_primitives: 90,
    };
    let earlier = EvaluationCounts {
        blocking_evaluations: 3,
        asynchronous_evaluations: 1,
        waits_for_submitted_arrays: 0,
        cross_stream_waits: 10,
        gpu_primitives: 1_000,
        cpu_primitives: 30,
    };

    assert_eq!(
        later.since(earlier),
        EvaluationCounts {
            blocking_evaluations: 6,
            asynchronous_evaluations: 3,
            waits_for_submitted_arrays: 2,
            cross_stream_waits: 70,
            gpu_primitives: 4_000,
            cpu_primitives: 60,
        }
    );
    assert_eq!(earlier.since(later), EvaluationCounts::default());
}
