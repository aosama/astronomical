use model_primitives::synthetic_weights_test_support::initialize_test_runtime as test_runtime;
use serial_test::file_serial;

use super::{GrowingSequence, SEQUENCE_GROWTH_ROWS, reserved_row_count};

#[test]
fn should_reserve_whole_steps() {
    assert_eq!(reserved_row_count(1), SEQUENCE_GROWTH_ROWS);
    assert_eq!(
        reserved_row_count(SEQUENCE_GROWTH_ROWS),
        SEQUENCE_GROWTH_ROWS
    );
    assert_eq!(
        reserved_row_count(SEQUENCE_GROWTH_ROWS + 1),
        2 * SEQUENCE_GROWTH_ROWS
    );
}

#[test]
fn should_reserve_one_step_for_an_empty_start() {
    assert_eq!(reserved_row_count(0), SEQUENCE_GROWTH_ROWS);
}

#[test]
#[file_serial]
fn should_keep_rows_in_order_across_a_step_boundary() {
    let runtime = test_runtime();
    let first = runtime
        .array_from_f32(&[1.0, 2.0], &[1, 2, 1])
        .expect("first rows");
    let mut sequence = GrowingSequence::starting_with(&runtime, &first, 1).expect("start");
    let storage_before = sequence.storage().shape();
    let filler = vec![5.0_f32; (SEQUENCE_GROWTH_ROWS - 2) as usize];
    let filler = runtime
        .array_from_f32(&filler, &[1, SEQUENCE_GROWTH_ROWS - 2, 1])
        .expect("filler");
    sequence
        .append(&runtime, &filler)
        .expect("fills the first step");
    assert_eq!(
        sequence.storage().shape(),
        storage_before,
        "no new step yet"
    );
    let last = runtime.array_from_f32(&[9.0], &[1, 1, 1]).expect("last");
    sequence
        .append(&runtime, &last)
        .expect("crosses into the next step");
    assert_eq!(sequence.used_rows(), SEQUENCE_GROWTH_ROWS + 1);
    assert_eq!(sequence.storage().shape()[1], 2 * SEQUENCE_GROWTH_ROWS);
    let used = sequence
        .used(&runtime)
        .expect("used")
        .to_vec_f32()
        .expect("values");
    assert_eq!(used[0], 1.0);
    assert_eq!(used[1], 2.0);
    assert_eq!(used[2], 5.0);
    assert_eq!(*used.last().expect("one row at least"), 9.0);
}
