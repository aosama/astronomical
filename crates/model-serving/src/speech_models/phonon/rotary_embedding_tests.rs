use serial_test::serial;

use super::{rotary_table, rotate_interleaved_pairs};
use crate::speech_models::speech_test_support::{
    assert_close, initialize_test_runtime, read_back, upload,
};

#[test]
fn should_start_the_rotary_table_with_no_rotation() {
    let rotary_table = rotary_table(0, 2, 4, 10_000.0);
    assert_eq!(&rotary_table.cosines[..4], &[1.0; 4]);
    assert_eq!(&rotary_table.sines[..4], &[0.0; 4]);
    assert!(
        (rotary_table.cosines[4] - 1.0f32.cos()).abs() < 1e-6,
        "position 1, pair 0 turns one radian"
    );
}

// Boundary: an offset continues the positions of an earlier call.
#[test]
fn should_continue_positions_from_an_offset() {
    let offset_table = rotary_table(1, 1, 4, 10_000.0);
    let whole_table = rotary_table(0, 2, 4, 10_000.0);
    assert_eq!(offset_table.cosines, whole_table.cosines[4..]);
}

#[test]
#[serial]
fn should_rotate_each_interleaved_pair_by_its_angle() {
    let runtime = initialize_test_runtime();
    // One head, one position (position 1), two pairs: (1, 0) and (0, 1).
    let heads = upload(&runtime, &[1.0, 0.0, 0.0, 1.0], &[1, 1, 4]);
    let rotated = read_back(
        &runtime,
        &rotate_interleaved_pairs(&runtime, &heads, 1, 10_000.0).expect("rotation"),
    );
    let second_pair_angle = 1.0f32 / 100.0;
    let expected = [
        1.0f32.cos(),
        1.0f32.sin(),
        -second_pair_angle.sin(),
        second_pair_angle.cos(),
    ];
    assert_close(&rotated, &expected, 1e-5);
}

// Edge: position 0 leaves every pair unrotated.
#[test]
#[serial]
fn should_leave_position_zero_unrotated() {
    let runtime = initialize_test_runtime();
    let heads = upload(&runtime, &[0.5, -1.5, 2.0, 3.0], &[1, 1, 4]);
    let rotated = read_back(
        &runtime,
        &rotate_interleaved_pairs(&runtime, &heads, 0, 10_000.0).expect("rotation"),
    );
    assert_close(&rotated, &[0.5, -1.5, 2.0, 3.0], 1e-6);
}
