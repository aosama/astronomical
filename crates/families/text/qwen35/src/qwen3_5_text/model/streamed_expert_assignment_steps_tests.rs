//! Beside-source coverage for the pure rotation decisions of a streamed
//! expert chunk. The sorting and wrapping steps build graph arrays and run
//! under the serial engine lane.

use jangtq2_qwen35::configuration::Jangtq2CodebookRotation::{Hadamard32, NoRotation};

use super::{gate_and_up_share_hadamard32_rotation, remaining_gate_up_rotations};

#[test]
fn should_share_one_rotated_copy_when_gate_and_up_are_both_hadamard32() {
    assert!(gate_and_up_share_hadamard32_rotation(
        Hadamard32, Hadamard32
    ));
}

#[test]
fn should_not_share_a_rotated_copy_when_only_one_projection_is_rotated() {
    assert!(!gate_and_up_share_hadamard32_rotation(
        Hadamard32, NoRotation
    ));
    assert!(!gate_and_up_share_hadamard32_rotation(
        NoRotation, Hadamard32
    ));
}

#[test]
fn should_not_share_a_rotated_copy_when_neither_projection_is_rotated() {
    assert!(!gate_and_up_share_hadamard32_rotation(
        NoRotation, NoRotation
    ));
}

#[test]
fn should_apply_no_further_rotation_once_a_shared_copy_exists() {
    assert_eq!(
        remaining_gate_up_rotations(true, Hadamard32, Hadamard32),
        (NoRotation, NoRotation)
    );
}

#[test]
fn should_keep_each_projections_own_rotation_without_a_shared_copy() {
    assert_eq!(
        remaining_gate_up_rotations(false, Hadamard32, NoRotation),
        (Hadamard32, NoRotation)
    );
    assert_eq!(
        remaining_gate_up_rotations(false, NoRotation, NoRotation),
        (NoRotation, NoRotation)
    );
}
