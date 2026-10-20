//! Unit tests for the validated bit widths, group sizes, and the candidate
//! ladder.

use super::*;

#[test]
fn should_accept_the_mlx_affine_widths_and_the_keep_bfloat16_tier() {
    for bits in [2, 3, 4, 5, 6, 8, KEEP_BFLOAT16_BITS] {
        assert!(
            BitWidth::new(bits).is_ok(),
            "{bits} bits should be accepted"
        );
    }
    assert!(BitWidth::new(16).expect("valid").keeps_bfloat16());
    assert!(!BitWidth::new(4).expect("valid").keeps_bfloat16());
}

#[test]
fn should_reject_widths_the_artifact_readers_refuse() {
    for bits in [0, 1, 7, 9, 32] {
        assert!(matches!(
            BitWidth::new(bits),
            Err(QuantizationError::UnsupportedBitWidth { .. })
        ));
    }
}

#[test]
fn should_accept_only_the_three_affine_group_sizes() {
    for group_size in [32, 64, 128] {
        assert!(GroupSize::new(group_size).is_ok());
    }
    assert!(matches!(
        GroupSize::new(16),
        Err(QuantizationError::UnsupportedGroupSize { group_size: 16 })
    ));
}

#[test]
#[ignore = "pending implementation of CandidateBitWidths::new"]
fn should_sort_the_ladder_ascending_and_expose_its_ends() {
    let ladder = CandidateBitWidths::new(&[8, 2, 4]).expect("a valid ladder");

    assert_eq!(ladder.lowest().bits(), 2);
    assert_eq!(ladder.second_lowest().map(BitWidth::bits), Some(4));
    assert_eq!(ladder.highest().bits(), 8);
}

#[test]
#[ignore = "pending implementation of CandidateBitWidths::new"]
fn should_reject_an_empty_or_duplicated_ladder() {
    assert!(matches!(
        CandidateBitWidths::new(&[]),
        Err(QuantizationError::InvalidCandidateBitWidths { .. })
    ));
    assert!(matches!(
        CandidateBitWidths::new(&[4, 4, 8]),
        Err(QuantizationError::InvalidCandidateBitWidths { .. })
    ));
}

#[test]
#[ignore = "pending implementation of CandidateBitWidths::next_above"]
fn should_step_up_the_ladder_and_stop_at_the_top() {
    let ladder = CandidateBitWidths::new(&[2, 4, 8]).expect("a valid ladder");

    assert_eq!(
        ladder.next_above(BitWidth::new(2).expect("valid")),
        Some(BitWidth::new(4).expect("valid"))
    );
    assert_eq!(ladder.next_above(BitWidth::new(8).expect("valid")), None);
    // A width that is not on the ladder has no next step.
    assert_eq!(ladder.next_above(BitWidth::new(3).expect("valid")), None);
}
