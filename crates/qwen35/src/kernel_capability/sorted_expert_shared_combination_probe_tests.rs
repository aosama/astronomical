//! The probe's verdict rules, plus one real launch on this machine's GPU.

use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

use super::{SortedExpertSharedCombinationProbe, validate_bit_identical_probe_outputs};
use crate::kernel_capability::{CustomMetalKernelProbe, KernelCapabilityError};

const ONE_POINT_FIVE_BITS: u32 = 0x3FC0_0000;
const MINUS_TWO_BITS: u32 = 0xC000_0000;
const POSITIVE_ZERO_BITS: u32 = 0x0000_0000;
const NEGATIVE_ZERO_BITS: u32 = 0x8000_0000;

#[test]
fn should_accept_identical_non_zero_outputs() {
    assert_eq!(
        validate_bit_identical_probe_outputs(
            &[ONE_POINT_FIVE_BITS, MINUS_TWO_BITS],
            &[ONE_POINT_FIVE_BITS, MINUS_TWO_BITS]
        ),
        Ok(())
    );
}

#[test]
fn should_reject_outputs_that_differ_in_one_bit() {
    assert!(matches!(
        validate_bit_identical_probe_outputs(
            &[ONE_POINT_FIVE_BITS, MINUS_TWO_BITS],
            &[ONE_POINT_FIVE_BITS + 1, MINUS_TWO_BITS]
        ),
        Err(KernelCapabilityError::OutputMismatch { .. })
    ));
}

#[test]
fn should_reject_agreeing_zeros_of_two_dropped_dispatches() {
    assert!(matches!(
        validate_bit_identical_probe_outputs(
            &[POSITIVE_ZERO_BITS, NEGATIVE_ZERO_BITS],
            &[POSITIVE_ZERO_BITS, NEGATIVE_ZERO_BITS]
        ),
        Err(KernelCapabilityError::OutputMismatch { .. })
    ));
}

#[test]
fn should_reject_identical_outputs_that_are_not_finite() {
    let positive_infinity_bits = f32::INFINITY.to_bits();
    assert!(matches!(
        validate_bit_identical_probe_outputs(
            &[ONE_POINT_FIVE_BITS, positive_infinity_bits],
            &[ONE_POINT_FIVE_BITS, positive_infinity_bits]
        ),
        Err(KernelCapabilityError::OutputMismatch { .. })
    ));
}

#[test]
fn should_reject_an_empty_probe_output() {
    assert!(matches!(
        validate_bit_identical_probe_outputs(&[], &[]),
        Err(KernelCapabilityError::OutputMismatch { .. })
    ));
}

#[test]
#[serial]
fn should_pass_the_probe_on_this_machines_gpu() {
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the probe test memory limits are valid"),
    )
    .expect("the direct MLX runtime initializes");
    assert_eq!(
        SortedExpertSharedCombinationProbe::new(&runtime).probe(),
        Ok(())
    );
}
