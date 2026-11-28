//! Which tiled sorted-expert kernel widths a load keeps, from forced probe
//! verdicts: declaring a kernel builds no GPU work, so no device runs here.

use crate::kernel_capability::{
    CustomKernelVerdict, KernelUnsupportedReason, SORTED_EXPERT_TILED_MATMUL,
    SORTED_EXPERT_TILED_MATMUL_6BIT, WorkerKernelCapabilities,
};

use super::kept_sorted_expert_tiled;
use crate::sorted_expert_matmul::TiledExpertKernels;

fn verdicts(four_bit_passed: bool, six_bit_passed: bool) -> WorkerKernelCapabilities {
    WorkerKernelCapabilities::with_forced_verdicts_for_tests([
        (SORTED_EXPERT_TILED_MATMUL, verdict_of(four_bit_passed)),
        (SORTED_EXPERT_TILED_MATMUL_6BIT, verdict_of(six_bit_passed)),
    ])
}

fn verdict_of(passed: bool) -> CustomKernelVerdict {
    if passed {
        CustomKernelVerdict::Supported
    } else {
        CustomKernelVerdict::Unsupported(KernelUnsupportedReason::OutputMismatch {
            description: "forced for the test".to_owned(),
        })
    }
}

fn kept_widths(four_bit_passed: bool, six_bit_passed: bool) -> Option<Vec<i32>> {
    kept_sorted_expert_tiled(&verdicts(four_bit_passed, six_bit_passed), true)
        .as_ref()
        .map(TiledExpertKernels::weight_bits)
}

#[test]
fn should_keep_both_widths_when_both_probes_passed() {
    assert_eq!(kept_widths(true, true), Some(vec![4, 6]));
}

#[test]
fn should_keep_the_4bit_kernels_when_only_the_6bit_probe_failed() {
    assert_eq!(kept_widths(true, false), Some(vec![4]));
}

#[test]
fn should_keep_the_6bit_kernels_when_only_the_4bit_probe_failed() {
    assert_eq!(kept_widths(false, true), Some(vec![6]));
}

#[test]
fn should_keep_no_tiled_kernels_when_no_probe_passed() {
    assert_eq!(kept_widths(false, false), None);
}

#[test]
fn should_keep_no_tiled_kernels_without_the_sorted_route_they_read() {
    assert!(kept_sorted_expert_tiled(&verdicts(true, true), false).is_none());
}
