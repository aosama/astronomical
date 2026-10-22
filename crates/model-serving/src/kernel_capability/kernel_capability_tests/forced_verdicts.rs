//! Offline contracts for the forced capability-verdict injection seam.
//!
//! CI hardware cannot make a real kernel fail, so the capability owner
//! accepts forced verdicts through this documented test-only constructor.
//! Production code must never call it.

use super::{
    COUNTING_BOUNDARY_CHECKPOINT_FAMILY, COUNTING_EXPERT_DECODE_FAMILY,
    COUNTING_GATED_DELTA_FAMILY, COUNTING_WEIGHTED_SUM_FAMILY,
};
use crate::{CustomKernelVerdict, KernelUnsupportedReason, WorkerKernelCapabilities};

#[test]
fn should_force_an_unsupported_verdict_for_a_single_family() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let capabilities = WorkerKernelCapabilities::with_forced_verdicts_for_tests([(
        COUNTING_WEIGHTED_SUM_FAMILY,
        CustomKernelVerdict::Unsupported(KernelUnsupportedReason::OutputMismatch {
            description: "forced test verdict".to_owned(),
        }),
    )]);

    assert_eq!(
        capabilities.verdict(COUNTING_WEIGHTED_SUM_FAMILY),
        CustomKernelVerdict::Unsupported(KernelUnsupportedReason::OutputMismatch {
            description: "forced test verdict".to_owned(),
        })
    );
    assert_eq!(
        capabilities.verdict(COUNTING_GATED_DELTA_FAMILY),
        CustomKernelVerdict::Unsupported(KernelUnsupportedReason::Unprobed),
        "families absent from the forced set stay fail-closed"
    );
}

#[test]
fn should_force_a_supported_verdict() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let capabilities = WorkerKernelCapabilities::with_forced_verdicts_for_tests([(
        COUNTING_GATED_DELTA_FAMILY,
        CustomKernelVerdict::Supported,
    )]);

    assert!(capabilities.is_custom_kernel_supported(COUNTING_GATED_DELTA_FAMILY));
}

#[test]
fn should_honor_every_forced_reason_for_every_family() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    // Families are data, not tests: one loop proves the fail-closed verdict
    // contract across any family set, so a new family is covered by adding
    // its identifier to this catalogue, not a new test.
    let reason_cases = [
        KernelUnsupportedReason::Compilation {
            description: "forced compilation failure".to_owned(),
        },
        KernelUnsupportedReason::Execution {
            description: "forced execution failure".to_owned(),
        },
        KernelUnsupportedReason::OutputMismatch {
            description: "forced output mismatch".to_owned(),
        },
    ];
    let every_family = [
        COUNTING_WEIGHTED_SUM_FAMILY,
        COUNTING_EXPERT_DECODE_FAMILY,
        COUNTING_GATED_DELTA_FAMILY,
        COUNTING_BOUNDARY_CHECKPOINT_FAMILY,
    ];

    for unsupported_reason in &reason_cases {
        let forced_verdicts = every_family.map(|family| {
            (
                family,
                CustomKernelVerdict::Unsupported(unsupported_reason.clone()),
            )
        });
        let capabilities =
            WorkerKernelCapabilities::with_forced_verdicts_for_tests(forced_verdicts);

        for family in every_family {
            assert_eq!(
                capabilities.verdict(family),
                CustomKernelVerdict::Unsupported(unsupported_reason.clone()),
                "family {family:?} must honor the forced {unsupported_reason:?} verdict",
            );
            assert!(
                !capabilities.is_custom_kernel_supported(family),
                "a forced-unsupported family must never dispatch the custom kernel",
            );
        }
    }
}
