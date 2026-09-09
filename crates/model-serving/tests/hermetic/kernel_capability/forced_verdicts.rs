//! Hermetic contracts for the forced capability-verdict injection seam.
//!
//! CI hardware cannot make a real kernel fail, so the capability owner
//! accepts forced verdicts through this documented test-only constructor.
//! Production code must never call it.

use astronomical_model_serving::{
    CustomKernelVerdict, CustomMetalKernelFamily, KernelUnsupportedReason, WorkerKernelCapabilities,
};

#[test]
fn should_force_an_unsupported_verdict_for_a_single_family() {
    let capabilities = WorkerKernelCapabilities::with_forced_verdicts_for_tests([(
        CustomMetalKernelFamily::SortedExpertWeightedSum,
        CustomKernelVerdict::Unsupported(KernelUnsupportedReason::OutputMismatch {
            description: "forced test verdict".to_owned(),
        }),
    )]);

    assert_eq!(
        capabilities.verdict(CustomMetalKernelFamily::SortedExpertWeightedSum),
        CustomKernelVerdict::Unsupported(KernelUnsupportedReason::OutputMismatch {
            description: "forced test verdict".to_owned(),
        })
    );
    assert_eq!(
        capabilities.verdict(CustomMetalKernelFamily::GatedDeltaSequence),
        CustomKernelVerdict::Unsupported(KernelUnsupportedReason::Unprobed),
        "families absent from the forced set stay fail-closed"
    );
}

#[test]
fn should_force_a_supported_verdict() {
    let capabilities = WorkerKernelCapabilities::with_forced_verdicts_for_tests([(
        CustomMetalKernelFamily::GatedDeltaSequence,
        CustomKernelVerdict::Supported,
    )]);

    assert!(capabilities.is_custom_kernel_supported(CustomMetalKernelFamily::GatedDeltaSequence));
}

#[test]
fn should_honor_every_forced_reason_for_every_family() {
    // Families are data, not tests: one loop proves the fail-closed verdict
    // contract across the whole family catalogue, so a new family is covered
    // by adding an enum variant, not a new test.
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
        CustomMetalKernelFamily::SortedExpertWeightedSum,
        CustomMetalKernelFamily::FusedQuantizedExpertDecode,
        CustomMetalKernelFamily::GatedDeltaSequence,
        CustomMetalKernelFamily::GatedDeltaBoundaryCheckpoint,
        CustomMetalKernelFamily::TargetVerificationQuantizedLinear,
        CustomMetalKernelFamily::TargetVerificationFourRowQuantizedLinear,
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
