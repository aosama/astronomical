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
