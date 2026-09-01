//! Hermetic contracts for the per-worker custom-Metal-kernel capability owner.
//!
//! The owner must turn probe outcomes into verdicts, reuse verdicts without
//! re-probing, and fail closed for families that were never probed. Fake
//! probes count their invocations so the once-per-worker-process contract is
//! observable without a GPU.

mod forced_verdicts;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use astronomical_model_serving::{
    CustomKernelVerdict, CustomMetalKernelFamily, CustomMetalKernelProbe, KernelCapabilityError,
    KernelUnsupportedReason, PerformanceAttribution, WorkerKernelCapabilities,
    validate_probe_outputs,
};

struct CountingProbe {
    family: CustomMetalKernelFamily,
    outcome: Result<(), KernelCapabilityError>,
    invocation_count: Arc<AtomicUsize>,
}

impl CountingProbe {
    fn supported(family: CustomMetalKernelFamily) -> (Self, Arc<AtomicUsize>) {
        Self::with_outcome(family, Ok(()))
    }

    fn with_outcome(
        family: CustomMetalKernelFamily,
        outcome: Result<(), KernelCapabilityError>,
    ) -> (Self, Arc<AtomicUsize>) {
        let invocation_count = Arc::new(AtomicUsize::new(0));
        (
            Self {
                family,
                outcome,
                invocation_count: Arc::clone(&invocation_count),
            },
            invocation_count,
        )
    }
}

impl CustomMetalKernelProbe for CountingProbe {
    fn family(&self) -> CustomMetalKernelFamily {
        self.family
    }

    fn probe(&self) -> Result<(), KernelCapabilityError> {
        self.invocation_count.fetch_add(1, Ordering::SeqCst);
        self.outcome.clone()
    }
}

fn supported_probe_pair(family: CustomMetalKernelFamily) -> (CountingProbe, Arc<AtomicUsize>) {
    CountingProbe::supported(family)
}

#[test]
fn should_probe_every_family_once_and_reuse_verdicts_without_reprobing() {
    let (weighted_sum_probe, weighted_sum_invocations) =
        supported_probe_pair(CustomMetalKernelFamily::SortedExpertWeightedSum);
    let (gated_delta_probe, gated_delta_invocations) =
        supported_probe_pair(CustomMetalKernelFamily::GatedDeltaSequence);
    let mut performance_attribution = PerformanceAttribution::disabled();

    let capabilities = WorkerKernelCapabilities::probe_custom_kernels(
        &[&weighted_sum_probe, &gated_delta_probe],
        &mut performance_attribution,
    );

    assert_eq!(
        capabilities.verdict(CustomMetalKernelFamily::SortedExpertWeightedSum),
        CustomKernelVerdict::Supported
    );
    assert_eq!(
        capabilities.verdict(CustomMetalKernelFamily::GatedDeltaSequence),
        CustomKernelVerdict::Supported
    );
    assert!(
        capabilities.is_custom_kernel_supported(CustomMetalKernelFamily::SortedExpertWeightedSum)
    );

    for _ in 0..10 {
        let _ = capabilities.verdict(CustomMetalKernelFamily::SortedExpertWeightedSum);
    }

    assert_eq!(
        weighted_sum_invocations.load(Ordering::SeqCst),
        1,
        "verdict reads must never re-probe a supported family"
    );
    assert_eq!(
        gated_delta_invocations.load(Ordering::SeqCst),
        1,
        "each family probes exactly once per worker process"
    );
}

#[test]
fn should_report_a_typed_compilation_reason_without_losing_other_families() {
    let (failing_probe, failing_invocations) = CountingProbe::with_outcome(
        CustomMetalKernelFamily::SortedExpertWeightedSum,
        Err(KernelCapabilityError::Compilation {
            description: "the probe kernel source failed to compile".to_owned(),
        }),
    );
    let (supported_probe, _) =
        supported_probe_pair(CustomMetalKernelFamily::TargetVerificationQuantizedLinear);
    let mut performance_attribution = PerformanceAttribution::disabled();

    let capabilities = WorkerKernelCapabilities::probe_custom_kernels(
        &[&failing_probe, &supported_probe],
        &mut performance_attribution,
    );

    assert_eq!(
        capabilities.verdict(CustomMetalKernelFamily::SortedExpertWeightedSum),
        CustomKernelVerdict::Unsupported(KernelUnsupportedReason::Compilation {
            description: "the probe kernel source failed to compile".to_owned(),
        })
    );
    assert!(
        !capabilities.is_custom_kernel_supported(CustomMetalKernelFamily::SortedExpertWeightedSum)
    );
    assert_eq!(
        capabilities.verdict(CustomMetalKernelFamily::TargetVerificationQuantizedLinear),
        CustomKernelVerdict::Supported,
        "one unsupported family must never demote an independently supported family"
    );
    assert_eq!(failing_invocations.load(Ordering::SeqCst), 1);
}

#[test]
fn should_distinguish_execution_failures_from_output_mismatches() {
    let (execution_probe, _) = CountingProbe::with_outcome(
        CustomMetalKernelFamily::GatedDeltaSequence,
        Err(KernelCapabilityError::Execution {
            description: "the bounded probe launch failed".to_owned(),
        }),
    );
    let (mismatch_probe, _) = CountingProbe::with_outcome(
        CustomMetalKernelFamily::TargetVerificationFourRowQuantizedLinear,
        Err(KernelCapabilityError::OutputMismatch {
            description: "probe output value 0 read 0.000000 but expected 133.000000".to_owned(),
        }),
    );
    let mut performance_attribution = PerformanceAttribution::disabled();

    let capabilities = WorkerKernelCapabilities::probe_custom_kernels(
        &[&execution_probe, &mismatch_probe],
        &mut performance_attribution,
    );

    assert_eq!(
        capabilities.verdict(CustomMetalKernelFamily::GatedDeltaSequence),
        CustomKernelVerdict::Unsupported(KernelUnsupportedReason::Execution {
            description: "the bounded probe launch failed".to_owned(),
        })
    );
    assert_eq!(
        capabilities.verdict(CustomMetalKernelFamily::TargetVerificationFourRowQuantizedLinear),
        CustomKernelVerdict::Unsupported(KernelUnsupportedReason::OutputMismatch {
            description: "probe output value 0 read 0.000000 but expected 133.000000".to_owned(),
        })
    );
}

#[test]
fn should_fail_closed_for_a_family_that_was_never_probed() {
    let (weighted_sum_probe, _) =
        supported_probe_pair(CustomMetalKernelFamily::SortedExpertWeightedSum);
    let mut performance_attribution = PerformanceAttribution::disabled();

    let capabilities = WorkerKernelCapabilities::probe_custom_kernels(
        &[&weighted_sum_probe],
        &mut performance_attribution,
    );

    assert_eq!(
        capabilities.verdict(CustomMetalKernelFamily::GatedDeltaBoundaryCheckpoint),
        CustomKernelVerdict::Unsupported(KernelUnsupportedReason::Unprobed),
        "an unprobed family must never be assumed supported"
    );
}

#[test]
fn should_accept_exact_probe_outputs_and_reject_silent_zero_dispatches() {
    let expected_outputs = [133.0, 134.0, 120.0, 121.0];

    assert_eq!(
        validate_probe_outputs(&expected_outputs, &expected_outputs),
        Ok(())
    );
    // A silently dropped Metal dispatch that returns zeros is a known
    // historical failure signature; the probe validator must reject it.
    assert!(matches!(
        validate_probe_outputs(&[0.0, 0.0, 0.0, 0.0], &expected_outputs),
        Err(KernelCapabilityError::OutputMismatch { .. })
    ));
    assert!(matches!(
        validate_probe_outputs(&[133.0, 134.0], &expected_outputs),
        Err(KernelCapabilityError::OutputMismatch { .. })
    ));
}
