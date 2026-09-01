//! Per-worker capability verdicts for Astronomical custom Metal kernels.
//!
//! A capable GPU keeps the measured custom-kernel fast path; a GPU that
//! cannot compile or execute a kernel still serves the same request through
//! the equivalent public MLX API. Verdicts are proven by compile plus a
//! bounded execution probe with expected-value validation — never by a chip
//! name — and are computed once per worker process, then reused by every
//! model swap and REST request without re-probing.

use std::collections::HashMap;
#[cfg(feature = "direct-mlx")]
use std::sync::OnceLock;

use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};

#[cfg(feature = "direct-mlx")]
pub mod gated_delta_probes;
#[cfg(feature = "direct-mlx")]
pub mod sorted_expert_weighted_sum_probe;
#[cfg(feature = "direct-mlx")]
pub mod target_verification_probes;

#[cfg(feature = "direct-mlx")]
pub use gated_delta_probes::{GatedDeltaBoundaryCheckpointProbe, GatedDeltaSequenceProbe};
#[cfg(feature = "direct-mlx")]
pub use sorted_expert_weighted_sum_probe::SortedExpertWeightedSumProbe;
#[cfg(feature = "direct-mlx")]
pub use target_verification_probes::{
    TargetVerificationFourRowProbe, TargetVerificationProjectionProbe,
};

/// One Astronomical custom Metal kernel whose dispatch can fall back to a
/// public MLX API. Kernel sources are fixed constants, so a verdict depends
/// only on the GPU and operating system, never on the loaded model.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CustomMetalKernelFamily {
    /// Sorted mixture-of-experts weighted reduction, shared by Qwen and Laguna.
    SortedExpertWeightedSum,
    /// Fused Qwen3.5 gated-delta sequence recurrence.
    GatedDeltaSequence,
    /// Boundary-checkpoint variant of the fused gated-delta recurrence.
    GatedDeltaBoundaryCheckpoint,
    /// One-row target-verification quantized projection.
    TargetVerificationQuantizedLinear,
    /// Four-row split-K target-verification quantized projection.
    TargetVerificationFourRowQuantizedLinear,
}

/// Why one custom kernel family cannot run on this GPU.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelUnsupportedReason {
    /// The kernel source failed to compile on this device.
    Compilation { description: String },
    /// The bounded representative launch failed to execute.
    Execution { description: String },
    /// The launch executed but produced values that fail expected-value
    /// validation; a silently dropped dispatch returning zeros is a known
    /// historical failure signature on this platform.
    OutputMismatch { description: String },
    /// The family was never probed in this worker process. Fail closed.
    Unprobed,
}

/// The capability verdict for one custom kernel family.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CustomKernelVerdict {
    /// The probe passed; production dispatch may use the custom kernel.
    Supported,
    /// The probe failed or was never run; production dispatch must use the
    /// equivalent public MLX API for the rest of the worker process.
    Unsupported(KernelUnsupportedReason),
}

/// A bounded capability probe for one custom kernel family.
///
/// Implementations compile the kernel and execute a representative launch on
/// minimal inputs, validating the output against fixed deterministic expected
/// values. The runtime binding lives inside each implementation so hermetic
/// tests can drive the capability owner with fake probes.
pub trait CustomMetalKernelProbe {
    fn family(&self) -> CustomMetalKernelFamily;

    /// Runs the bounded probe; `Ok(())` proves this GPU can run the kernel.
    fn probe(&self) -> Result<(), KernelCapabilityError>;
}

/// The failure evidence a probe reports.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelCapabilityError {
    Compilation { description: String },
    Execution { description: String },
    OutputMismatch { description: String },
}

/// Validates probe output values against fixed expected values with the same
/// relative tolerance the direct-MLX contracts use. Deliberately rejects the
/// all-zeros signature of a silently dropped dispatch.
pub fn validate_probe_outputs(
    actual_output_values: &[f32],
    expected_output_values: &[f32],
) -> Result<(), KernelCapabilityError> {
    if actual_output_values.len() != expected_output_values.len() {
        return Err(KernelCapabilityError::OutputMismatch {
            description: format!(
                "probe produced {} output values but {} were expected",
                actual_output_values.len(),
                expected_output_values.len()
            ),
        });
    }
    for (value_index, (actual_value, expected_value)) in actual_output_values
        .iter()
        .zip(expected_output_values.iter())
        .enumerate()
    {
        let comparison_scale = expected_value.abs().max(1.0);
        if (actual_value - expected_value).abs() > 1e-5 * comparison_scale {
            return Err(KernelCapabilityError::OutputMismatch {
                description: format!(
                    "probe output value {value_index} read {actual_value:.6} but expected {expected_value:.6}"
                ),
            });
        }
    }
    Ok(())
}

/// The retained capability verdicts for one worker process.
///
/// Verdicts are computed once at construction — at first model load, when the
/// worker already holds its runtime — and reused by every subsequent request
/// and model swap. Verdicts are never persisted to disk: an operating-system
/// update can change what compiles and runs, so a disk cache is a staleness
/// hazard for a probe cost measured in milliseconds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkerKernelCapabilities {
    verdicts: HashMap<CustomMetalKernelFamily, CustomKernelVerdict>,
}

impl WorkerKernelCapabilities {
    /// Probes every provided family exactly once and retains the verdicts.
    ///
    /// Each probe runs under one attribution operation so the probe cost on
    /// the model-load critical path is measurable.
    pub fn probe_custom_kernels(
        probes: &[&dyn CustomMetalKernelProbe],
        performance_attribution: &mut PerformanceAttribution,
    ) -> Self {
        let mut verdicts = HashMap::with_capacity(probes.len());
        for probe in probes {
            let verdict = performance_attribution.measure_operation(
                PerformanceOperation::CustomKernelCapabilityProbe,
                |_| match probe.probe() {
                    Ok(()) => CustomKernelVerdict::Supported,
                    Err(error) => CustomKernelVerdict::Unsupported(match error {
                        KernelCapabilityError::Compilation { description } => {
                            KernelUnsupportedReason::Compilation { description }
                        }
                        KernelCapabilityError::Execution { description } => {
                            KernelUnsupportedReason::Execution { description }
                        }
                        KernelCapabilityError::OutputMismatch { description } => {
                            KernelUnsupportedReason::OutputMismatch { description }
                        }
                    }),
                },
            );
            verdicts.insert(probe.family(), verdict);
        }
        Self { verdicts }
    }

    /// Returns the retained verdict for one family; unprobed families fail
    /// closed.
    #[must_use]
    pub fn verdict(&self, family: CustomMetalKernelFamily) -> CustomKernelVerdict {
        self.verdicts
            .get(&family)
            .cloned()
            .unwrap_or(CustomKernelVerdict::Unsupported(
                KernelUnsupportedReason::Unprobed,
            ))
    }

    /// Returns whether production dispatch may use the custom kernel.
    #[must_use]
    pub fn is_custom_kernel_supported(&self, family: CustomMetalKernelFamily) -> bool {
        self.verdict(family) == CustomKernelVerdict::Supported
    }

    /// Builds an owner from explicit verdicts. CI hardware cannot make a real
    /// kernel fail, so hermetic and direct-MLX journeys force verdicts through
    /// this documented test-only constructor; production must never call it.
    #[doc(hidden)]
    pub fn with_forced_verdicts_for_tests(
        forced_verdicts: impl IntoIterator<Item = (CustomMetalKernelFamily, CustomKernelVerdict)>,
    ) -> Self {
        Self {
            verdicts: forced_verdicts.into_iter().collect(),
        }
    }
}

/// Returns the retained kernel-capability verdicts for this worker process.
///
/// Verdicts depend only on the GPU and operating system, so the first model
/// load probes every available family once and every later load in the same
/// process — including model hot-swaps — reuses the retained result. The
/// inference worker is one process per machine session, so a process-global
/// owner is the exact once-per-worker lifetime the issue requires; the
/// supervisor never links model-serving. Families without an implemented
/// probe stay unprobed and fail closed until their milestone lands.
#[cfg(feature = "direct-mlx")]
pub fn worker_process_kernel_capabilities(
    runtime: &astronomical_runtime_integration::MlxRuntime,
    performance_attribution: &mut PerformanceAttribution,
) -> &'static WorkerKernelCapabilities {
    static WORKER_PROCESS_KERNEL_CAPABILITIES: OnceLock<WorkerKernelCapabilities> = OnceLock::new();
    WORKER_PROCESS_KERNEL_CAPABILITIES.get_or_init(|| {
        let sorted_expert_weighted_sum_probe = SortedExpertWeightedSumProbe::new(runtime);
        let target_verification_probe = TargetVerificationProjectionProbe::new(runtime);
        let target_verification_four_row_probe = TargetVerificationFourRowProbe::new(runtime);
        let gated_delta_probe = GatedDeltaSequenceProbe::new(runtime);
        let gated_delta_checkpoint_probe = GatedDeltaBoundaryCheckpointProbe::new(runtime);
        WorkerKernelCapabilities::probe_custom_kernels(
            &[
                &sorted_expert_weighted_sum_probe,
                &target_verification_probe,
                &target_verification_four_row_probe,
                &gated_delta_probe,
                &gated_delta_checkpoint_probe,
            ],
            performance_attribution,
        )
    })
}
