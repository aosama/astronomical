//! How the worker reports its custom kernel verdicts: a line in its log as
//! each probe decides, and the probe half of the status report.

use ipc_protocol::{
    CustomMetalKernelProbeFailureKind, CustomMetalKernelProbeOutcome,
    CustomMetalKernelProbeVerdict, bounded_probe_failure_summary,
};

use crate::kernel_capability_registry::{
    CustomKernelVerdict, CustomMetalKernelFamily, KernelUnsupportedReason, WorkerKernelCapabilities,
};

/// A failed probe is written at warning level, which the worker log keeps
/// by default, so a kernel that fell back to MLX is visible without changing
/// the log level; a passed probe is informational.
pub fn log_probe_verdict(family: CustomMetalKernelFamily, verdict: &CustomKernelVerdict) {
    match verdict {
        CustomKernelVerdict::Supported => {
            tracing::info!(
                kernel_family = family.as_str(),
                "custom Metal kernel probe passed; serving may use it"
            );
        }
        CustomKernelVerdict::Unsupported(reason) => {
            tracing::warn!(
                kernel_family = family.as_str(),
                reason = %unsupported_description(reason),
                "custom Metal kernel probe failed; serving uses MLX's own route"
            );
        }
    }
}

/// Every retained verdict as the status report carries it, sorted by family.
#[must_use]
pub fn probe_verdict_report(
    capabilities: &WorkerKernelCapabilities,
) -> Vec<CustomMetalKernelProbeVerdict> {
    let mut probe_verdicts = Vec::new();
    for (family, verdict) in capabilities.verdicts_by_family() {
        probe_verdicts.push(CustomMetalKernelProbeVerdict {
            family: family.as_str().to_owned(),
            outcome: probe_outcome(verdict),
        });
    }
    probe_verdicts
}

fn probe_outcome(verdict: &CustomKernelVerdict) -> CustomMetalKernelProbeOutcome {
    let CustomKernelVerdict::Unsupported(reason) = verdict else {
        return CustomMetalKernelProbeOutcome::Passed;
    };
    let kind = match reason {
        KernelUnsupportedReason::Compilation { .. } => {
            CustomMetalKernelProbeFailureKind::Compilation
        }
        KernelUnsupportedReason::Execution { .. } => CustomMetalKernelProbeFailureKind::Execution,
        KernelUnsupportedReason::OutputMismatch { .. } => {
            CustomMetalKernelProbeFailureKind::OutputMismatch
        }
        KernelUnsupportedReason::Unprobed => CustomMetalKernelProbeFailureKind::NotProbed,
    };
    CustomMetalKernelProbeOutcome::Failed {
        kind,
        summary: bounded_probe_failure_summary(unsupported_description(reason)),
    }
}

fn unsupported_description(reason: &KernelUnsupportedReason) -> &str {
    match reason {
        KernelUnsupportedReason::Compilation { description }
        | KernelUnsupportedReason::Execution { description }
        | KernelUnsupportedReason::OutputMismatch { description } => description,
        KernelUnsupportedReason::Unprobed => "no probe ran for it in this worker process",
    }
}

#[cfg(test)]
#[path = "kernel_capability_report_tests.rs"]
mod kernel_capability_report_tests;
