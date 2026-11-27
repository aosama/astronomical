//! What this Mac's GPU proved about Astronomical's own Metal kernels, and
//! which of them the loaded model holds.

use serde::{Deserialize, Serialize};

/// Reported by the engine once a load settles, so a person or a tool can
/// tell from the status page, without reading the worker log, whether a
/// custom kernel serves on this Mac or the model fell back to MLX's own.
///
/// Two facts, kept apart on purpose: a probe that passed only says the GPU
/// runs the kernel exactly; the model holds it only where its shape and
/// weights take it (a dense model never holds a mixture-of-experts kernel).
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CustomMetalKernelReport {
    /// One verdict per probed kernel family, in family order. The probes
    /// run once per worker process, at its first model load.
    pub probe_verdicts: Vec<CustomMetalKernelProbeVerdict>,
    /// The kernels the loaded model holds, by stable use id, sorted.
    pub in_use: Vec<String>,
}

/// One kernel family's probe outcome.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CustomMetalKernelProbeVerdict {
    /// The probe family id, such as `gated_delta_sequence`.
    pub family: String,
    pub outcome: CustomMetalKernelProbeOutcome,
}

/// Whether the probe proved the kernel on this GPU.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "result", rename_all = "snake_case", deny_unknown_fields)]
pub enum CustomMetalKernelProbeOutcome {
    /// The kernel gave its reference's values; serving may use it.
    Passed,
    /// The kernel did not; serving uses MLX's own route. `summary` is the
    /// first line of the reason, bounded and without local paths; the worker
    /// log holds the whole reason.
    Failed {
        kind: CustomMetalKernelProbeFailureKind,
        summary: String,
    },
}

/// The step at which a probe failed.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CustomMetalKernelProbeFailureKind {
    /// The kernel did not compile.
    Compilation,
    /// The GPU could not build or run it.
    Execution,
    /// It ran but its values differ from its reference.
    OutputMismatch,
    /// No probe ran for it in this worker process, so it is not used.
    NotProbed,
}

/// The longest probe failure summary the status page carries.
pub const PROBE_FAILURE_SUMMARY_MAXIMUM_CHARACTERS: usize = 160;

/// The first line of a probe failure's description, cut before any source
/// location (`" at /…"`, which names a path on the machine that built MLX)
/// and capped at [`PROBE_FAILURE_SUMMARY_MAXIMUM_CHARACTERS`].
#[must_use]
pub fn bounded_probe_failure_summary(description: &str) -> String {
    let first_line = description.lines().next().unwrap_or_default();
    let without_location = first_line
        .split_once(" at /")
        .map_or(first_line, |(before_location, _)| before_location)
        .trim();
    without_location
        .chars()
        .take(PROBE_FAILURE_SUMMARY_MAXIMUM_CHARACTERS)
        .collect()
}

#[cfg(test)]
#[path = "custom_metal_kernel_report_tests.rs"]
mod custom_metal_kernel_report_tests;
