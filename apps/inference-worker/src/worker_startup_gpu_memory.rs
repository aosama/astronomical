use std::time::Duration;

use tokio::{process::Command, time::timeout};

use crate::worker_startup_error::WorkerStartupError;

const BYTES_PER_MEBIBYTE: u64 = 1024 * 1024;
const IOGPU_WIRED_LIMIT_SYSCTL_KEY: &str = "iogpu.wired_limit_mb";
const SYSCTL_SAMPLE_TIMEOUT: Duration = Duration::from_secs(2);
const SYSCTL_EXECUTABLE_PATH: &str = "/usr/sbin/sysctl";

/// The macOS GPU wired-memory policy reported by `iogpu.wired_limit_mb`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuWiredMemoryLimitSetting {
    /// A positive ceiling explicitly configured in mebibytes through sysctl.
    ExplicitLimitBytes(usize),
    /// The sysctl reported zero, which is a policy sentinel rather than a zero-byte limit.
    SystemDefaultPolicy,
}

pub(crate) struct SampledGpuMemoryLimits {
    pub(crate) machine_mlx_memory_ceiling_bytes: usize,
    pub(crate) recommended_gpu_working_set_bytes: usize,
}

/// The fraction of the recommended working set the default kernel wired policy
/// allows on large-memory machines (issue #1279).
///
/// With `iogpu.wired_limit_mb` at its zero sentinel, macOS enforces a wired
/// boundary in the documented 66–75%-of-memory band that no query exposes,
/// while the MLX recommended working set reports a much larger figure (on the
/// captured 48 GB machine: ~95%). Composing the machine ceiling from the
/// recommendation admits ceilings the operating system silently throttles:
/// decode throughput collapsed 6.5x with active memory pinned at ~75% of the
/// recommendation. The 75% upper edge of the documented band is the honest
/// default-policy estimate; startup wiring verification clamps it further on
/// machines where the enforced boundary sits lower.
pub const DEFAULT_POLICY_WIRED_WORKING_SET_FRACTION: u64 = 3;
pub const DEFAULT_POLICY_WIRED_WORKING_SET_DENOMINATOR: u64 = 4;

/// Parses the machine GPU wired-memory sysctl setting.
pub fn parse_iogpu_wired_limit_setting(
    wired_limit_mebibytes_text: &str,
) -> Result<GpuWiredMemoryLimitSetting, WorkerStartupError> {
    let wired_limit_mebibytes = wired_limit_mebibytes_text
        .trim()
        .parse::<u64>()
        .map_err(|_| WorkerStartupError::InvalidGpuWiredMemoryLimit {
            description: "wired-memory limit is not an unsigned integer",
        })?;
    if wired_limit_mebibytes == 0 {
        return Ok(GpuWiredMemoryLimitSetting::SystemDefaultPolicy);
    }
    let wired_limit_bytes = wired_limit_mebibytes
        .checked_mul(BYTES_PER_MEBIBYTE)
        .ok_or(WorkerStartupError::InvalidGpuWiredMemoryLimit {
            description: "wired-memory limit exceeds the byte range",
        })?;
    let explicit_limit_bytes = usize::try_from(wired_limit_bytes).map_err(|_| {
        WorkerStartupError::InvalidGpuWiredMemoryLimit {
            description: "wired-memory limit exceeds the platform integer range",
        }
    })?;
    Ok(GpuWiredMemoryLimitSetting::ExplicitLimitBytes(
        explicit_limit_bytes,
    ))
}

/// The default-policy estimate of the enforced wired boundary.
#[must_use]
pub fn default_policy_wired_boundary_bytes(recommended_gpu_working_set_bytes: usize) -> usize {
    let bounded_working_set_bytes = recommended_gpu_working_set_bytes as u128
        * DEFAULT_POLICY_WIRED_WORKING_SET_FRACTION as u128;
    usize::try_from(
        bounded_working_set_bytes / DEFAULT_POLICY_WIRED_WORKING_SET_DENOMINATOR as u128,
    )
    .unwrap_or(usize::MAX)
}

/// Derives equal MLX active-memory and allocator-cache limits from the system ceiling.
#[must_use]
pub const fn derive_mlx_memory_limits_from_gpu_wired_limit(
    gpu_wired_memory_limit_bytes: usize,
) -> (usize, usize) {
    (gpu_wired_memory_limit_bytes, gpu_wired_memory_limit_bytes)
}

/// Resolves the machine-specific GPU wired-memory ceiling without changing it.
pub async fn sample_iogpu_wired_limit_bytes() -> Result<usize, WorkerStartupError> {
    Ok(sample_gpu_memory_limits()
        .await?
        .machine_mlx_memory_ceiling_bytes)
}

pub(crate) async fn sample_gpu_memory_limits() -> Result<SampledGpuMemoryLimits, WorkerStartupError>
{
    let mut sysctl_command = Command::new(SYSCTL_EXECUTABLE_PATH);
    sysctl_command
        .arg("-n")
        .arg(IOGPU_WIRED_LIMIT_SYSCTL_KEY)
        .kill_on_drop(true);
    let sysctl_output = timeout(SYSCTL_SAMPLE_TIMEOUT, sysctl_command.output())
        .await
        .map_err(|_| WorkerStartupError::GpuWiredMemoryLimitSampleTimedOut)?
        .map_err(WorkerStartupError::SampleGpuWiredMemoryLimit)?;
    if !sysctl_output.status.success() {
        return Err(WorkerStartupError::GpuWiredMemoryLimitSampleFailed);
    }
    let wired_limit_mebibytes_text = String::from_utf8_lossy(&sysctl_output.stdout);
    let wired_memory_limit_setting =
        parse_iogpu_wired_limit_setting(wired_limit_mebibytes_text.as_ref())?;
    let recommended_gpu_working_set_bytes =
        astronomical_runtime_integration::maximum_recommended_gpu_working_set_size_bytes()
            .map_err(WorkerStartupError::ReadMlxRecommendedGpuWorkingSet)?;
    let machine_mlx_memory_ceiling_bytes = match wired_memory_limit_setting {
        GpuWiredMemoryLimitSetting::ExplicitLimitBytes(explicit_limit_bytes) => {
            explicit_limit_bytes
        }
        // The default policy's enforced wired boundary is not queryable and is
        // materially below the recommendation; estimate it from the documented
        // band and let startup verification prove or clamp the result.
        GpuWiredMemoryLimitSetting::SystemDefaultPolicy => {
            default_policy_wired_boundary_bytes(recommended_gpu_working_set_bytes)
        }
    };
    Ok(SampledGpuMemoryLimits {
        machine_mlx_memory_ceiling_bytes,
        recommended_gpu_working_set_bytes,
    })
}
