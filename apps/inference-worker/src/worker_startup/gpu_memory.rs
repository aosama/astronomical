//! The machine MLX memory-ceiling sampling: physical RAM read from the
//! host, resolved into the ceiling by the model-serving memory policy.
//!
//! One number bounds everything a user may dedicate to model serving, and
//! it is `hw.memsize` itself: Astronomical keeps no operating-system reserve
//! of its own and leaves memory pressure to MLX and macOS (#1902). No
//! graphics-memory policy value from the kernel and no device recommendation
//! participate in the bound,
//! so the daemon, the worker, and the menu share a single source of truth.
//! `tests/gpu/wired_limit_claim_probe.rs` measures how much a healthy
//! machine can actually wire. On machines where Metal's
//! recommended working set sits below this ceiling, the worker's wired
//! claim is capped at the recommendation and the effective serving ceiling
//! is clamped to that budget — the ceiling stays the upper bound while the
//! wire budget decides what is actually servable without paging the model.

#[cfg(target_os = "macos")]
use std::time::Duration;

use model_serving::machine_ceiling_from_physical_ram_bytes;
#[cfg(target_os = "macos")]
use tokio::{process::Command, time::timeout};

use super::error::WorkerStartupError;

#[cfg(target_os = "macos")]
const PHYSICAL_MEMORY_SIZE_SYSCTL_KEY: &str = "hw.memsize";
#[cfg(target_os = "macos")]
const SYSCTL_SAMPLE_TIMEOUT: Duration = Duration::from_secs(2);
#[cfg(target_os = "macos")]
const SYSCTL_EXECUTABLE_PATH: &str = "/usr/sbin/sysctl";
#[cfg(target_os = "linux")]
const LINUX_MEMINFO_PATH: &str = "/proc/meminfo";

/// Samples the machine MLX memory ceiling from the physical memory size.
pub async fn machine_mlx_memory_ceiling_bytes() -> Result<usize, WorkerStartupError> {
    let physical_ram_bytes = sample_physical_ram_bytes().await?;
    let machine_ceiling_bytes = machine_ceiling_from_physical_ram_bytes(physical_ram_bytes);
    usize::try_from(machine_ceiling_bytes).map_err(|_| {
        WorkerStartupError::InvalidPhysicalMemorySize {
            description: "machine MLX memory ceiling exceeds the platform integer range",
        }
    })
}

/// `hw.memsize`, read through `sysctl`.
#[cfg(target_os = "macos")]
async fn sample_physical_ram_bytes() -> Result<u64, WorkerStartupError> {
    let mut sysctl_command = Command::new(SYSCTL_EXECUTABLE_PATH);
    sysctl_command
        .arg("-n")
        .arg(PHYSICAL_MEMORY_SIZE_SYSCTL_KEY)
        .kill_on_drop(true);
    let sysctl_output = timeout(SYSCTL_SAMPLE_TIMEOUT, sysctl_command.output())
        .await
        .map_err(|_| WorkerStartupError::PhysicalMemorySizeSampleTimedOut)?
        .map_err(WorkerStartupError::SamplePhysicalMemorySize)?;
    if !sysctl_output.status.success() {
        return Err(WorkerStartupError::PhysicalMemorySizeSampleFailed);
    }
    let physical_ram_bytes_text = String::from_utf8_lossy(&sysctl_output.stdout);
    physical_ram_bytes_text.trim().parse::<u64>().map_err(|_| {
        WorkerStartupError::InvalidPhysicalMemorySize {
            description: "hw.memsize is not an unsigned integer",
        }
    })
}

/// `MemTotal` from `/proc/meminfo`, Linux's physical memory size.
#[cfg(target_os = "linux")]
async fn sample_physical_ram_bytes() -> Result<u64, WorkerStartupError> {
    let meminfo_text = tokio::fs::read_to_string(LINUX_MEMINFO_PATH)
        .await
        .map_err(WorkerStartupError::ReadLinuxMeminfo)?;
    memory_policy::serving_ceiling::physical_ram_bytes_from_linux_meminfo(&meminfo_text).ok_or(
        WorkerStartupError::InvalidPhysicalMemorySize {
            description: "/proc/meminfo has no readable MemTotal line",
        },
    )
}
