//! Linux system telemetry: memory pressure from the kernel's pressure stall
//! information. Linux builds run MLX on the CPU, so there is no GPU to
//! sample and GPU utilization reads as unavailable.

use super::SystemTelemetrySource;
use super::memory_pressure_levels::parse_linux_memory_pressure_level;

const PRESSURE_STALL_MEMORY_PATH: &str = "/proc/pressure/memory";

/// The Linux telemetry source.
pub(crate) struct LinuxSystemTelemetry;

impl SystemTelemetrySource for LinuxSystemTelemetry {
    fn sample_gpu_utilization_percentage() -> Option<f64> {
        None
    }

    async fn sample_memory_pressure_level() -> Option<&'static str> {
        match tokio::fs::read_to_string(PRESSURE_STALL_MEMORY_PATH).await {
            Ok(pressure_stall_text) => parse_linux_memory_pressure_level(&pressure_stall_text),
            Err(pressure_read_error) => {
                tracing::debug!(%pressure_read_error, "Linux memory pressure probe failed");
                None
            }
        }
    }
}
