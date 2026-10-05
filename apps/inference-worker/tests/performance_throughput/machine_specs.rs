//! Capture of the host specifications that a throughput measurement is taken
//! against, so a historical record can be interpreted against the machine that
//! produced it.

use astronomical_inference_worker::worker_startup;
use serde::Serialize;
use tokio::time::{Duration, timeout};

const SYSCTL_SAMPLE_TIMEOUT: Duration = Duration::from_secs(2);
const SYSCTL_EXECUTABLE_PATH: &str = "/usr/sbin/sysctl";

/// Host specifications recorded alongside each throughput measurement.
///
/// Every field is best-effort: a host that cannot report one of these values
/// simply omits it from the serialized record rather than failing the journey.
#[derive(Clone, Debug, Serialize)]
pub struct MachineSpecs {
    pub os: &'static str,
    pub arch: &'static str,
    pub cpu_cores: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_memory_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_wired_memory_bytes: Option<u64>,
}

impl MachineSpecs {
    /// Captures the host specifications asynchronously. The GPU wired-memory
    /// ceiling is sampled from the machine via sysctl, so the capture cannot
    /// be synchronous.
    pub async fn capture() -> Self {
        let cpu_cores = std::thread::available_parallelism()
            .map(|count| count.get() as u32)
            .unwrap_or(0);
        let cpu_model = run_sysctl("machdep.cpu.brand_string").await;
        let total_memory_bytes = run_sysctl("hw.memsize")
            .await
            .and_then(|value| value.trim().parse::<u64>().ok());
        let gpu_wired_memory_bytes = timeout(
            SYSCTL_SAMPLE_TIMEOUT,
            worker_startup::sample_iogpu_wired_limit_bytes(),
        )
        .await
        .ok()
        .and_then(|result| result.ok())
        .map(|value| value as u64);
        Self {
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            cpu_cores,
            cpu_model,
            total_memory_bytes,
            gpu_wired_memory_bytes,
        }
    }
}

/// Runs a single sysctl read and returns trimmed UTF-8 output, or `None` when
/// the call fails or returns non-UTF-8 output.
async fn run_sysctl(key: &str) -> Option<String> {
    let output = timeout(
        SYSCTL_SAMPLE_TIMEOUT,
        tokio::process::Command::new(SYSCTL_EXECUTABLE_PATH)
            .args(["-n", key])
            .output(),
    )
    .await
    .ok()?
    .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}
