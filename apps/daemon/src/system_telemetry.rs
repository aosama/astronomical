//! Local system telemetry exposed by the daemon.
//!
//! The Swift menu bar app samples GPU utilization directly from IOKit. A
//! local API client can request the same measurement through
//! `GET /v1/system/telemetry`. Each platform supplies its readings through
//! [`SystemTelemetrySource`]: IOKit and the memory-pressure sysctl on macOS,
//! pressure stall information on Linux. Either reading may be unavailable,
//! and the response then carries `null` for it.

use std::future::Future;

use axum::{
    Json, Router,
    response::{IntoResponse, Response},
    routing::get,
};
use serde::Serialize;

#[cfg(target_os = "linux")]
mod linux_system_telemetry;
#[cfg(target_os = "macos")]
mod macos_system_telemetry;
mod memory_pressure_levels;

pub use memory_pressure_levels::{
    parse_linux_memory_pressure_level, parse_macos_memory_pressure_level,
};

#[cfg(target_os = "linux")]
type PlatformSystemTelemetry = linux_system_telemetry::LinuxSystemTelemetry;
#[cfg(target_os = "macos")]
type PlatformSystemTelemetry = macos_system_telemetry::MacOsSystemTelemetry;

/// Where one platform's readings come from. Each reading is optional, so a
/// missing probe never fails the response.
pub(crate) trait SystemTelemetrySource {
    /// Blocking; the route runs it on a blocking thread.
    fn sample_gpu_utilization_percentage() -> Option<f64>;

    /// `normal`, `warning` or `critical`, or `None` when unavailable.
    fn sample_memory_pressure_level() -> impl Future<Output = Option<&'static str>> + Send;
}

/// JSON body returned by `GET /v1/system/telemetry`.
#[derive(Serialize)]
pub(crate) struct SystemTelemetryDocument {
    pub gpu_utilization_percentage: Option<f64>,
    pub memory_pressure: Option<&'static str>,
}

/// `GET /v1/system/telemetry` — samples local GPU utilization and memory
/// pressure without allowing either unavailable probe to fail the response.
pub(crate) async fn sample_telemetry() -> Response {
    let gpu_utilization_percentage = match tokio::task::spawn_blocking(
        PlatformSystemTelemetry::sample_gpu_utilization_percentage,
    )
    .await
    {
        Ok(gpu_utilization_percentage) => gpu_utilization_percentage,
        Err(telemetry_task_error) => {
            tracing::debug!(%telemetry_task_error, "GPU utilization sample task did not complete");
            None
        }
    };
    let memory_pressure = PlatformSystemTelemetry::sample_memory_pressure_level().await;
    Json(SystemTelemetryDocument {
        gpu_utilization_percentage,
        memory_pressure,
    })
    .into_response()
}

/// Returns the system-telemetry route, ready to merge into any daemon
/// `Router`. Keeps `application.rs` from growing past the 500-line principle.
pub(crate) fn system_telemetry_routes<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new().route("/v1/system/telemetry", get(sample_telemetry))
}
