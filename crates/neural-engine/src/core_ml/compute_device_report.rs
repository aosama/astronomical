//! Which compute devices Core ML can see.

use crate::core_ml::ffi::{NativeComputeDeviceReport, astronomical_core_ml_report_compute_devices};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComputeDeviceReport {
    pub neural_engine_count: usize,
    pub neural_engine_core_count: usize,
    pub gpu_count: usize,
    pub cpu_count: usize,
}

impl ComputeDeviceReport {
    #[must_use]
    pub fn has_neural_engine(&self) -> bool {
        self.neural_engine_count > 0 && self.neural_engine_core_count > 0
    }
}

/// Asks Core ML (`MLAllComputeDevices`) for every device it can schedule on.
#[must_use]
pub fn report_compute_devices() -> ComputeDeviceReport {
    let mut native_report = NativeComputeDeviceReport::default();
    // SAFETY: the bridge writes one plain struct through a valid, exclusive
    // pointer and keeps no reference to it.
    unsafe {
        astronomical_core_ml_report_compute_devices(&mut native_report);
    }
    from_native_report(native_report)
}

fn from_native_report(native_report: NativeComputeDeviceReport) -> ComputeDeviceReport {
    ComputeDeviceReport {
        neural_engine_count: usize::try_from(native_report.neural_engine_count).unwrap_or(0),
        neural_engine_core_count: usize::try_from(native_report.neural_engine_core_count)
            .unwrap_or(0),
        gpu_count: usize::try_from(native_report.gpu_count).unwrap_or(0),
        cpu_count: usize::try_from(native_report.cpu_count).unwrap_or(0),
    }
}

#[cfg(test)]
#[path = "compute_device_report_tests.rs"]
mod tests;
