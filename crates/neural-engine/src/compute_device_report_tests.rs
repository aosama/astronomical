use super::{ComputeDeviceReport, from_native_report};
use crate::native_bridge::NativeComputeDeviceReport;

#[test]
fn should_recognize_a_neural_engine_with_cores() {
    let report = from_native_report(NativeComputeDeviceReport {
        neural_engine_count: 1,
        neural_engine_core_count: 16,
        gpu_count: 1,
        cpu_count: 1,
    });

    assert!(report.has_neural_engine());
    assert_eq!(report.neural_engine_core_count, 16);
}

#[test]
fn should_not_count_a_neural_engine_without_cores() {
    let report = ComputeDeviceReport {
        neural_engine_count: 1,
        neural_engine_core_count: 0,
        gpu_count: 1,
        cpu_count: 1,
    };

    assert!(!report.has_neural_engine());
}

#[test]
fn should_clamp_negative_native_counts_to_zero() {
    let report = from_native_report(NativeComputeDeviceReport {
        neural_engine_count: -1,
        neural_engine_core_count: -16,
        gpu_count: 0,
        cpu_count: 1,
    });

    assert!(!report.has_neural_engine());
    assert_eq!(report.cpu_count, 1);
}
