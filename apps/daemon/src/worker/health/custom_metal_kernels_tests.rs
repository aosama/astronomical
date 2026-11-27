//! The custom Metal kernel report a daemon publishes for the loaded model.

use std::sync::{Arc, RwLock};

use ipc_protocol::{
    CustomMetalKernelProbeFailureKind, CustomMetalKernelProbeOutcome,
    CustomMetalKernelProbeVerdict, CustomMetalKernelReport, WorkerEvent,
};

use super::health_test_support::{no_capabilities, ready_snapshot};
use super::{WorkerHealthSnapshot, publish_custom_metal_kernels, publish_engine_status_event};

fn report_with_one_refusal() -> CustomMetalKernelReport {
    CustomMetalKernelReport {
        probe_verdicts: vec![
            CustomMetalKernelProbeVerdict {
                family: "gated_delta_sequence".to_owned(),
                outcome: CustomMetalKernelProbeOutcome::Passed,
            },
            CustomMetalKernelProbeVerdict {
                family: "sorted_expert_matmul".to_owned(),
                outcome: CustomMetalKernelProbeOutcome::Failed {
                    kind: CustomMetalKernelProbeFailureKind::Execution,
                    summary: "Unable to load kernel".to_owned(),
                },
            },
        ],
        in_use: vec!["gated_delta_sequence".to_owned()],
    }
}

fn published_report(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
) -> Option<CustomMetalKernelReport> {
    health_snapshot
        .read()
        .expect("the snapshot reads")
        .custom_metal_kernels
        .clone()
}

#[test]
fn should_start_with_no_report_until_a_worker_sends_one() {
    assert_eq!(ready_snapshot().custom_metal_kernels, None);
}

#[test]
fn should_publish_the_report_a_worker_announces() {
    let health_snapshot = Arc::new(RwLock::new(ready_snapshot()));
    publish_engine_status_event(
        &health_snapshot,
        WorkerEvent::CustomMetalKernelsChanged {
            custom_metal_kernels: report_with_one_refusal(),
        },
    );
    assert_eq!(
        published_report(&health_snapshot),
        Some(report_with_one_refusal())
    );
}

#[test]
fn should_clear_the_report_when_a_load_drops_the_model() {
    let health_snapshot = Arc::new(RwLock::new(ready_snapshot()));
    publish_custom_metal_kernels(&health_snapshot, Some(report_with_one_refusal()));
    publish_custom_metal_kernels(&health_snapshot, None);
    assert_eq!(published_report(&health_snapshot), None);
}

#[test]
fn should_not_carry_the_report_into_a_replacement_model() {
    let mut first_model_snapshot = ready_snapshot();
    first_model_snapshot.custom_metal_kernels = Some(report_with_one_refusal());
    let replacement_snapshot = WorkerHealthSnapshot::ready_with_replacement_model(
        String::from("second-model"),
        no_capabilities(),
        1,
        &first_model_snapshot,
    );
    assert_eq!(replacement_snapshot.custom_metal_kernels, None);
}
