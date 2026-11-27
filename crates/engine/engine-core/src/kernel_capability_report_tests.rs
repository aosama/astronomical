use ipc_protocol::{CustomMetalKernelProbeFailureKind, CustomMetalKernelProbeOutcome};

use super::probe_verdict_report;
use crate::kernel_capability_registry::{
    CustomKernelVerdict, CustomMetalKernelFamily, KernelUnsupportedReason, WorkerKernelCapabilities,
};

#[test]
fn should_report_every_verdict_sorted_by_family_with_a_bounded_reason() {
    let capabilities = WorkerKernelCapabilities::with_forced_verdicts_for_tests([
        (CustomMetalKernelFamily::new("sorted_expert_matmul"), CustomKernelVerdict::Unsupported(KernelUnsupportedReason::Execution {
            description: "Unable to load kernel\nerror: unsupported body\n at /Users/runner/array.cpp:352".to_owned(),
        })),
        (CustomMetalKernelFamily::new("gated_delta_sequence"), CustomKernelVerdict::Supported),
    ]);
    let probe_verdicts = probe_verdict_report(&capabilities);
    assert_eq!(probe_verdicts.len(), 2);
    assert_eq!(probe_verdicts[0].family, "gated_delta_sequence");
    assert_eq!(
        probe_verdicts[0].outcome,
        CustomMetalKernelProbeOutcome::Passed
    );
    assert_eq!(
        probe_verdicts[1].outcome,
        CustomMetalKernelProbeOutcome::Failed {
            kind: CustomMetalKernelProbeFailureKind::Execution,
            summary: "Unable to load kernel".to_owned(),
        }
    );
}

#[test]
fn should_report_a_family_left_unprobed_as_not_probed() {
    let capabilities = WorkerKernelCapabilities::with_forced_verdicts_for_tests([(
        CustomMetalKernelFamily::new("row_batched_matmul"),
        CustomKernelVerdict::Unsupported(KernelUnsupportedReason::Unprobed),
    )]);
    let CustomMetalKernelProbeOutcome::Failed { kind, .. } =
        &probe_verdict_report(&capabilities)[0].outcome
    else {
        panic!("an unprobed family is not a passed one");
    };
    assert_eq!(*kind, CustomMetalKernelProbeFailureKind::NotProbed);
}

#[test]
fn should_report_nothing_for_a_worker_without_verdicts() {
    let capabilities = WorkerKernelCapabilities::with_forced_verdicts_for_tests([]);
    assert!(probe_verdict_report(&capabilities).is_empty());
}

#[test]
fn should_report_an_output_mismatch_and_a_compile_failure_by_their_kinds() {
    let capabilities = WorkerKernelCapabilities::with_forced_verdicts_for_tests([
        (
            CustomMetalKernelFamily::new("a_family"),
            CustomKernelVerdict::Unsupported(KernelUnsupportedReason::Compilation {
                description: "no matching function".to_owned(),
            }),
        ),
        (
            CustomMetalKernelFamily::new("b_family"),
            CustomKernelVerdict::Unsupported(KernelUnsupportedReason::OutputMismatch {
                description: "value 3 differs".to_owned(),
            }),
        ),
    ]);
    let probe_verdicts = probe_verdict_report(&capabilities);
    assert!(matches!(
        probe_verdicts[0].outcome,
        CustomMetalKernelProbeOutcome::Failed {
            kind: CustomMetalKernelProbeFailureKind::Compilation,
            ..
        }
    ));
    assert!(matches!(
        probe_verdicts[1].outcome,
        CustomMetalKernelProbeOutcome::Failed {
            kind: CustomMetalKernelProbeFailureKind::OutputMismatch,
            ..
        }
    ));
}
