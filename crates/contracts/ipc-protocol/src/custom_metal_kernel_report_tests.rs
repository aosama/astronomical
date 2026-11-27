use super::{
    CustomMetalKernelProbeFailureKind, CustomMetalKernelProbeOutcome,
    CustomMetalKernelProbeVerdict, CustomMetalKernelReport,
    PROBE_FAILURE_SUMMARY_MAXIMUM_CHARACTERS, bounded_probe_failure_summary,
};

#[test]
fn should_keep_a_short_one_line_reason_as_it_is() {
    assert_eq!(
        bounded_probe_failure_summary("the kernel gave 3.0 where MLX gave 2.0"),
        "the kernel gave 3.0 where MLX gave 2.0"
    );
}

#[test]
fn should_keep_only_the_first_line_and_drop_the_source_location() {
    let description = "failed to evaluate an MLX array: [metal::Device] Unable to load kernel x\nerror: unsupported body\n at /Users/runner/Library/Caches/mlx_c-src/mlx/c/array.cpp:352";
    assert_eq!(
        bounded_probe_failure_summary(description),
        "failed to evaluate an MLX array: [metal::Device] Unable to load kernel x"
    );
    assert_eq!(
        bounded_probe_failure_summary("refused at /private/var/build/kernel.metal:4"),
        "refused"
    );
}

#[test]
fn should_cap_a_long_reason() {
    let long_reason = "x".repeat(PROBE_FAILURE_SUMMARY_MAXIMUM_CHARACTERS * 3);
    assert_eq!(
        bounded_probe_failure_summary(&long_reason).chars().count(),
        PROBE_FAILURE_SUMMARY_MAXIMUM_CHARACTERS
    );
}

#[test]
fn should_give_an_empty_summary_for_an_empty_reason() {
    assert_eq!(bounded_probe_failure_summary(""), "");
}

#[test]
fn should_round_trip_through_json_with_snake_case_outcomes() {
    let report = CustomMetalKernelReport {
        probe_verdicts: vec![
            CustomMetalKernelProbeVerdict {
                family: "gated_delta_sequence".to_owned(),
                outcome: CustomMetalKernelProbeOutcome::Passed,
            },
            CustomMetalKernelProbeVerdict {
                family: "sorted_expert_matmul".to_owned(),
                outcome: CustomMetalKernelProbeOutcome::Failed {
                    kind: CustomMetalKernelProbeFailureKind::OutputMismatch,
                    summary: "value 3 differs".to_owned(),
                },
            },
        ],
        in_use: vec!["gated_delta_sequence".to_owned()],
    };
    let report_json = serde_json::to_value(&report).expect("the report serializes");
    assert_eq!(
        report_json["probe_verdicts"][0]["outcome"]["result"],
        "passed"
    );
    assert_eq!(
        report_json["probe_verdicts"][1]["outcome"]["kind"],
        "output_mismatch"
    );
    let read_back: CustomMetalKernelReport =
        serde_json::from_value(report_json).expect("the report reads back");
    assert_eq!(read_back, report);
}

#[test]
fn should_refuse_an_unknown_field() {
    let refusal = serde_json::from_str::<CustomMetalKernelReport>(
        r#"{"probe_verdicts":[],"in_use":[],"extra":1}"#,
    );
    assert!(refusal.is_err());
}
