use super::*;

#[test]
fn should_serialize_rejected_generation_outcome_and_failure_description() {
    let performance_attribution_json = serialize_generation_report(
        PerformanceAttribution::enabled(),
        PerformanceAttributionOutcome::Rejected,
    );

    assert_eq!(performance_attribution_json["report_kind"], "generation");
    assert_eq!(performance_attribution_json["outcome"], "rejected");
    assert_eq!(performance_attribution_json["request_id"], 42);
    assert_eq!(
        performance_attribution_json["failure_description"],
        "simulated generation failure"
    );
}

#[test]
fn should_serialize_unobserved_prefill_state_in_model_loading_report() {
    let performance_attribution_json = serialize_model_loading_report(
        PerformanceAttribution::enabled(),
        PerformanceAttributionOutcome::Success,
    );

    assert_eq!(
        performance_attribution_json["prefill_transient_observation_completed"],
        false
    );
    assert_eq!(
        performance_attribution_json["prefill_observed_transient_high_water_bytes"],
        0
    );
}

#[test]
fn should_serialize_prefill_evidence_in_generation_report() {
    let performance_attribution_json = serialize_generation_report(
        PerformanceAttribution::enabled(),
        PerformanceAttributionOutcome::Success,
    );

    assert_eq!(
        performance_attribution_json["prefill_transient_observation_completed"],
        true
    );
    assert_eq!(
        performance_attribution_json["prefill_observed_transient_high_water_bytes"],
        ATTRIBUTED_PREFILL_TRANSIENT_HIGH_WATER_BYTES
    );
}

#[test]
fn should_serialize_process_io_as_an_all_or_unavailable_pair() {
    let performance_attribution_json = serialize_generation_report(
        PerformanceAttribution::enabled(),
        PerformanceAttributionOutcome::Success,
    );

    let physical_disk_read_bytes =
        performance_attribution_json["process_physical_disk_read_bytes"].as_u64();
    let physical_disk_written_bytes =
        performance_attribution_json["process_physical_disk_written_bytes"].as_u64();
    let unavailability_reason =
        performance_attribution_json["process_io_unavailability_reason"].as_str();

    assert_eq!(
        physical_disk_read_bytes.is_some(),
        physical_disk_written_bytes.is_some(),
        "read and write deltas must come from the same process-I/O interval"
    );
    assert_eq!(
        physical_disk_read_bytes.is_some(),
        unavailability_reason.is_none(),
        "available byte deltas and an unavailability reason are mutually exclusive"
    );
}

#[test]
fn should_exclude_outer_diagnostic_spans_from_attributed_elapsed_time() {
    let mut performance_attribution = PerformanceAttribution::enabled();
    performance_attribution.record_completed_operation(
        PerformanceOperation::PromptPrefillAdvanceSpan,
        std::time::Duration::ZERO,
        std::time::Duration::from_secs(10),
    );
    performance_attribution.record_completed_operation(
        PerformanceOperation::MtpPromptHistoryInitializationSpan,
        std::time::Duration::ZERO,
        std::time::Duration::from_secs(5),
    );
    performance_attribution.record_completed_operation(
        PerformanceOperation::PromptTokenization,
        std::time::Duration::ZERO,
        std::time::Duration::from_nanos(7),
    );

    let performance_attribution_json = serialize_generation_report(
        performance_attribution,
        PerformanceAttributionOutcome::Success,
    );

    assert_eq!(
        performance_attribution_json["attributed_elapsed_nanoseconds"],
        7
    );
}
