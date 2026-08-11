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
