use super::support::generation_metadata;
use super::*;
use astronomical_model_serving::ExpertSourceRequestPhase;
use std::time::Duration;

#[test]
fn should_serialize_bounded_expert_source_demand_by_layer_and_phase() {
    let mut performance_attribution = PerformanceAttribution::enabled();
    performance_attribution.record_expert_source_load(
        7,
        ExpertSourceRequestPhase::Prefill,
        480_000_000,
        12,
    );
    performance_attribution.record_expert_source_load(
        7,
        ExpertSourceRequestPhase::Decode,
        24_000_000,
        3,
    );
    performance_attribution.record_expert_source_resident_hit(
        7,
        ExpertSourceRequestPhase::Decode,
        12_000_000,
        2,
    );

    let performance_attribution_json = serialize_generation_report(
        performance_attribution,
        PerformanceAttributionOutcome::Success,
    );
    let layer_reports = performance_attribution_json["expert_source_by_layer"]
        .as_array()
        .expect("generation attribution should contain bounded per-layer source evidence");

    assert_eq!(layer_reports.len(), 2);
    assert_eq!(layer_reports[0]["layer_index"], 7);
    assert_eq!(layer_reports[0]["request_phase"], "prefill");
    assert_eq!(
        layer_reports[0]["logical_source_payload_bytes"],
        480_000_000
    );
    assert_eq!(
        layer_reports[0]["maximum_source_page_payload_bytes"],
        480_000_000
    );
    assert_eq!(layer_reports[0]["source_interval_count"], 12);
    assert_eq!(layer_reports[0]["source_load_count"], 1);
    assert_eq!(layer_reports[1]["request_phase"], "decode");
    assert_eq!(layer_reports[1]["logical_source_payload_bytes"], 24_000_000);
    assert_eq!(layer_reports[1]["resident_hit_count"], 1);
    assert_eq!(layer_reports[1]["avoided_source_payload_bytes"], 12_000_000);
    assert_eq!(layer_reports[1]["avoided_source_interval_count"], 2);
}

#[test]
fn should_saturate_expert_source_evidence_without_growing_per_event_state() {
    let mut performance_attribution = PerformanceAttribution::enabled();
    performance_attribution.record_expert_source_load(
        2,
        ExpertSourceRequestPhase::RetentionTransition,
        u64::MAX,
        u64::MAX,
    );
    performance_attribution.record_expert_source_load(
        2,
        ExpertSourceRequestPhase::RetentionTransition,
        1,
        1,
    );

    let performance_attribution_json = serialize_generation_report(
        performance_attribution,
        PerformanceAttributionOutcome::Success,
    );
    let transition_report = &performance_attribution_json["expert_source_by_layer"][0];

    assert_eq!(transition_report["logical_source_payload_bytes"], u64::MAX);
    assert_eq!(
        transition_report["maximum_source_page_payload_bytes"],
        u64::MAX
    );
    assert_eq!(transition_report["source_interval_count"], u64::MAX);
    assert_eq!(transition_report["source_load_count"], 2);
}

#[test]
fn should_keep_disabled_expert_source_attribution_inert() {
    let mut performance_attribution = PerformanceAttribution::disabled();

    performance_attribution.record_expert_source_load(
        usize::MAX,
        ExpertSourceRequestPhase::Prefill,
        u64::MAX,
        u64::MAX,
    );
    performance_attribution.record_expert_source_resident_hit(
        usize::MAX,
        ExpertSourceRequestPhase::Decode,
        u64::MAX,
        u64::MAX,
    );

    assert!(!performance_attribution.is_enabled());
    assert!(
        performance_attribution
            .finish_generation(generation_metadata(PerformanceAttributionOutcome::Success))
            .is_none(),
        "disabled attribution must not allocate or publish per-layer evidence"
    );
}

#[test]
fn should_serialize_page_readiness_wait_and_failure_by_layer_and_phase() {
    let mut performance_attribution = PerformanceAttribution::enabled();
    performance_attribution.record_expert_page_readiness_wait(
        11,
        ExpertSourceRequestPhase::Decode,
        Duration::from_nanos(1_200),
        true,
    );
    performance_attribution.record_expert_page_readiness_wait(
        11,
        ExpertSourceRequestPhase::Decode,
        Duration::from_nanos(800),
        false,
    );

    let performance_attribution_json = serialize_generation_report(
        performance_attribution,
        PerformanceAttributionOutcome::Failed,
    );
    let decode_report = &performance_attribution_json["expert_source_by_layer"][0];

    assert_eq!(decode_report["layer_index"], 11);
    assert_eq!(decode_report["request_phase"], "decode");
    assert_eq!(decode_report["page_readiness_wait_count"], 2);
    assert_eq!(decode_report["page_readiness_wait_nanoseconds"], 2_000);
    assert_eq!(
        decode_report["maximum_page_readiness_wait_nanoseconds"],
        1_200
    );
    assert_eq!(decode_report["page_readiness_failure_count"], 1);
}

#[test]
fn should_execute_readiness_work_without_layer_state_when_disabled() {
    let mut performance_attribution = PerformanceAttribution::disabled();
    let readiness_outcome = performance_attribution.measure_expert_page_readiness(
        usize::MAX,
        ExpertSourceRequestPhase::Prefill,
        || Ok::<_, &'static str>("ready"),
    );

    assert_eq!(readiness_outcome, Ok("ready"));
    assert!(
        performance_attribution
            .finish_generation(generation_metadata(PerformanceAttributionOutcome::Success))
            .is_none()
    );
}
