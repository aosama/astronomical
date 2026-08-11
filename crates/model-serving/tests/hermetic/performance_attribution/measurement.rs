use super::*;

#[test]
fn should_keep_the_disabled_attribution_handle_pointer_sized() {
    assert!(
        std::mem::size_of::<PerformanceAttribution>() <= std::mem::size_of::<usize>(),
        "disabled attribution travels through every engine command and must not enlarge the command queue"
    );
}

#[test]
fn should_execute_measured_work_without_recording_when_attribution_is_disabled() {
    let mut performance_attribution = PerformanceAttribution::disabled();

    let operation_output = performance_attribution.measure_operation(
        PerformanceOperation::PromptTokenization,
        |_performance_attribution| 42_u32,
    );

    assert_eq!(operation_output, 42);
    assert!(
        performance_attribution
            .operation_measurement(PerformanceOperation::PromptTokenization)
            .is_none()
    );
}

#[test]
fn should_record_an_error_returning_operation() {
    let mut performance_attribution = PerformanceAttribution::enabled();

    let operation_outcome: Result<(), &'static str> = performance_attribution.measure_operation(
        PerformanceOperation::PersistentPromptCacheOpenAndScan,
        |_performance_attribution| Err("disk read failed"),
    );

    assert_eq!(operation_outcome, Err("disk read failed"));
    assert_eq!(
        performance_attribution
            .operation_measurement(PerformanceOperation::PersistentPromptCacheOpenAndScan)
            .map(|operation_measurement| operation_measurement.occurrence_count()),
        Some(1)
    );
}

#[test]
fn should_aggregate_repeated_operation_measurements() {
    let mut performance_attribution = PerformanceAttribution::enabled();
    performance_attribution.record_completed_operation(
        PerformanceOperation::PromptTokenization,
        std::time::Duration::from_nanos(5),
        std::time::Duration::from_nanos(25),
    );
    performance_attribution.record_completed_operation(
        PerformanceOperation::PromptTokenization,
        std::time::Duration::from_nanos(50),
        std::time::Duration::from_nanos(100),
    );

    let operation_measurement = performance_attribution
        .operation_measurement(PerformanceOperation::PromptTokenization)
        .expect("two recorded operations should have an aggregate");

    assert_eq!(operation_measurement.occurrence_count(), 2);
    assert_eq!(operation_measurement.total_elapsed_nanoseconds(), 70);
    assert_eq!(operation_measurement.minimum_elapsed_nanoseconds(), 20);
    assert_eq!(operation_measurement.maximum_elapsed_nanoseconds(), 50);
    assert_eq!(operation_measurement.first_started_offset_nanoseconds(), 5);
    assert_eq!(operation_measurement.last_ended_offset_nanoseconds(), 100);
}

#[test]
fn should_saturate_repeated_operation_elapsed_time() {
    let mut performance_attribution = PerformanceAttribution::enabled();
    let maximum_elapsed_duration = std::time::Duration::from_nanos(u64::MAX);
    for _operation_occurrence_index in 0..2 {
        performance_attribution.record_completed_operation(
            PerformanceOperation::PromptRendering,
            std::time::Duration::ZERO,
            maximum_elapsed_duration,
        );
    }

    let operation_measurement = performance_attribution
        .operation_measurement(PerformanceOperation::PromptRendering)
        .expect("saturated operations should still retain an aggregate");

    assert_eq!(operation_measurement.total_elapsed_nanoseconds(), u64::MAX);
    assert_eq!(
        operation_measurement.maximum_elapsed_nanoseconds(),
        u64::MAX
    );
}

#[test]
fn should_aggregate_performance_counters_with_saturation() {
    let mut performance_attribution = PerformanceAttribution::enabled();
    performance_attribution.record_counter(PerformanceCounter::GeneratedTokenCount, u64::MAX);
    performance_attribution.record_counter(PerformanceCounter::GeneratedTokenCount, 1);

    assert_eq!(
        performance_attribution.counter_value(PerformanceCounter::GeneratedTokenCount),
        u64::MAX
    );
}

#[test]
fn should_retain_the_largest_maximum_counter_observation() {
    let mut performance_attribution = PerformanceAttribution::enabled();
    let maximum_counter =
        PerformanceCounter::NativeExpertCacheMaximumRouteDependencySynchronizationElapsedNanoseconds;
    performance_attribution.record_maximum_counter(maximum_counter, 40);
    performance_attribution.record_maximum_counter(maximum_counter, 10);
    performance_attribution.record_maximum_counter(maximum_counter, 80);

    assert_eq!(performance_attribution.counter_value(maximum_counter), 80);
}

#[test]
fn should_saturate_each_mtp_outcome_counter_independently() {
    let mut performance_attribution = PerformanceAttribution::enabled();

    for performance_counter in [
        PerformanceCounter::MtpAdmittedAttemptCount,
        PerformanceCounter::MtpAcceptedDraftCount,
        PerformanceCounter::MtpRejectedDraftCount,
        PerformanceCounter::MtpOperationalFallbackCount,
    ] {
        performance_attribution.record_counter(performance_counter, u64::MAX);
        performance_attribution.record_counter(performance_counter, 1);

        assert_eq!(
            performance_attribution.counter_value(performance_counter),
            u64::MAX
        );
    }
}
