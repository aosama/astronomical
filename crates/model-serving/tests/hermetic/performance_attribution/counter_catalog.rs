use super::*;

/// Result of the catalog guard: every counter must be addressable inside the
/// enabled report storage.
///
/// The regression this protects arrived as a live production failure rather
/// than a build failure. A newly declared counter whose storage slot was never
/// reserved panicked inside the inference worker while it recorded admission
/// evidence for a resident vision request, so the worker stopped and the user
/// saw "model execution failed inside the local worker" instead of an answer.
#[test]
fn should_record_and_read_back_every_catalogued_counter() {
    for performance_counter in PerformanceCounter::ALL {
        let mut performance_attribution = PerformanceAttribution::enabled();

        performance_attribution.record_counter(performance_counter, 7);
        assert_eq!(
            performance_attribution.counter_value(performance_counter),
            7,
            "a catalogued counter must have a reserved storage slot: {}",
            performance_counter.identifier()
        );

        performance_attribution.record_snapshot_counter(performance_counter, 11);
        assert_eq!(
            performance_attribution.counter_value(performance_counter),
            11,
            "a catalogued counter must be overwritable as an absolute snapshot: {}",
            performance_counter.identifier()
        );

        performance_attribution.record_maximum_counter(performance_counter, 13);
        assert_eq!(
            performance_attribution.counter_value(performance_counter),
            13,
            "a catalogued counter must accept a running maximum: {}",
            performance_counter.identifier()
        );
    }
}

/// Serialized report records are keyed by identifier, so two counters that
/// share one identifier would silently overwrite each other in diagnostics.
#[test]
fn should_give_every_catalogued_counter_a_unique_report_identifier() {
    let mut catalogued_identifiers = std::collections::HashSet::new();

    for performance_counter in PerformanceCounter::ALL {
        let identifier = performance_counter.identifier();
        assert!(
            !identifier.is_empty(),
            "a counter without a report identifier cannot be read from diagnostics"
        );
        assert!(
            catalogued_identifiers.insert(identifier),
            "duplicate counter identifier {identifier}"
        );
    }

    assert_eq!(
        catalogued_identifiers.len(),
        PerformanceCounter::COUNT,
        "the identifier set must cover exactly the reserved counter storage"
    );
}

/// Report serialization pairs `ALL` with the fixed storage by position, so a
/// catalog entry out of declaration order, or a declaration that never reached
/// `ALL`, makes the report mislabel one value and drop another.
#[test]
fn should_keep_counter_catalog_positions_aligned_with_report_storage() {
    assert_eq!(
        PerformanceCounter::ALL.len(),
        PerformanceCounter::COUNT,
        "the enumerated catalog must cover the whole reserved counter storage"
    );

    for (declaration_position, performance_counter) in
        PerformanceCounter::ALL.into_iter().enumerate()
    {
        assert_eq!(
            performance_counter as usize,
            declaration_position,
            "catalog position {declaration_position} must store its own value: {}",
            performance_counter.identifier()
        );
    }

    assert_eq!(
        PerformanceCounter::AdmissionReserveGlobalMaximumSourceCount as usize,
        PerformanceCounter::COUNT - 1,
        "declare a new counter by appending its variant and taking COUNT from it, so a stale COUNT cannot leave it unaddressable"
    );
}
