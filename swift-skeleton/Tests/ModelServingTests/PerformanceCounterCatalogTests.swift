import Foundation

import Testing

import ModelServing

/// Result of the catalog guard: every counter must be addressable inside the
/// enabled report storage, port of
/// crates/model-serving/tests/hermetic/performance_attribution/counter_catalog.rs.
///
/// The regression this protects arrived as a live production failure rather
/// than a build failure. A newly declared counter whose storage slot was never
/// reserved panicked inside the inference worker while it recorded admission
/// evidence for a resident vision request, so the worker stopped and the user
/// saw "model execution failed inside the local worker" instead of an answer.
@Suite
final class PerformanceCounterCatalogTests {

    @Test
    func shouldRecordAndReadBackEveryCataloguedCounter() {
        for performanceCounter in PerformanceCounter.allCases {
            let performanceAttribution = PerformanceAttribution.enabled();

            performanceAttribution.recordCounter(performanceCounter, amount: 7);
            #expect(
                performanceAttribution.counterValue(performanceCounter) == 7,
                "a catalogued counter must have a reserved storage slot: \(performanceCounter.identifier)")

            performanceAttribution.recordSnapshotCounter(performanceCounter, amount: 11);
            #expect(
                performanceAttribution.counterValue(performanceCounter) == 11,
                "a catalogued counter must be overwritable as an absolute snapshot: \(performanceCounter.identifier)")

            performanceAttribution.recordMaximumCounter(performanceCounter, amount: 13);
            #expect(
                performanceAttribution.counterValue(performanceCounter) == 13,
                "a catalogued counter must accept a running maximum: \(performanceCounter.identifier)")
        }
    }

    /// Serialized report records are keyed by identifier, so two counters that
    /// share one identifier would silently overwrite each other in diagnostics.
    @Test
    func shouldGiveEveryCataloguedCounterAUniqueReportIdentifier() {
        var cataloguedIdentifiers = Set<String>();

        for performanceCounter in PerformanceCounter.allCases {
            let identifier = performanceCounter.identifier;
            #expect(
                !identifier.isEmpty,
                "a counter without a report identifier cannot be read from diagnostics");
            #expect(
                cataloguedIdentifiers.insert(identifier).inserted,
                "duplicate counter identifier \(identifier)");
        }

        #expect(
            cataloguedIdentifiers.count == PerformanceCounter.count,
            "the identifier set must cover exactly the reserved counter storage");
    }

    /// Report serialization pairs `allCases` with the fixed storage by position,
    /// so a catalog entry out of declaration order, or a declaration that never
    /// reached `allCases`, makes the report mislabel one value and drop another.
    @Test
    func shouldKeepCounterCatalogPositionsAlignedWithReportStorage() {
        #expect(
            PerformanceCounter.allCases.count == PerformanceCounter.count,
            "the enumerated catalog must cover the whole reserved counter storage");

        for (declarationPosition, performanceCounter) in PerformanceCounter.allCases
            .enumerated()
        {
            #expect(
                performanceCounter.rawValue == declarationPosition,
                "catalog position \(declarationPosition) must store its own value: \(performanceCounter.identifier)");
        }

        #expect(
            PerformanceCounter.admissionReserveGlobalMaximumSourceCount.rawValue
                == PerformanceCounter.count - 1,
            "declare a new counter by appending its case and taking count from it, so a stale count cannot leave it unaddressable");
    }
}
