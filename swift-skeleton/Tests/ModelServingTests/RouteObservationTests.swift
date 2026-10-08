import Foundation

import Testing

import ModelServing

/// Hermetic contracts for the decode route-observation history, port of
/// crates/model-serving/tests/hermetic/route_observation.rs.
@Suite
final class RouteObservationTests {

    private static func observation(
        inputTokenId: UInt32,
        routeMarker: UInt16
    ) -> RouteObservationRecord {
        return RouteObservationRecord(
            inputTokenId: inputTokenId,
            previousTokenRoute: nil,
            tokenRoute: [[routeMarker]])
    }

    @Test
    func shouldRetainObservationsInArrivalOrder() {
        let ring = RouteObservationRing(capacity: 4)
        for tokenId in UInt32(0)..<3 {
            ring.recordObservation(Self.observation(
                inputTokenId: tokenId,
                routeMarker: UInt16(tokenId)))
        }

        #expect(ring.observations().map(\.inputTokenId) == [0, 1, 2])
        #expect(ring.observationCount == 3)
        #expect(ring.storedObservationCount == 3)
        #expect(ring.evictedObservationCount == 0)
    }

    @Test
    func shouldOverwriteTheOldestObservationWhenFull() {
        let ring = RouteObservationRing(capacity: 3)
        for tokenId in UInt32(0)..<7 {
            ring.recordObservation(Self.observation(
                inputTokenId: tokenId,
                routeMarker: UInt16(tokenId)))
        }

        #expect(ring.observations().map(\.inputTokenId) == [4, 5, 6])
        #expect(ring.observationCount == 3)
        #expect(ring.storedObservationCount == 7)
        #expect(ring.evictedObservationCount == 4)
    }

    @Test
    func shouldPreserveCompleteRecordContents() {
        let previousTokenRoute: ObservedExpertRoute = [[3, 9], nil, [41]]
        let tokenRoute: ObservedExpertRoute = [[1, 2, 3], nil, [7]]
        let record = RouteObservationRecord(
            inputTokenId: 12_345,
            previousTokenRoute: previousTokenRoute,
            tokenRoute: tokenRoute)
        let ring = RouteObservationRing(capacity: 2)
        ring.recordObservation(Self.observation(inputTokenId: 1, routeMarker: 1))
        ring.recordObservation(record)

        let retained = ring.observations()
        #expect(retained.count == 2)
        #expect(retained[1] == record)
        #expect(retained[1].previousTokenRoute == previousTokenRoute)
        #expect(retained[1].tokenRoute == tokenRoute)
    }

    @Test
    func shouldAdmitEveryNewestObservationAtSingleObservationCapacity() {
        let ring = RouteObservationRing(capacity: 1)
        for tokenId in UInt32(0)..<5 {
            ring.recordObservation(Self.observation(
                inputTokenId: tokenId,
                routeMarker: UInt16(tokenId)))
        }

        #expect(ring.observations().map(\.inputTokenId) == [4])
        #expect(ring.evictedObservationCount == 4)
    }

    @Test
    func shouldClampZeroCapacityToOneSoCaptureNeverStalls() {
        let ring = RouteObservationRing(capacity: 0)
        ring.recordObservation(Self.observation(inputTokenId: 9, routeMarker: 9))

        #expect(ring.capacity == 1)
        #expect(ring.observationCount == 1)
    }

    @Test
    func shouldSortAndDeduplicateOneLayerRoute() {
        let compacted = RouteObservationCompaction.sortedUniqueLayerRoutedExpertIds(
            rawExpertIds: [40, 8, 40, 15, 8])

        #expect(compacted == [8, 15, 40])
    }

    @Test
    func shouldRejectIdentifiersBeyondTheExpertPopulation() {
        let compacted = RouteObservationCompaction.sortedUniqueLayerRoutedExpertIds(
            rawExpertIds: [8, UInt32(UInt16.max) + 1])

        #expect(compacted == nil)
    }

    @Test
    func shouldCompactAnEmptyRouteIntoAnEmptyLayerSelection() {
        let compacted = RouteObservationCompaction.sortedUniqueLayerRoutedExpertIds(
            rawExpertIds: [])

        #expect(compacted == [])
    }
}
