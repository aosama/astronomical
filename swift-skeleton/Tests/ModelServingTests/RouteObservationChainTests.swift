import Foundation

import Testing

import ModelServing

/// Request-owned route observation chain contracts, port of
/// crates/model-serving/tests/hermetic/performance_attribution/route_observation.rs.
@Suite
final class RouteObservationChainTests {

    @Test
    func disabledAttributionDoesNotRecordARouteObservationChain() {
        let performanceAttribution = PerformanceAttribution.disabled();
        let previousRoute = performanceAttribution.advanceRouteObservationChain([[1, 2]]);
        #expect(
            previousRoute == nil,
            "disabled attribution must not allocate a previous-route chain");
    }

    @Test
    func enabledAttributionChainsThePreviousObservedRouteWithinOneRequest() {
        let performanceAttribution = PerformanceAttribution.enabled();
        let firstPreviousRoute = performanceAttribution.advanceRouteObservationChain([
            [3, 7],
        ]);
        #expect(
            firstPreviousRoute == nil,
            "the first observed token of a request has no previous route");
        let secondPreviousRoute = performanceAttribution.advanceRouteObservationChain([
            [4, 8],
        ]);
        #expect(secondPreviousRoute == [[3, 7]]);
    }
}
