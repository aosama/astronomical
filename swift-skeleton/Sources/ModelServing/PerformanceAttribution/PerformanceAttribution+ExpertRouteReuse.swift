import Foundation;

extension PerformanceAttribution {

    /// Compares one decode layer's selected experts with the preceding decode
    /// token, counting predicted, matched, and completely matched layers.
    public func recordPreviousTokenExpertRouteReuse(
        layerIndex: Int,
        tokenCount: Int32,
        selectedExpertIds: [Int]
    ) -> Void {
        guard var enabledAttribution = enabledAttribution else {
            return;
        }
        guard tokenCount == 1 else {
            return;
        }

        let sortedUniqueSelectedExpertIds = Array(Set(selectedExpertIds)).sorted();
        if enabledAttribution.previousTokenSelectedExpertIdsByLayer.count <= layerIndex {
            enabledAttribution.previousTokenSelectedExpertIdsByLayer
                .append(contentsOf: Array(
                    repeating: nil,
                    count: layerIndex + 1
                        - enabledAttribution.previousTokenSelectedExpertIdsByLayer.count));
        }
        let previousSelectedExpertIds = enabledAttribution
            .previousTokenSelectedExpertIdsByLayer[layerIndex];
        enabledAttribution.previousTokenSelectedExpertIdsByLayer[layerIndex] =
            sortedUniqueSelectedExpertIds;
        guard let previousSelectedExpertIds = previousSelectedExpertIds else {
            self.enabledAttribution = enabledAttribution;
            return;
        }

        let predictedExpertCount = integerCountToUInt64Saturating(
            previousSelectedExpertIds.count);
        let currentUniqueSelectedExpertIdSet = Set(sortedUniqueSelectedExpertIds);
        let matchedExpertCount = integerCountToUInt64Saturating(
            previousSelectedExpertIds.filter { currentUniqueSelectedExpertIdSet.contains($0) }
                .count);
        let completelyMatchedLayerCount: UInt64 =
            previousSelectedExpertIds == sortedUniqueSelectedExpertIds ? 1 : 0;
        if enabledAttribution.previousTokenExpertRouteReuseByLayer.count <= layerIndex {
            enabledAttribution.previousTokenExpertRouteReuseByLayer
                .append(contentsOf: Array(
                    repeating: PreviousTokenExpertRouteReuseMeasurement.empty,
                    count: layerIndex + 1
                        - enabledAttribution.previousTokenExpertRouteReuseByLayer.count));
        }
        let layerMeasurement = enabledAttribution
            .previousTokenExpertRouteReuseByLayer[layerIndex];
        enabledAttribution.previousTokenExpertRouteReuseByLayer[layerIndex] =
            PreviousTokenExpertRouteReuseMeasurement(
                predictedExpertCount: layerMeasurement.predictedExpertCount
                    &+ predictedExpertCount,
                matchedExpertCount: layerMeasurement.matchedExpertCount
                    &+ matchedExpertCount,
                completelyMatchedLayerCount: layerMeasurement.completelyMatchedLayerCount
                    &+ completelyMatchedLayerCount,
                examinedLayerCount: layerMeasurement.examinedLayerCount &+ 1);
        self.enabledAttribution = enabledAttribution;

        recordCounter(.expertRoutePredictedExpertCount, amount: predictedExpertCount);
        recordCounter(.expertRouteMatchedExpertCount, amount: matchedExpertCount);
        recordCounter(
            .expertRouteCompletelyMatchedLayerCount,
            amount: completelyMatchedLayerCount);
        recordCounter(.expertRouteExaminedLayerCount, amount: 1);
    }

    /// Returns the previous observed decode token's route and stores this
    /// token's route as the new chain head. The chain lives on the
    /// request-owned attribution, so a new request starts fresh without any
    /// explicit reset (issue #536).
    public func advanceRouteObservationChain(
        _ tokenRoute: ObservedExpertRoute
    ) -> ObservedExpertRoute? {
        guard var enabledAttribution = enabledAttribution else {
            return nil;
        }
        let previousRoute = enabledAttribution.routeObservationPreviousRoute;
        enabledAttribution.routeObservationPreviousRoute = tokenRoute;
        self.enabledAttribution = enabledAttribution;
        return previousRoute;
    }

    /// The previous decode token's true route, used as predictor input.
    public func previousObservedExpertRoute() -> ObservedExpertRoute? {
        enabledAttribution?.routeObservationPreviousRoute;
    }
}

private func integerCountToUInt64Saturating(_ integerCount: Int) -> UInt64 {
    integerCount <= 0 ? 0 : UInt64(integerCount);
}
