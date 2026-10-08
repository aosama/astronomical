import Foundation

import Testing

import ModelServing

/// Previous-token expert route reuse contracts, port of
/// crates/model-serving/tests/hermetic/performance_attribution/expert_route_reuse.rs.
@Suite
final class ExpertRouteReuseTests {

    @Test
    func shouldMeasurePreviousTokenExpertRouteReuseWithoutSerializingExpertIds() throws {
        let performanceAttribution = PerformanceAttribution.enabled();

        performanceAttribution.recordPreviousTokenExpertRouteReuse(
            layerIndex: 3,
            tokenCount: 1,
            selectedExpertIds: [9, 2, 2, 5]);
        performanceAttribution.recordPreviousTokenExpertRouteReuse(
            layerIndex: 3,
            tokenCount: 1,
            selectedExpertIds: [7, 5, 2, 2]);

        #expect(
            performanceAttribution.counterValue(.expertRoutePredictedExpertCount) == 3);
        #expect(
            performanceAttribution.counterValue(.expertRouteMatchedExpertCount) == 2);
        #expect(
            performanceAttribution.counterValue(.expertRouteExaminedLayerCount) == 1);
        #expect(
            performanceAttribution.counterValue(.expertRouteCompletelyMatchedLayerCount)
                == 0);

        let performanceAttributionReport = try #require(
            performanceAttribution.finishGeneration(
                PerformanceAttributionTestSupport.generationMetadata(outcome: .success)),
            "enabled attribution should produce one generation report");
        let performanceAttributionJson = try PerformanceAttributionTestSupport
            .serialize(performanceAttributionReport);
        let routeReuseRows = PerformanceAttributionTestSupport.rows(
            performanceAttributionJson,
            "previous_token_expert_route_reuse_by_layer");
        #expect(
            PerformanceAttributionTestSupport.integer(routeReuseRows[0], "layer_index") == 3);
        #expect(
            PerformanceAttributionTestSupport.integer(routeReuseRows[0], "predicted_expert_count")
                == 3);
        #expect(
            PerformanceAttributionTestSupport.integer(routeReuseRows[0], "matched_expert_count")
                == 2);
        #expect(
            try !PerformanceAttributionTestSupport
                .serializedText(performanceAttributionReport)
                .contains("\"expert_ids\""));
    }

    @Test
    func shouldMeasureExpertRouteReuseIndependentlyForEachLayer() {
        let performanceAttribution = PerformanceAttribution.enabled();

        performanceAttribution.recordPreviousTokenExpertRouteReuse(
            layerIndex: 1,
            tokenCount: 1,
            selectedExpertIds: [2, 3]);
        performanceAttribution.recordPreviousTokenExpertRouteReuse(
            layerIndex: 4,
            tokenCount: 1,
            selectedExpertIds: [8, 9]);
        performanceAttribution.recordPreviousTokenExpertRouteReuse(
            layerIndex: 1,
            tokenCount: 1,
            selectedExpertIds: [2, 3]);
        performanceAttribution.recordPreviousTokenExpertRouteReuse(
            layerIndex: 4,
            tokenCount: 1,
            selectedExpertIds: [9, 10]);

        #expect(
            performanceAttribution.counterValue(.expertRoutePredictedExpertCount) == 4);
        #expect(
            performanceAttribution.counterValue(.expertRouteMatchedExpertCount) == 3);
        #expect(
            performanceAttribution.counterValue(.expertRouteCompletelyMatchedLayerCount)
                == 1);
        #expect(
            performanceAttribution.counterValue(.expertRouteExaminedLayerCount) == 2);
    }

    @Test
    func shouldSkipMultiTokenExpertRoutesAndDisabledAttribution() {
        let performanceAttribution = PerformanceAttribution.enabled();
        performanceAttribution.recordPreviousTokenExpertRouteReuse(
            layerIndex: 2,
            tokenCount: 4,
            selectedExpertIds: [1, 2]);
        performanceAttribution.recordPreviousTokenExpertRouteReuse(
            layerIndex: 2,
            tokenCount: 1,
            selectedExpertIds: [1, 2]);

        #expect(
            performanceAttribution.counterValue(.expertRouteExaminedLayerCount) == 0,
            "a multi-token route must not become the previous decode-token prediction");

        let disabledPerformanceAttribution = PerformanceAttribution.disabled();
        disabledPerformanceAttribution.recordPreviousTokenExpertRouteReuse(
            layerIndex: 2,
            tokenCount: 1,
            selectedExpertIds: [1, 2]);
        disabledPerformanceAttribution.recordPreviousTokenExpertRouteReuse(
            layerIndex: 2,
            tokenCount: 1,
            selectedExpertIds: [1, 2]);
        #expect(
            disabledPerformanceAttribution.counterValue(.expertRouteExaminedLayerCount)
                == 0);
    }
}
