import Foundation

import Testing

import ModelServing

/// Behavioral coverage for chunk-and-layer expert source attribution, port of
/// crates/model-serving/tests/hermetic/performance_attribution/expert_streaming_source.rs.
@Suite
final class ExpertStreamingSourceTests {

    @Test
    func shouldAggregateExpertSourcePlansByPhaseAndLayerWithoutExpertIds() throws {
        let performanceAttribution = PerformanceAttribution.enabled();

        performanceAttribution.recordExpertStreamingSourcePlan(
            layerIndex: 7,
            routeTokenCount: 2_048,
            routedExpertCount: 8,
            streamedExpertCount: 256,
            sourceShardCount: 3,
            payloadByteCount: 900,
            streamedThroughExpertPacks: false);
        performanceAttribution.recordExpertStreamingSourcePlan(
            layerIndex: 7,
            routeTokenCount: 417,
            routedExpertCount: 7,
            streamedExpertCount: 256,
            sourceShardCount: 3,
            payloadByteCount: 900,
            streamedThroughExpertPacks: false);
        performanceAttribution.recordExpertStreamingSourcePlan(
            layerIndex: 7,
            routeTokenCount: 1,
            routedExpertCount: 8,
            streamedExpertCount: 8,
            sourceShardCount: 2,
            payloadByteCount: 30,
            streamedThroughExpertPacks: false);

        #expect(
            performanceAttribution
                .counterValue(.mandatoryPrefillExpertSourcePayloadBytes) == 1_800);
        #expect(
            performanceAttribution
                .counterValue(.mandatoryDecodeExpertSourcePayloadBytes) == 30);

        let performanceAttributionReport = try #require(
            performanceAttribution.finishGeneration(
                PerformanceAttributionTestSupport.generationMetadata(outcome: .success)),
            "enabled attribution should produce one generation report");
        let performanceAttributionJson = try PerformanceAttributionTestSupport
            .serialize(performanceAttributionReport);
        let sourceSummaries = PerformanceAttributionTestSupport.rows(
            performanceAttributionJson,
            "expert_streaming_source_summaries");

        #expect(sourceSummaries.count == 2);
        #expect(
            PerformanceAttributionTestSupport.text(sourceSummaries[0], "phase")
                == "prefill");
        #expect(
            PerformanceAttributionTestSupport.integer(sourceSummaries[0], "source_plan_count")
                == 2);
        #expect(
            PerformanceAttributionTestSupport.integer(
                sourceSummaries[0],
                "total_route_token_count") == 2_465);
        #expect(
            PerformanceAttributionTestSupport.integer(sourceSummaries[0], "payload_byte_count")
                == 1_800);
        #expect(
            PerformanceAttributionTestSupport.text(sourceSummaries[1], "phase")
                == "decode");
        #expect(
            PerformanceAttributionTestSupport.integer(sourceSummaries[1], "source_plan_count")
                == 1);
        #expect(
            PerformanceAttributionTestSupport.integer(
                sourceSummaries[1],
                "total_streamed_expert_count") == 8);
        #expect(
            PerformanceAttributionTestSupport.integer(
                sourceSummaries[1],
                "total_source_shard_count") == 2);
        #expect(
            PerformanceAttributionTestSupport.integer(sourceSummaries[1], "payload_byte_count")
                == 30);
        #expect(
            try !PerformanceAttributionTestSupport
                .serializedText(performanceAttributionReport)
                .contains("expert_ids"));
    }

    @Test
    func shouldBoundExpertSourceSummariesToOneRowPerPhaseAndLayer() throws {
        let performanceAttribution = PerformanceAttribution.enabled();

        performanceAttribution.recordExpertStreamingSourcePlan(
            layerIndex: 2,
            routeTokenCount: 128,
            routedExpertCount: 8,
            streamedExpertCount: 256,
            sourceShardCount: 1,
            payloadByteCount: 100,
            streamedThroughExpertPacks: false);
        performanceAttribution.recordExpertStreamingSourcePlan(
            layerIndex: 4,
            routeTokenCount: 128,
            routedExpertCount: 8,
            streamedExpertCount: 256,
            sourceShardCount: 1,
            payloadByteCount: 100,
            streamedThroughExpertPacks: false);
        performanceAttribution.recordExpertStreamingSourcePlan(
            layerIndex: 2,
            routeTokenCount: 128,
            routedExpertCount: 8,
            streamedExpertCount: 256,
            sourceShardCount: 1,
            payloadByteCount: 100,
            streamedThroughExpertPacks: false);

        let performanceAttributionReport = try #require(
            performanceAttribution.finishGeneration(
                PerformanceAttributionTestSupport.generationMetadata(outcome: .success)),
            "enabled attribution should produce one generation report");
        let performanceAttributionJson = try PerformanceAttributionTestSupport
            .serialize(performanceAttributionReport);
        let sourceSummaries = PerformanceAttributionTestSupport.rows(
            performanceAttributionJson,
            "expert_streaming_source_summaries");

        #expect(sourceSummaries.count == 2);
        #expect(
            PerformanceAttributionTestSupport.integer(sourceSummaries[0], "layer_index")
                == 2);
        #expect(
            PerformanceAttributionTestSupport.integer(sourceSummaries[0], "source_plan_count")
                == 2);
        #expect(
            PerformanceAttributionTestSupport.integer(sourceSummaries[0], "payload_byte_count")
                == 200);
        #expect(
            PerformanceAttributionTestSupport.integer(sourceSummaries[1], "layer_index")
                == 4);
        #expect(
            PerformanceAttributionTestSupport.integer(sourceSummaries[1], "source_plan_count")
                == 1);
    }
}
