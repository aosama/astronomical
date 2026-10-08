import Foundation;

/// Bounded request evidence for expert source plans issued by SSD streaming.
///
/// Aggregate byte counters show total storage demand but cannot reveal whether
/// one nonresident layer was reread for every prompt chunk. One summary per
/// phase and layer preserves that evidence without allocating per token-layer
/// pass.
enum ExpertStreamingPhase: String, Sendable {

    case prefill;
    case decode;
}

/// One aggregated row per phase and layer, serialized without routed expert
/// identifiers so diagnostics stay bounded and privacy-safe.
struct ExpertStreamingSourceSummary: Sendable {

    private(set) var phase: ExpertStreamingPhase;
    private(set) var layerIndex: Int;
    private(set) var sourcePlanCount: UInt64;
    private(set) var totalRouteTokenCount: UInt64;
    private(set) var totalRoutedExpertCount: UInt64;
    private(set) var totalStreamedExpertCount: UInt64;
    private(set) var totalSourceShardCount: UInt64;
    private(set) var payloadByteCount: UInt64;
    /// True when at least one recorded plan streamed through per-expert
    /// `.apack` files instead of SafeTensors shard ranges.
    private(set) var streamedThroughExpertPacks: Bool;

    static func empty(phase: ExpertStreamingPhase, layerIndex: Int) -> ExpertStreamingSourceSummary {
        ExpertStreamingSourceSummary(
            phase: phase,
            layerIndex: layerIndex,
            sourcePlanCount: 0,
            totalRouteTokenCount: 0,
            totalRoutedExpertCount: 0,
            totalStreamedExpertCount: 0,
            totalSourceShardCount: 0,
            payloadByteCount: 0,
            streamedThroughExpertPacks: false);
    }

    mutating func recordSourcePlan(
        routeTokenCount: UInt64,
        routedExpertCount: UInt64,
        streamedExpertCount: UInt64,
        sourceShardCount: UInt64,
        payloadByteCount: UInt64,
        streamedThroughExpertPacks: Bool
    ) -> Void {
        let (nextSourcePlanCount, planOverflow) = sourcePlanCount
            .addingReportingOverflow(1);
        sourcePlanCount = planOverflow ? UInt64.max : nextSourcePlanCount;
        totalRouteTokenCount = totalRouteTokenCount
            &+ routeTokenCount;
        totalRoutedExpertCount = totalRoutedExpertCount
            &+ routedExpertCount;
        totalStreamedExpertCount = totalStreamedExpertCount
            &+ streamedExpertCount;
        totalSourceShardCount = totalSourceShardCount
            &+ sourceShardCount;
        self.payloadByteCount = self.payloadByteCount
            &+ payloadByteCount;
        self.streamedThroughExpertPacks = self.streamedThroughExpertPacks
            || streamedThroughExpertPacks;
    }
}
