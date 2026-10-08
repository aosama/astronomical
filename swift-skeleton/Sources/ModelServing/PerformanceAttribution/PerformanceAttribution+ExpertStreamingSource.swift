import Foundation;

extension PerformanceAttribution {

    /// Records one successful lazy source plan without retaining routed expert IDs.
    public func recordExpertStreamingSourcePlan(
        layerIndex: Int,
        routeTokenCount: Int32,
        routedExpertCount: Int,
        streamedExpertCount: Int,
        sourceShardCount: Int,
        payloadByteCount: UInt64,
        streamedThroughExpertPacks: Bool
    ) -> Void {
        guard var enabledAttribution = enabledAttribution else {
            return;
        }
        let phase: ExpertStreamingPhase = routeTokenCount > 1 ? .prefill : .decode;
        let mandatorySourcePayloadCounter: PerformanceCounter = phase == .prefill
            ? .mandatoryPrefillExpertSourcePayloadBytes
            : .mandatoryDecodeExpertSourcePayloadBytes;
        enabledAttribution.counterValues[mandatorySourcePayloadCounter.rawValue] =
            enabledAttribution.counterValues[mandatorySourcePayloadCounter.rawValue]
                &+ payloadByteCount;
        let phaseSlot: Int = phase == .prefill ? 0 : 1;
        let (layerSummaryOffset, multiplicationOverflow) = layerIndex
            .multipliedReportingOverflow(by: 2);
        if multiplicationOverflow {
            self.enabledAttribution = enabledAttribution;
            return;
        }
        let (summaryIndex, additionOverflow) = layerSummaryOffset
            .addingReportingOverflow(phaseSlot);
        if additionOverflow {
            self.enabledAttribution = enabledAttribution;
            return;
        }
        if enabledAttribution.expertStreamingSourceSummaries.count <= summaryIndex {
            enabledAttribution.expertStreamingSourceSummaries.append(
                contentsOf: Array(
                    repeating: nil,
                    count: summaryIndex + 1
                        - enabledAttribution.expertStreamingSourceSummaries.count));
        }
        let summary = enabledAttribution.expertStreamingSourceSummaries[summaryIndex]
            ?? ExpertStreamingSourceSummary.empty(phase: phase, layerIndex: layerIndex);
        var mutableSummary = summary;
        mutableSummary.recordSourcePlan(
            routeTokenCount: nonNegativeCountToUInt64(routeTokenCount),
            routedExpertCount: nonNegativeCountToUInt64(routedExpertCount),
            streamedExpertCount: nonNegativeCountToUInt64(streamedExpertCount),
            sourceShardCount: nonNegativeCountToUInt64(sourceShardCount),
            payloadByteCount: payloadByteCount,
            streamedThroughExpertPacks: streamedThroughExpertPacks);
        enabledAttribution.expertStreamingSourceSummaries[summaryIndex] = mutableSummary;
        self.enabledAttribution = enabledAttribution;
    }
}

/// Counting values cannot be negative, so negatives collapse to zero the same
/// way the Rust owner's `u64::try_from(...).unwrap_or(0)` does for token counts.
private func nonNegativeCountToUInt64(_ integerCount: Int32) -> UInt64 {
    integerCount <= 0 ? 0 : UInt64(integerCount);
}

/// 64-bit counting values always fit `UInt64` once nonnegative, so saturation
/// cannot trigger; the guard keeps the mapping total.
private func nonNegativeCountToUInt64(_ integerCount: Int) -> UInt64 {
    integerCount <= 0 ? 0 : UInt64(integerCount);
}
