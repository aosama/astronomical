import Foundation

/// Request-pressure freezes and deterministic reclamation for
/// `RetainedExpertPageCache`, split from the ownership container so each
/// file keeps to one job: this one decides which owned pages to drop and
/// under which temporary ceilings.
extension RetainedExpertPageCache {

    /**
     * Freezes retained pages at a smaller ceiling so the remaining prompt
     * fits: "please free about this many expert bytes" becomes an absolute
     * cap of current payload minus the requested release. Whole layers are
     * the only eviction unit, so the cache may free more than asked but
     * never less while enough payload exists.
     */
    public func limitForRequestPressure(reclamationTargetBytes: UInt64) -> Bool {
        let pressureMaximum: UInt64 =
            SaturatingArithmetic.subtract(self.residentPayloadBytes, reclamationTargetBytes)
        return self.limitForRequestPressureToMaximum(
            pressureMaximumResidentPayloadBytes: pressureMaximum)
    }

    /**
     * Installs an absolute request-scoped ceiling derived from one admitted
     * forward. Unlike deficit-based reclamation, this leaves room for
     * mandatory reads to become retained while preventing those reads from
     * consuming the exact context, streaming-page, and transient reserve
     * already admitted.
     */
    public func limitForRequestPressureToMaximum(
        pressureMaximumResidentPayloadBytes: UInt64
    ) -> Bool {
        self.requestPressureMaximumResidentPayloadBytes = pressureMaximumResidentPayloadBytes
        let reclamation: RetainedExpertReclamation = self.reclaimToEffectiveCeiling()
        return reclamation.releasedPayloadBytes() > 0
    }

    /**
     * Lifts the temporary request-pressure freeze without loading pages:
     * only the smaller cap is forgotten, so the long-lived budget governs
     * again. Returns whether a freeze was actually present.
     */
    public func resumeAfterRequestPressure() -> Bool {
        if self.requestPressureMaximumResidentPayloadBytes != nil {
            self.requestPressureMaximumResidentPayloadBytes = nil
            return true
        }
        return false
    }

    /// Releases elastic pages before stable complete layers for an exact deficit.
    public func reclaimForRequestPressure(
        requiredPayloadBytes: UInt64
    ) -> RetainedExpertReclamation {
        let targetPayloadBytes: UInt64 =
            SaturatingArithmetic.subtract(self.residentPayloadBytes, requiredPayloadBytes)
        return self.reclaimToPayloadCeiling(payloadCeilingBytes: targetPayloadBytes)
    }

    func reclaimToEffectiveCeiling() -> RetainedExpertReclamation {
        return self.reclaimToPayloadCeiling(
            payloadCeilingBytes: self.effectiveMaximumResidentPayloadBytes())
    }

    func reclaimToPayloadCeiling(
        payloadCeilingBytes: UInt64
    ) -> RetainedExpertReclamation {
        var reclamation: RetainedExpertReclamation = RetainedExpertReclamation()
        while self.residentPayloadBytes > payloadCeilingBytes {
            guard let layerIndex: Int =
                self.lowestCoveragePartialLayerIndex(protectedLayerIndex: -1) else {
                break
            }
            let releasedPayloadBytes: UInt64 = self.retainedLayers[layerIndex]?.payloadBytes ?? 0
            self.removeLayer(at: layerIndex)
            reclamation.releasedPartialLayerCount += 1
            reclamation.releasedPartialPayloadBytes = SaturatingArithmetic.add(
                reclamation.releasedPartialPayloadBytes,
                releasedPayloadBytes)
        }
        for layerIndex: Int in stride(from: self.retainedLayers.count - 1, through: 0, by: -1) {
            if self.residentPayloadBytes <= payloadCeilingBytes {
                break
            }
            guard let retainedEntry: RetainedExpertLayerEntry<ExpertPage> =
                self.retainedLayers[layerIndex] else {
                continue
            }
            if retainedEntry.pageClass != .stableCompleteLayer {
                continue
            }
            self.removeLayer(at: layerIndex)
            reclamation.releasedCompleteLayerCount += 1
            reclamation.releasedCompletePayloadBytes = SaturatingArithmetic.add(
                reclamation.releasedCompletePayloadBytes,
                retainedEntry.payloadBytes)
        }
        return reclamation
    }

    /// The lowest demand-per-byte elastic page; ties prefer the lowest layer.
    func lowestCoveragePartialLayerIndex(protectedLayerIndex: Int) -> Int? {
        var bestLayerIndex: Int? = nil
        var bestEntry: RetainedExpertLayerEntry<ExpertPage>? = nil
        for (layerIndex, layerSlot) in self.retainedLayers.enumerated() {
            guard let retainedEntry: RetainedExpertLayerEntry<ExpertPage> = layerSlot else {
                continue
            }
            if layerIndex == protectedLayerIndex
                || retainedEntry.pageClass != .elasticRoutedExperts {
                continue
            }
            if let currentBestLayerIndex: Int = bestLayerIndex,
                let currentBestEntry: RetainedExpertLayerEntry<ExpertPage> = bestEntry {
                if self.crossCoverageOrder(
                    leftLayerIndex: layerIndex,
                    leftEntry: retainedEntry,
                    rightLayerIndex: currentBestLayerIndex,
                    rightEntry: currentBestEntry) == .orderedAscending {
                    bestLayerIndex = layerIndex
                    bestEntry = retainedEntry
                }
            } else {
                bestLayerIndex = layerIndex
                bestEntry = retainedEntry
            }
        }
        return bestLayerIndex
    }

    /// Compares two elastic layers by demand-per-byte via cross
    /// multiplication over 128-bit products; ties prefer the lower index.
    func crossCoverageOrder(
        leftLayerIndex: Int,
        leftEntry: RetainedExpertLayerEntry<ExpertPage>,
        rightLayerIndex: Int,
        rightEntry: RetainedExpertLayerEntry<ExpertPage>
    ) -> ComparisonResult {
        let leftDemand: UInt64 = self.coveredDemand(
            layerIndex: leftLayerIndex,
            expertIds: leftEntry.expertIds)
        let rightDemand: UInt64 = self.coveredDemand(
            layerIndex: rightLayerIndex,
            expertIds: rightEntry.expertIds)
        let leftScore: (UInt64, UInt64) =
            leftDemand.multipliedFullWidth(by: rightEntry.payloadBytes)
        let rightScore: (UInt64, UInt64) =
            rightDemand.multipliedFullWidth(by: leftEntry.payloadBytes)
        if leftScore == rightScore {
            if leftLayerIndex == rightLayerIndex {
                return .orderedSame
            }
            return leftLayerIndex < rightLayerIndex ? .orderedAscending : .orderedDescending
        }
        return leftScore < rightScore ? .orderedAscending : .orderedDescending
    }

    func coveredDemand(layerIndex: Int, expertIds: [Int]) -> UInt64 {
        let layerDemandCounts: [UInt64] = self.expertDemandCountsByLayer[layerIndex]
        var demandTotal: UInt64 = 0
        for expertId: Int in expertIds {
            if expertId >= 0 && expertId < layerDemandCounts.count {
                demandTotal = SaturatingArithmetic.add(demandTotal, layerDemandCounts[expertId])
            }
        }
        return demandTotal
    }

    func recordEviction(pageClass: RetainedExpertPageClass) {
        self.evictionCount = SaturatingArithmetic.add(self.evictionCount, 1)
        if pageClass == .stableCompleteLayer {
            self.completeLayerEvictionCount =
                SaturatingArithmetic.add(self.completeLayerEvictionCount, 1)
        } else {
            self.partialLayerEvictionCount =
                SaturatingArithmetic.add(self.partialLayerEvictionCount, 1)
        }
    }

    /// The tighter of the long-lived budget and any live request-pressure freeze.
    func effectiveMaximumResidentPayloadBytes() -> UInt64 {
        let pressureCeiling: UInt64 =
            self.requestPressureMaximumResidentPayloadBytes ?? UInt64.max
        return min(self.normalMaximumResidentPayloadBytes, pressureCeiling)
    }
}
