import Foundation

/// One retained layer's owned page plus its ownership metadata.
struct RetainedExpertLayerEntry<Page> {
    var page: Page
    var pageClass: RetainedExpertPageClass
    var expertIds: [Int]
    var payloadBytes: UInt64
}

/**
 * Deterministic RAM ownership and demand evidence for loaded expert pages.
 *
 * This is a byte-accounting and ownership container, not a loader: the model
 * performs SafeTensors input/output and MLX evaluation before offering a
 * page here, so a retained page is always fully usable. The policy uses
 * observed route frequency to preserve or reclaim already-owned partial
 * pages and never turns demand into speculative I/O. Ceilings arrive from
 * the composed memory budget — never from machine-specific constants — and
 * paging timings feed the switchable performance attribution log on the
 * serving critical path.
 *
 * - Parameters:
 *   - ExpertPage: The opaque page payload a model family materializes.
 */
public final class RetainedExpertPageCache<ExpertPage: ExpertWeightPage> {

    /// One stable slot per decoder layer; nil means execution must stream it.
    var retainedLayers: [RetainedExpertLayerEntry<ExpertPage>?]

    /// Cumulative routed demand used to choose a useful page per layer.
    var expertDemandCountsByLayer: [[UInt64]]

    /// Multiplier applied to each recorded assignment; the last prefill
    /// chunk raises this so tail routes outrank earlier prompt routes.
    private var demandAssignmentWeight: UInt64

    /// Sum of payload bytes for every owned slot; metadata excluded.
    var residentPayloadBytes: UInt64

    /// Long-lived limit supplied by the composed MLX RAM budget.
    var normalMaximumResidentPayloadBytes: UInt64

    /// Temporary upper bound installed while one request needs expert bytes
    /// back; finalization removes it so retention refills without guessing
    /// the original machine budget.
    var requestPressureMaximumResidentPayloadBytes: UInt64?

    var evictionCount: UInt64
    var diskPageLoadCount: UInt64
    var diskBatchLoadCount: UInt64
    var mandatoryReadPromotionCount: UInt64
    var completeLayerEvictionCount: UInt64
    var partialLayerEvictionCount: UInt64

    public init(layerCount: Int) {
        self.retainedLayers = Array(repeating: nil, count: layerCount)
        self.expertDemandCountsByLayer = Array(repeating: [], count: layerCount)
        self.demandAssignmentWeight = 1
        self.residentPayloadBytes = 0
        self.normalMaximumResidentPayloadBytes = 0
        self.requestPressureMaximumResidentPayloadBytes = nil
        self.evictionCount = 0
        self.diskPageLoadCount = 0
        self.diskBatchLoadCount = 0
        self.mandatoryReadPromotionCount = 0
        self.completeLayerEvictionCount = 0
        self.partialLayerEvictionCount = 0
    }

    /// The page retained for one layer, if any.
    public func retainedLayer(at layerIndex: Int) -> ExpertPage? {
        if layerIndex < 0 || layerIndex >= self.retainedLayers.count {
            return nil
        }
        return self.retainedLayers[layerIndex]?.page
    }

    /// Mutates the retained page in place; returns whether the layer held one.
    public func mutateRetainedLayer(
        at layerIndex: Int,
        using mutator: (inout ExpertPage) -> Void
    ) -> Bool {
        guard layerIndex >= 0 && layerIndex < self.retainedLayers.count else {
            return false
        }
        guard var entry: RetainedExpertLayerEntry<ExpertPage> = self.retainedLayers[layerIndex] else {
            return false
        }
        mutator(&entry.page)
        self.retainedLayers[layerIndex] = entry
        return true
    }

    /// Re-reads the page byte count after an in-place overlay grow.
    public func syncPagePayloadBytes(at layerIndex: Int) {
        guard layerIndex >= 0 && layerIndex < self.retainedLayers.count else {
            return
        }
        guard var entry: RetainedExpertLayerEntry<ExpertPage> = self.retainedLayers[layerIndex] else {
            return
        }
        let updatedPayloadBytes: UInt64 = entry.page.residentPayloadByteCount()
        self.residentPayloadBytes = SaturatingArithmetic.add(
            SaturatingArithmetic.subtract(self.residentPayloadBytes, entry.payloadBytes),
            updatedPayloadBytes)
        entry.payloadBytes = updatedPayloadBytes
        self.retainedLayers[layerIndex] = entry
    }

    /**
     * Repeats exact projected-byte accounting before a caller transfers
     * ownership; never mutates state.
     */
    public func canCommitMaterializedPage(
        at layerIndex: Int,
        candidatePayloadBytes: UInt64
    ) -> Bool {
        if candidatePayloadBytes == 0 {
            return false
        }
        guard layerIndex >= 0 && layerIndex < self.retainedLayers.count else {
            return false
        }
        let existingPayloadBytes: UInt64 = self.retainedLayers[layerIndex]?.payloadBytes ?? 0
        let (payloadWithoutReplacedPage, subtractOverflowed) =
            self.residentPayloadBytes.subtractingReportingOverflow(existingPayloadBytes)
        if subtractOverflowed {
            return false
        }
        let (projectedPayloadBytes, addOverflowed) =
            payloadWithoutReplacedPage.addingReportingOverflow(candidatePayloadBytes)
        if addOverflowed {
            return false
        }
        return projectedPayloadBytes <= self.effectiveMaximumResidentPayloadBytes()
    }

    /**
     * Publishes the long-lived budget and reclaims down to it. A live
     * request-pressure freeze still wins until `resumeAfterRequestPressure`
     * lifts it, but the normal value survives so finalization can resume
     * retention without waiting for another budget publication.
     */
    public func updateMaximumResidentPayloadBytes(
        to maximumPayloadBytes: UInt64
    ) -> RetainedExpertReclamation {
        self.normalMaximumResidentPayloadBytes = maximumPayloadBytes
        return self.reclaimToEffectiveCeiling()
    }

    /**
     * Commits a complete layer loaded by a mandatory prefill read.
     *
     * - Throws: `RetainedExpertLayerCommitError` when the metadata is
     *   invalid; prior ownership is never mutated on a thrown error.
     */
    public func commitMaterializedCompleteLayer(
        at layerIndex: Int,
        expertCapacity: Int,
        expertPage: ExpertPage
    ) throws -> RetainedExpertLayerCommit<ExpertPage> {
        if layerIndex < 0 || layerIndex >= self.retainedLayers.count {
            throw RetainedExpertLayerCommitError.layerOutOfRange(layerIndex: layerIndex)
        }
        if expertCapacity == 0 {
            throw RetainedExpertLayerCommitError.zeroExpertCapacity(layerIndex: layerIndex)
        }
        if expertPage.residentPayloadByteCount() == 0 {
            throw RetainedExpertLayerCommitError.zeroPayload(layerIndex: layerIndex)
        }
        if let existingEntry: RetainedExpertLayerEntry<ExpertPage> = self.retainedLayers[layerIndex],
            existingEntry.pageClass == .stableCompleteLayer {
            return RetainedExpertLayerCommit(
                outcome: .preservedExisting,
                uncommittedPage: expertPage)
        }
        let completeExpertIds: [Int] = Array(0..<expertCapacity)
        let commit: RetainedExpertLayerCommit<ExpertPage> = try self.commitEntry(
            at: layerIndex,
            pageClass: .stableCompleteLayer,
            expertIds: completeExpertIds,
            expertPage: expertPage)
        if case .committed = commit.outcome {
            self.mandatoryReadPromotionCount =
                SaturatingArithmetic.add(self.mandatoryReadPromotionCount, 1)
        }
        return commit
    }

    /**
     * Commits the first exact routed page loaded by mandatory decode
     * execution. The existing owner stays unless the proposed route set is
     * a strict superset — replacing a useful partial with a disjoint miss
     * page would drop the experts the next token still needs.
     *
     * - Throws: `RetainedExpertLayerCommitError` when the metadata is
     *   invalid; prior ownership is never mutated on a thrown error.
     */
    public func commitMaterializedRoutedPage(
        at layerIndex: Int,
        expertCapacity: Int,
        expertIds: [Int],
        expertPage: ExpertPage
    ) throws -> RetainedExpertLayerCommit<ExpertPage> {
        try self.validateRoutedPageMetadata(
            layerIndex: layerIndex,
            expertCapacity: expertCapacity,
            expertIds: expertIds)
        if expertPage.residentPayloadByteCount() == 0 {
            throw RetainedExpertLayerCommitError.zeroPayload(layerIndex: layerIndex)
        }
        if let existingEntry: RetainedExpertLayerEntry<ExpertPage> = self.retainedLayers[layerIndex] {
            if existingEntry.pageClass == .stableCompleteLayer
                || RetainedExpertPageCache.routedExpertIdsAreStrictSuperset(
                    existingExpertIds: existingEntry.expertIds,
                    proposedExpertIds: expertIds) == false {
                return RetainedExpertLayerCommit(
                    outcome: .preservedExisting,
                    uncommittedPage: expertPage)
            }
        }
        return try self.commitEntry(
            at: layerIndex,
            pageClass: .elasticRoutedExperts,
            expertIds: expertIds,
            expertPage: expertPage)
    }

    /// Transfers one retained page out without counting the move as an
    /// eviction, so prefill-pinned experts become ordinary decode-cache
    /// residents instead of being recorded as a squeeze.
    public func takeRetainedLayer(at layerIndex: Int) -> ExpertPage? {
        guard layerIndex >= 0 && layerIndex < self.retainedLayers.count else {
            return nil
        }
        guard let takenEntry: RetainedExpertLayerEntry<ExpertPage> = self.retainedLayers[layerIndex] else {
            return nil
        }
        self.retainedLayers[layerIndex] = nil
        self.residentPayloadBytes =
            SaturatingArithmetic.subtract(self.residentPayloadBytes, takenEntry.payloadBytes)
        return takenEntry.page
    }

    /// Removes one stale page before a barrier-safe topology rebuild.
    @discardableResult
    public func removeLayer(at layerIndex: Int) -> Bool {
        guard layerIndex >= 0 && layerIndex < self.retainedLayers.count else {
            return false
        }
        guard let removedEntry: RetainedExpertLayerEntry<ExpertPage> = self.retainedLayers[layerIndex] else {
            return false
        }
        self.retainedLayers[layerIndex] = nil
        self.residentPayloadBytes =
            SaturatingArithmetic.subtract(self.residentPayloadBytes, removedEntry.payloadBytes)
        self.recordEviction(pageClass: removedEntry.pageClass)
        return true
    }

    /// Records routed experts without retaining request-owned arrays;
    /// identifiers at or beyond the capacity are ignored.
    public func recordExpertDemand(
        layerIndex: Int,
        expertCapacity: Int,
        selectedExpertIds: [Int]
    ) {
        guard layerIndex >= 0 && layerIndex < self.expertDemandCountsByLayer.count else {
            return
        }
        var layerDemandCounts: [UInt64] = self.expertDemandCountsByLayer[layerIndex]
        if layerDemandCounts.count < expertCapacity {
            layerDemandCounts.append(contentsOf: Array(repeating: 0, count: expertCapacity - layerDemandCounts.count))
        }
        let assignmentWeight: UInt64 = max(self.demandAssignmentWeight, 1)
        for selectedExpertId: Int in selectedExpertIds {
            if selectedExpertId >= 0 && selectedExpertId < layerDemandCounts.count {
                layerDemandCounts[selectedExpertId] =
                    SaturatingArithmetic.add(layerDemandCounts[selectedExpertId], assignmentWeight)
            }
        }
        self.expertDemandCountsByLayer[layerIndex] = layerDemandCounts
    }

    /// Raises or restores the per-assignment demand multiplier.
    public func setDemandAssignmentWeight(_ assignmentWeight: UInt64) {
        self.demandAssignmentWeight = max(assignmentWeight, 1)
    }

    /// Starts a fresh evidence window after one topology plan consumes demand.
    public func clearExpertDemand() {
        for layerIndex: Int in 0..<self.expertDemandCountsByLayer.count {
            self.expertDemandCountsByLayer[layerIndex] =
                Array(repeating: 0, count: self.expertDemandCountsByLayer[layerIndex].count)
        }
        self.demandAssignmentWeight = 1
    }

    /// Records disk I/O counts for telemetry attribution.
    public func recordDiskLoad(expertCount: Int, batchCount: Int) {
        self.diskPageLoadCount = SaturatingArithmetic.add(self.diskPageLoadCount, UInt64(expertCount))
        self.diskBatchLoadCount = SaturatingArithmetic.add(self.diskBatchLoadCount, UInt64(batchCount))
    }

    /**
     * Planner-ready ownership metadata in layer order, without exposing
     * page payloads. `expertCapacity` mirrors the Rust contract signature;
     * residency already records the exact expert identifiers it covers.
     */
    public func topologySnapshot(expertCapacity: Int) -> [CurrentExpertLayerResidency] {
        var snapshot: [CurrentExpertLayerResidency] = []
        for (layerIndex, layerSlot) in self.retainedLayers.enumerated() {
            guard let retainedEntry: RetainedExpertLayerEntry<ExpertPage> = layerSlot else {
                continue
            }
            let residency: CurrentExpertLayerResidency = CurrentExpertLayerResidency(
                layerIndex: layerIndex,
                pageClass: retainedEntry.pageClass,
                retainedExpertIds: retainedEntry.expertIds,
                payloadBytes: retainedEntry.payloadBytes,
                coveredWeightedDemand: self.coveredDemand(
                    layerIndex: layerIndex,
                    expertIds: retainedEntry.expertIds))
            snapshot.append(residency)
        }
        return snapshot
    }

    /// Drops every retained page, reporting whether any bytes were owned.
    @discardableResult
    public func releaseAll() -> Bool {
        let hadRetainedLayers: Bool = self.residentPayloadBytes > 0
        for layerIndex: Int in 0..<self.retainedLayers.count {
            self.removeLayer(at: layerIndex)
        }
        self.residentPayloadBytes = 0
        return hadRetainedLayers
    }

    /// Total experts covered by retained pages.
    public func residentExpertCount() -> Int {
        var expertTotal: Int = 0
        for layerSlot: RetainedExpertLayerEntry<ExpertPage>? in self.retainedLayers {
            if let retainedEntry: RetainedExpertLayerEntry<ExpertPage> = layerSlot {
                expertTotal += retainedEntry.expertIds.count
            }
        }
        return expertTotal
    }

    /// Observed ownership state; byte figures match the policy decisions.
    public func statistics() -> RetainedExpertPageStatistics {
        var completeLayerCount: Int = 0
        var completeLayerPayloadByteCount: UInt64 = 0
        var partialLayerCount: Int = 0
        var entryCount: Int = 0
        for layerSlot: RetainedExpertLayerEntry<ExpertPage>? in self.retainedLayers {
            guard let retainedEntry: RetainedExpertLayerEntry<ExpertPage> = layerSlot else {
                continue
            }
            entryCount += 1
            if retainedEntry.pageClass == .stableCompleteLayer {
                completeLayerCount += 1
                completeLayerPayloadByteCount = SaturatingArithmetic.add(
                    completeLayerPayloadByteCount,
                    retainedEntry.payloadBytes)
            } else {
                partialLayerCount += 1
            }
        }
        return RetainedExpertPageStatistics(
            entryCount: entryCount,
            residentPayloadByteCount: self.residentPayloadBytes,
            maximumResidentPayloadByteCount: self.effectiveMaximumResidentPayloadBytes(),
            evictionCount: self.evictionCount,
            diskPageLoadCount: self.diskPageLoadCount,
            diskBatchLoadCount: self.diskBatchLoadCount,
            completeLayerCount: completeLayerCount,
            completeLayerPayloadByteCount: completeLayerPayloadByteCount,
            partialLayerCount: partialLayerCount,
            partialLayerPayloadByteCount:
                SaturatingArithmetic.subtract(self.residentPayloadBytes, completeLayerPayloadByteCount),
            mandatoryReadPromotionCount: self.mandatoryReadPromotionCount,
            completeLayerEvictionCount: self.completeLayerEvictionCount,
            partialLayerEvictionCount: self.partialLayerEvictionCount)
    }

    /// Drops the least-used retained page; whole layers are the unit.
    @discardableResult
    public func evictLeastUsedRetainedPage() -> Bool {
        return self.evictLeastUsedRetainedPageExcept(protectedLayerIndex: Int.max)
    }

    /// Evicts elsewhere so the layer being grown is not the one dropped.
    @discardableResult
    public func evictLeastUsedRetainedPageExcept(protectedLayerIndex: Int) -> Bool {
        if let coldestPartialLayerIndex: Int =
            self.lowestCoveragePartialLayerIndex(protectedLayerIndex: protectedLayerIndex) {
            return self.removeLayer(at: coldestPartialLayerIndex)
        }
        for layerIndex: Int in stride(from: self.retainedLayers.count - 1, through: 0, by: -1) {
            if layerIndex == protectedLayerIndex {
                continue
            }
            if let retainedEntry: RetainedExpertLayerEntry<ExpertPage> = self.retainedLayers[layerIndex],
                retainedEntry.pageClass == .stableCompleteLayer {
                return self.removeLayer(at: layerIndex)
            }
        }
        return false
    }

    private func commitEntry(
        at layerIndex: Int,
        pageClass: RetainedExpertPageClass,
        expertIds: [Int],
        expertPage: ExpertPage
    ) throws -> RetainedExpertLayerCommit<ExpertPage> {
        let replacementPayloadBytes: UInt64 = expertPage.residentPayloadByteCount()
        let existingPayloadBytes: UInt64 = self.retainedLayers[layerIndex]?.payloadBytes ?? 0
        let (payloadWithoutReplacedPage, subtractOverflowed) =
            self.residentPayloadBytes.subtractingReportingOverflow(existingPayloadBytes)
        if subtractOverflowed {
            throw RetainedExpertLayerCommitError.inconsistentPayloadAccounting(layerIndex: layerIndex)
        }
        let (projectedPayloadBytes, addOverflowed) =
            payloadWithoutReplacedPage.addingReportingOverflow(replacementPayloadBytes)
        if addOverflowed {
            throw RetainedExpertLayerCommitError.payloadByteCountOverflow(layerIndex: layerIndex)
        }
        if projectedPayloadBytes > self.effectiveMaximumResidentPayloadBytes() {
            return RetainedExpertLayerCommit(
                outcome: .rejectedByCurrentCeiling,
                uncommittedPage: expertPage)
        }
        let replacementEntry: RetainedExpertLayerEntry<ExpertPage> = RetainedExpertLayerEntry(
            page: expertPage,
            pageClass: pageClass,
            expertIds: expertIds,
            payloadBytes: replacementPayloadBytes)
        if let replacedEntry: RetainedExpertLayerEntry<ExpertPage> =
            self.retainedLayers[layerIndex] {
            self.recordEviction(pageClass: replacedEntry.pageClass)
        }
        self.retainedLayers[layerIndex] = replacementEntry
        self.residentPayloadBytes = projectedPayloadBytes
        return RetainedExpertLayerCommit(
            outcome: .committed(RetainedExpertLayerCommitDelta(
                releasedPayloadBytes: existingPayloadBytes,
                committedPayloadBytes: replacementPayloadBytes)),
            uncommittedPage: nil)
    }

    private func validateRoutedPageMetadata(
        layerIndex: Int,
        expertCapacity: Int,
        expertIds: [Int]
    ) throws -> Void {
        if layerIndex < 0 || layerIndex >= self.retainedLayers.count {
            throw RetainedExpertLayerCommitError.layerOutOfRange(layerIndex: layerIndex)
        }
        if expertCapacity == 0 {
            throw RetainedExpertLayerCommitError.zeroExpertCapacity(layerIndex: layerIndex)
        }
        let identifiersAreAscendingAndInRange: Bool = expertIds.count > 1
            ? zip(expertIds, expertIds.dropFirst()).allSatisfy({ (pair: (Int, Int)) -> Bool in
                return pair.0 < pair.1
            })
            : true
        let identifiersAreValid: Bool = expertIds.isEmpty == false
            && expertIds.count < expertCapacity
            && identifiersAreAscendingAndInRange
            && expertIds.allSatisfy({ (expertId: Int) -> Bool in
                return expertId >= 0 && expertId < expertCapacity
            })
        if identifiersAreValid == false {
            throw RetainedExpertLayerCommitError.invalidExpertIds(layerIndex: layerIndex)
        }
    }

    /// A proposed page replaces a partial only when it covers strictly more
    /// experts than the owner already holds.
    private static func routedExpertIdsAreStrictSuperset(
        existingExpertIds: [Int],
        proposedExpertIds: [Int]
    ) -> Bool {
        if proposedExpertIds.count <= existingExpertIds.count {
            return false
        }
        let proposedIdSet: Set<Int> = Set(proposedExpertIds)
        for existingExpertId: Int in existingExpertIds {
            if proposedIdSet.contains(existingExpertId) == false {
                return false
            }
        }
        return true
    }
}
