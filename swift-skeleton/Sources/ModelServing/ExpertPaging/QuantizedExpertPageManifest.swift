import Foundation

/// One selected expert page — possibly partial — assembled from one or
/// more shards. Port of the Rust `QuantizedExpertPageManifest`: the
/// compact page slots feed MLX `take`-style gathers while absent experts
/// keep the sentinel slot.
public struct QuantizedExpertPageManifest: Equatable, Sendable {

    /// Sentinel marking a global expert id absent from this compact page.
    public static let ABSENT_PAGE_SLOT: UInt32 = UInt32.max

    /// Expert ids seated in this page, in compact slot order.
    public var expertIds: [Int]

    /// Dense lookup consumed by gathered projections: global expert id to
    /// compact page slot, with `ABSENT_PAGE_SLOT` for absent experts.
    public var pageSlotByGlobalExpertId: [UInt32]

    /// Per-shard assembly plans, one per contributing shard.
    public var sourceManifests: [QuantizedExpertShardManifest]

    /// Total payload bytes the page spans across every shard.
    public var payloadByteCount: UInt64

    public init(
        expertIds: [Int],
        pageSlotByGlobalExpertId: [UInt32],
        sourceManifests: [QuantizedExpertShardManifest],
        payloadByteCount: UInt64
    ) {
        self.expertIds = expertIds
        self.pageSlotByGlobalExpertId = pageSlotByGlobalExpertId
        self.sourceManifests = sourceManifests
        self.payloadByteCount = payloadByteCount
    }

    /**
     * Reports whether every global expert holds a compact page slot,
     * meaning the page is complete for its layer.
     */
    public func containsAllExperts() -> Bool {
        if self.expertIds.count != self.pageSlotByGlobalExpertId.count {
            return false
        }
        for pageSlot: UInt32 in self.pageSlotByGlobalExpertId
        where pageSlot == QuantizedExpertPageManifest.ABSENT_PAGE_SLOT {
            return false
        }
        return true
    }

    /**
     * Reports whether the page holds every expert a route selected.
     *
     * - Parameter selectedExpertIds: The routed expert ids to check.
     * - Returns: True when no selected expert is missing from the page.
     */
    public func containsEveryExpert(selectedExpertIds: [Int]) -> Bool {
        return self.missingExpertIds(selectedExpertIds: selectedExpertIds).isEmpty
    }

    /**
     * Lists the routed experts this page is missing, in route order.
     *
     * - Parameter selectedExpertIds: The routed expert ids to check.
     * - Returns: The missing expert ids, duplicates preserved.
     */
    public func missingExpertIds(selectedExpertIds: [Int]) -> [Int] {
        var missingIds: [Int] = []
        for expertId: Int in selectedExpertIds {
            let pageHoldsExpert: Bool = self.pageSlot(forGlobalExpertId: expertId) != nil
            if pageHoldsExpert == false {
                missingIds.append(expertId)
            }
        }
        return missingIds
    }

    /**
     * Splits one route's assignments so retained experts execute against
     * the resident page and missing experts stream in, with no expert
     * executed twice. Assignment positions keep route order; the expert
     * id lists come out sorted and deduplicated.
     *
     * - Parameter selectedExpertIds: The routed expert ids, in route order.
     * - Returns: The disjoint retained/missing partition.
     */
    public func partitionRouteAssignments(selectedExpertIds: [Int]) -> ExpertRoutePartition {
        var retainedAssignmentPositions: [Int] = []
        var retainedExpertIds: [Int] = []
        var missingAssignmentPositions: [Int] = []
        var missingExpertIds: [Int] = []
        for (assignmentPosition, expertId): (Int, Int) in selectedExpertIds.enumerated() {
            let pageHoldsExpert: Bool = self.pageSlot(forGlobalExpertId: expertId) != nil
            if pageHoldsExpert {
                retainedAssignmentPositions.append(assignmentPosition)
                retainedExpertIds.append(expertId)
            } else {
                missingAssignmentPositions.append(assignmentPosition)
                missingExpertIds.append(expertId)
            }
        }
        return ExpertRoutePartition(
            retainedAssignmentPositions: retainedAssignmentPositions,
            retainedExpertIds: Self.sortedDeduplicated(retainedExpertIds),
            missingAssignmentPositions: missingAssignmentPositions,
            missingExpertIds: Self.sortedDeduplicated(missingExpertIds))
    }

    private func pageSlot(forGlobalExpertId expertId: Int) -> UInt32? {
        if expertId < 0 || expertId >= self.pageSlotByGlobalExpertId.count {
            return nil
        }
        let pageSlot: UInt32 = self.pageSlotByGlobalExpertId[expertId]
        if pageSlot == QuantizedExpertPageManifest.ABSENT_PAGE_SLOT {
            return nil
        }
        return pageSlot
    }

    private static func sortedDeduplicated(_ expertIds: [Int]) -> [Int] {
        var uniqueIds: Set<Int> = Set()
        var deduplicatedIds: [Int] = []
        for expertId: Int in expertIds.sorted() where uniqueIds.insert(expertId).inserted {
            deduplicatedIds.append(expertId)
        }
        return deduplicatedIds
    }
}
