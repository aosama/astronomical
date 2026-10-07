import Foundation

/// One resident expert considered for decode-cache eviction.
private struct DecodeEvictionCandidate {
    var slot: Int
    var expertId: Int
    var demand: UInt64
    var payloadBytes: UInt64
}

/// Slots that lost at least one expert during one eviction pass.
internal struct DecodeExpertEvictionOutcome {
    var evictedExpertCount: Int
    var vacatedSlots: [Int]
}

/// Coldest-expert eviction for the decode cache: one expert is the
/// irreducible unit, so eviction removes the lowest demand-per-byte
/// experts until the requested bytes are freed.
internal enum DecodeExpertEviction {

    /**
     * Removes the lowest demand-per-byte experts until `bytesToFree` is
     * met. Protected `(slot, expertId)` pairs survive this pass so the
     * current forward cannot lose experts it just routed; zero-byte
     * experts never serve as eviction progress.
     */
    internal static func evictColdestExperts<Weight: ResidentExpertWeight>(
        layers: [ResidentExpertSet<Weight>],
        ledger: DecodeDemandLedger,
        bytesToFree: UInt64,
        protected: [(Int, Int)]
    ) -> DecodeExpertEvictionOutcome {
        if bytesToFree == 0 {
            return DecodeExpertEvictionOutcome(evictedExpertCount: 0, vacatedSlots: [])
        }
        var candidates: [DecodeEvictionCandidate] = []
        for (slot, layer) in layers.enumerated() {
            for expertId: Int in layer.residentIds() {
                let isProtected: Bool = protected.contains(where: { (protectedPair: (Int, Int)) -> Bool in
                    return protectedPair.0 == slot && protectedPair.1 == expertId
                })
                if isProtected {
                    continue
                }
                guard let payloadBytes: UInt64 = layer.payloadBytesFor(expertId: expertId) else {
                    continue
                }
                if payloadBytes == 0 {
                    continue
                }
                let candidate: DecodeEvictionCandidate = DecodeEvictionCandidate(
                    slot: slot,
                    expertId: expertId,
                    demand: ledger.demand(slot: slot, expertId: expertId),
                    payloadBytes: payloadBytes)
                candidates.append(candidate)
            }
        }
        candidates.sort(by: { (leftCandidate: DecodeEvictionCandidate, rightCandidate: DecodeEvictionCandidate) -> Bool in
            return candidateOrder(
                leftCandidate: leftCandidate,
                rightCandidate: rightCandidate) == .orderedAscending
        })
        var freedBytes: UInt64 = 0
        var evictedExpertCount: Int = 0
        var vacatedSlots: [Int] = []
        for candidate: DecodeEvictionCandidate in candidates {
            if freedBytes >= bytesToFree {
                break
            }
            if let _: Weight = layers[candidate.slot].evict(expertId: candidate.expertId) {
                freedBytes = SaturatingArithmetic.add(freedBytes, candidate.payloadBytes)
                evictedExpertCount += 1
                if vacatedSlots.contains(candidate.slot) == false {
                    vacatedSlots.append(candidate.slot)
                }
            }
        }
        return DecodeExpertEvictionOutcome(
            evictedExpertCount: evictedExpertCount,
            vacatedSlots: vacatedSlots)
    }

    /// Ascending demand-per-byte order via 128-bit cross multiplication,
    /// tie-broken by slot then expert id.
    private static func candidateOrder(
        leftCandidate: DecodeEvictionCandidate,
        rightCandidate: DecodeEvictionCandidate
    ) -> ComparisonResult {
        let leftScore: (UInt64, UInt64) =
            leftCandidate.demand.multipliedFullWidth(by: rightCandidate.payloadBytes)
        let rightScore: (UInt64, UInt64) =
            rightCandidate.demand.multipliedFullWidth(by: leftCandidate.payloadBytes)
        if leftScore == rightScore {
            if leftCandidate.slot != rightCandidate.slot {
                return leftCandidate.slot < rightCandidate.slot
                    ? .orderedAscending : .orderedDescending
            }
            if leftCandidate.expertId != rightCandidate.expertId {
                return leftCandidate.expertId < rightCandidate.expertId
                    ? .orderedAscending : .orderedDescending
            }
            return .orderedSame
        }
        return leftScore < rightScore ? .orderedAscending : .orderedDescending
    }
}
