import Foundation

/// Computes retained-expert capacity while preserving future page and
/// transient work.
public enum RetainedExpertPayloadCapacity {

    /// Retained-expert bytes the ceiling can still hold after this load.
    ///
    /// When nothing paged is pending, the future page reserve is the larger of
    /// the pending allocation and one maximum page; otherwise both must fit
    /// together. Overflowing projections collapse to zero capacity instead of
    /// wrapping.
    public static func bytes(
        activeMemoryBytes: UInt64,
        activeMemoryCeilingBytes: UInt64,
        maximumExpertPageBytes: UInt64,
        pendingAllocationBytes: UInt64,
        observedTransientHighWaterBytes: UInt64,
        currentRetainedExpertPayloadBytes: UInt64,
        pendingRetainedExpertPayloadBytes: UInt64
    ) -> UInt64 {
        let futurePageReserveBytes: UInt64
        if pendingRetainedExpertPayloadBytes == 0 {
            futurePageReserveBytes = max(pendingAllocationBytes, maximumExpertPageBytes)
        } else {
            let (combinedReserveBytes, combinedOverflowed) =
                pendingAllocationBytes.addingReportingOverflow(maximumExpertPageBytes)
            if combinedOverflowed {
                return 0
            }
            futurePageReserveBytes = combinedReserveBytes
        }
        let (postLoadRetainedPayloadBytes, retainedOverflowed) =
            currentRetainedExpertPayloadBytes.addingReportingOverflow(
                pendingRetainedExpertPayloadBytes)
        if retainedOverflowed {
            return 0
        }
        let (liveReservedBytes, liveOverflowed) =
            activeMemoryBytes.addingReportingOverflow(futurePageReserveBytes)
        let effectiveLiveReservedBytes: UInt64 = liveOverflowed ? UInt64.max : liveReservedBytes
        let effectiveCeilingBytes: UInt64 = SaturatingArithmetic.subtract(
            activeMemoryCeilingBytes,
            observedTransientHighWaterBytes)
        if effectiveLiveReservedBytes <= effectiveCeilingBytes {
            return SaturatingArithmetic.add(
                postLoadRetainedPayloadBytes,
                effectiveCeilingBytes - effectiveLiveReservedBytes)
        }
        return SaturatingArithmetic.subtract(
            postLoadRetainedPayloadBytes,
            effectiveLiveReservedBytes - effectiveCeilingBytes)
    }
}
