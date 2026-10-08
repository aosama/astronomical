import Foundation

/// Exact class-specific ownership released under memory pressure.
public struct RetainedExpertReclamation: Equatable, Sendable {

    /// Elastic routed pages dropped.
    public var releasedPartialLayerCount: Int

    /// Payload bytes freed by dropped elastic routed pages.
    public var releasedPartialPayloadBytes: UInt64

    /// Stable complete layers dropped.
    public var releasedCompleteLayerCount: Int

    /// Payload bytes freed by dropped stable complete layers.
    public var releasedCompletePayloadBytes: UInt64

    public init(
        releasedPartialLayerCount: Int = 0,
        releasedPartialPayloadBytes: UInt64 = 0,
        releasedCompleteLayerCount: Int = 0,
        releasedCompletePayloadBytes: UInt64 = 0
    ) {
        self.releasedPartialLayerCount = releasedPartialLayerCount
        self.releasedPartialPayloadBytes = releasedPartialPayloadBytes
        self.releasedCompleteLayerCount = releasedCompleteLayerCount
        self.releasedCompletePayloadBytes = releasedCompletePayloadBytes
    }

    /// Total payload bytes freed across both page classes.
    public func releasedPayloadBytes() -> UInt64 {
        let (combinedTotal, addOverflow) =
            self.releasedPartialPayloadBytes.addingReportingOverflow(self.releasedCompletePayloadBytes)
        if addOverflow {
            return UInt64.max
        }
        return combinedTotal
    }
}
