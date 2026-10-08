import Foundation

/// Ceiling that fits static expert weights but not activation headroom.
///
/// Callers supply artifact geometry and the startup headroom policy already
/// decided; this type knows no model ids or gigabyte souvenirs.
public struct CompleteResidencyHeadroomBoundary: Equatable, Sendable {

    /// Model core plus complete expert payload, with no activation reserve.
    public let staticCompleteResidencyBytes: UInt64

    /// Exact complete expert payload; zero means this is not a sparse MoE
    /// boundary.
    public let completeExpertPayloadBytes: UInt64

    /// Startup activation headroom required before complete residency may
    /// promote.
    public let requiredHeadroomBytes: UInt64

    /// Composes the boundary from validated payload geometry and policy
    /// headroom.
    public init(
        fromModelGeometry geometry: MlxRamBudgetModelGeometry,
        requiredHeadroomBytes: UInt64
    ) {
        self.staticCompleteResidencyBytes = SaturatingArithmetic.add(
            geometry.modelCorePayloadBytes,
            geometry.completeExpertPayloadBytes)
        self.completeExpertPayloadBytes = geometry.completeExpertPayloadBytes
        self.requiredHeadroomBytes = requiredHeadroomBytes
    }

    /// Ceiling that still covers the expert payload but not expert payload
    /// plus headroom. Startup admission projects
    /// `idleActive + completeExperts + headroom`; idle active is measured
    /// after load and is often smaller than disk core bytes, so this ceiling
    /// is anchored on the expert payload, not core plus experts. Returns nil
    /// when the gap does not exist.
    public func pagingCeilingBytes() -> UInt64? {
        if self.completeExpertPayloadBytes == 0 || self.requiredHeadroomBytes == 0 {
            return nil
        }
        return SaturatingArithmetic.subtract(
            SaturatingArithmetic.add(
                self.completeExpertPayloadBytes,
                self.requiredHeadroomBytes),
            1)
    }
}
