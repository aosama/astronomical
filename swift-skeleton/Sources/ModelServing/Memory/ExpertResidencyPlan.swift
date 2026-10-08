import Foundation

/// One deterministic topology plan that never commands eager source reads.
public struct ExpertResidencyPlan: Equatable, Sendable {

    /// Lifecycle position this plan was composed for.
    public var phase: MemoryPhase

    /// Composed retained-expert byte ceiling the plan respects.
    public var retainedExpertCeilingBytes: UInt64

    /// Layers planned to hold their complete expert payload.
    public var completeLayerTargets: [Int]

    /// One target per sparse decoder layer, in layer order.
    public var layerTargets: [ExpertLayerResidencyTarget]

    /// Bytes reserved for routed-expert floors on non-complete layers.
    public var reservedRoutedOverlayBytes: UInt64

    /// Current ownership bytes the plan expects to preserve.
    public var expectedPreservedBytes: UInt64

    /// New retained bytes the plan may admit beyond preserved ownership.
    public var maximumNewRetainedBytes: UInt64

    /// Layers in the deterministic release order, partials first.
    public var deterministicReleaseOrder: [Int]

    /// Whether the plan fell back to partial-only mode because routed
    /// floors alone exceed the ceiling.
    public var isLowBudgetPartialMode: Bool

    public init(
        phase: MemoryPhase,
        retainedExpertCeilingBytes: UInt64,
        completeLayerTargets: [Int],
        layerTargets: [ExpertLayerResidencyTarget],
        reservedRoutedOverlayBytes: UInt64,
        expectedPreservedBytes: UInt64,
        maximumNewRetainedBytes: UInt64,
        deterministicReleaseOrder: [Int],
        isLowBudgetPartialMode: Bool
    ) {
        self.phase = phase
        self.retainedExpertCeilingBytes = retainedExpertCeilingBytes
        self.completeLayerTargets = completeLayerTargets
        self.layerTargets = layerTargets
        self.reservedRoutedOverlayBytes = reservedRoutedOverlayBytes
        self.expectedPreservedBytes = expectedPreservedBytes
        self.maximumNewRetainedBytes = maximumNewRetainedBytes
        self.deterministicReleaseOrder = deterministicReleaseOrder
        self.isLowBudgetPartialMode = isLowBudgetPartialMode
    }
}
