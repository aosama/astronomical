import Foundation

/// Planner-ready ownership metadata for one retained layer, without
/// exposing the page payloads themselves.
public struct CurrentExpertLayerResidency: Equatable, Sendable {

    /// Decoder layer this residency describes.
    public var layerIndex: Int

    /// Explicit ownership class of the retained page.
    public var pageClass: RetainedExpertPageClass

    /// Expert identifiers covered by the retained page.
    public var retainedExpertIds: [Int]

    /// Payload bytes the page occupies in wired memory.
    public var payloadBytes: UInt64

    /// Observed weighted route demand covered by the retained experts.
    public var coveredWeightedDemand: UInt64

    public init(
        layerIndex: Int,
        pageClass: RetainedExpertPageClass,
        retainedExpertIds: [Int],
        payloadBytes: UInt64,
        coveredWeightedDemand: UInt64
    ) {
        self.layerIndex = layerIndex
        self.pageClass = pageClass
        self.retainedExpertIds = retainedExpertIds
        self.payloadBytes = payloadBytes
        self.coveredWeightedDemand = coveredWeightedDemand
    }
}
