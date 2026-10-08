import Foundation

/// Exact immutable geometry for one sparse decoder layer.
public struct ExpertLayerGeometry: Equatable, Sendable {

    /// Contiguous decoder layer position this geometry describes.
    public var layerIndex: Int

    /// Payload bytes of every expert in the layer combined.
    public var completeLayerPayloadBytes: UInt64

    /// Payload bytes of one expert in this layer.
    public var expertPayloadBytes: UInt64

    /// Total expert count the layer's router can choose from.
    public var expertCapacity: Int

    /// Experts each token routes to in this layer.
    public var expertsPerToken: Int

    public init(
        layerIndex: Int,
        completeLayerPayloadBytes: UInt64,
        expertPayloadBytes: UInt64,
        expertCapacity: Int,
        expertsPerToken: Int
    ) {
        self.layerIndex = layerIndex
        self.completeLayerPayloadBytes = completeLayerPayloadBytes
        self.expertPayloadBytes = expertPayloadBytes
        self.expertCapacity = expertCapacity
        self.expertsPerToken = expertsPerToken
    }
}
