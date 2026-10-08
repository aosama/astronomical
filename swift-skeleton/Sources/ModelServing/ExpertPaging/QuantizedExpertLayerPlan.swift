import Foundation

import MLX

/// Startup-validated tensor geometry for one MoE layer, reused by every
/// decode-time page. Port of the Rust `QuantizedExpertLayerPlan`: the
/// layer plan freezes the packed geometry so expert pages assemble from
/// bounded reads with no per-request arithmetic beyond byte counting.
public struct QuantizedExpertLayerPlan: Sendable {

    /// Module prefix of the layer this plan describes.
    public var layerPrefix: String

    /// Validated tensor sources the layer's experts read from.
    public var tensorSources: [QuantizedTensorSource]

    /// Number of experts the layer stores.
    public var expertCapacity: Int

    /// Storage bit width of the layer's packed expert payloads.
    public var quantizationBits: Int32

    /// Affine group size of the layer's packed expert payloads.
    public var quantizationGroupSize: Int32

    /// Unanimous storage when every projection matches, affine when any
    /// projection is affine. Execution must consult
    /// `quantizationModeForProjection` because mixed OptiQ layers can
    /// contain both encodings.
    public var quantizationMode: ExpertLayerQuantizationMode

    /// Projection-specific encoding, needed for mixed-storage layers.
    public var quantizationModeByProjectionName: [String: ExpertLayerQuantizationMode]

    public init(
        layerPrefix: String,
        tensorSources: [QuantizedTensorSource],
        expertCapacity: Int,
        quantizationBits: Int32,
        quantizationGroupSize: Int32,
        quantizationMode: ExpertLayerQuantizationMode,
        quantizationModeByProjectionName: [String: ExpertLayerQuantizationMode] = [:]
    ) {
        self.layerPrefix = layerPrefix
        self.tensorSources = tensorSources
        self.expertCapacity = expertCapacity
        self.quantizationBits = quantizationBits
        self.quantizationGroupSize = quantizationGroupSize
        self.quantizationMode = quantizationMode
        self.quantizationModeByProjectionName = quantizationModeByProjectionName
    }

    /// Returns the encoding used by one projection, falling back to the layer's unanimous mode.
    public func quantizationModeForProjection(projectionName: String) -> ExpertLayerQuantizationMode {
        return self.quantizationModeByProjectionName[projectionName] ?? self.quantizationMode
    }

    /**
     * The exact packed payload bytes one expert occupies across every
     * tensor of the layer.
     *
     * - Returns: The per-expert payload byte count.
     * - Throws: `ExpertPagingError.expertPayloadAccountingOverflow` when
     *   the sum overflows 64-bit accounting.
     */
    public func expertPayloadByteCount() throws -> UInt64 {
        var payloadByteTotal: UInt64 = 0
        for tensorSource: QuantizedTensorSource in self.tensorSources {
            let tensorBytes: UInt64 = UInt64(tensorSource.bytesPerExpert)
            let (summedTotal, sumOverflowed) = payloadByteTotal.addingReportingOverflow(tensorBytes)
            if sumOverflowed {
                throw ExpertPagingError.expertPayloadAccountingOverflow(layerPrefix: self.layerPrefix)
            }
            payloadByteTotal = summedTotal
        }
        return payloadByteTotal
    }

    /**
     * The exact payload bytes a complete layer page (every expert) spans.
     *
     * - Returns: The complete-layer payload byte count.
     * - Throws: `ExpertPagingError.completeLayerPayloadAccountingOverflow`
     *   when the product overflows 64-bit accounting.
     */
    public func completeExpertPayloadByteCount() throws -> UInt64 {
        let perExpertByteCount: UInt64 = try self.expertPayloadByteCount()
        let (layerTotal, productOverflowed) = perExpertByteCount
            .multipliedReportingOverflow(by: UInt64(self.expertCapacity))
        if productOverflowed {
            throw ExpertPagingError.completeLayerPayloadAccountingOverflow(layerPrefix: self.layerPrefix)
        }
        return layerTotal
    }
}
