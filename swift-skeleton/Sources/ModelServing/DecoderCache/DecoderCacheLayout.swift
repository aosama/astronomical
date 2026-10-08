import Foundation

/// Validated, architecture-neutral request decoder-cache layout, port of
/// the Rust `DecoderCacheLayout`.
public struct DecoderCacheLayout: Equatable, Sendable {

    private let layerLayouts: [DecoderCacheLayerLayout]

    private let sequenceTensorCountValue: Int

    private let boundaryTensorCountValue: Int

    /// Validates a model-owned decoder-cache layout before it creates live
    /// state or opens SSD files.
    public init(layers: [DecoderCacheLayerLayout]) throws {
        var sequenceTensorCount = 0
        var boundaryTensorCount = 0
        for (layerIndex, layerLayout) in layers.enumerated() {
            var layerTensorRoleNames = Set<String>()
            try DecoderCacheLayout.validateLayerLayout(
                layerLayout,
                layerIndex: layerIndex,
                layerTensorRoleNames: &layerTensorRoleNames,
                sequenceTensorCount: &sequenceTensorCount,
                boundaryTensorCount: &boundaryTensorCount)
        }
        self.layerLayouts = layers
        self.sequenceTensorCountValue = sequenceTensorCount
        self.boundaryTensorCountValue = boundaryTensorCount
    }

    /// The number of decoder layers with cache state.
    public var layerCount: Int {
        return layerLayouts.count
    }

    /// Returns one model-owned layer layout by its decoder position.
    public func layer(_ decoderLayerIndex: Int) -> DecoderCacheLayerLayout? {
        guard layerLayouts.indices.contains(decoderLayerIndex) else {
            return nil
        }
        return layerLayouts[decoderLayerIndex]
    }

    /// The number of tensors persisted in every sequence-state block.
    public var sequenceTensorCount: Int {
        return sequenceTensorCountValue
    }

    /// The number of tensors persisted in every complete boundary snapshot.
    public var boundaryTensorCount: Int {
        return boundaryTensorCountValue
    }

    private static func validateLayerLayout(
        _ layerLayout: DecoderCacheLayerLayout,
        layerIndex: Int,
        layerTensorRoleNames: inout Set<String>,
        sequenceTensorCount: inout Int,
        boundaryTensorCount: inout Int
    ) throws {
        switch layerLayout {
        case .appendOnlyAttention(let keys, let values, let capacityGrowthTokens):
            if capacityGrowthTokens == 0 {
                throw DecoderCacheLayoutError.zeroCapacityGrowthTokens(layerIndex: layerIndex)
            }
            try validateSequenceTensor(
                keys, layerIndex: layerIndex, layerTensorRoleNames: &layerTensorRoleNames)
            try validateSequenceTensor(
                values, layerIndex: layerIndex, layerTensorRoleNames: &layerTensorRoleNames)
            if keys.dimensions != values.dimensions || keys.sequenceAxis != values.sequenceAxis {
                throw DecoderCacheLayoutError.attentionTensorContractMismatch(
                    layerIndex: layerIndex)
            }
            sequenceTensorCount += 2
        case .rotatingWindowAttention(let keys, let values, let windowSize):
            if windowSize == 0 {
                throw DecoderCacheLayoutError.zeroRotatingWindowSize(layerIndex: layerIndex)
            }
            try validateBoundaryTensor(
                keys, layerIndex: layerIndex, layerTensorRoleNames: &layerTensorRoleNames)
            try validateBoundaryTensor(
                values, layerIndex: layerIndex, layerTensorRoleNames: &layerTensorRoleNames)
            if keys.dimensions != values.dimensions
                || keys.dimensions.dropFirst(2).first != windowSize {
                throw DecoderCacheLayoutError.attentionTensorContractMismatch(
                    layerIndex: layerIndex)
            }
            boundaryTensorCount += 2
            for counterLayout in DecoderCacheLayerLayout.rotatingWindowCounterLayouts() {
                try validateBoundaryTensor(
                    counterLayout,
                    layerIndex: layerIndex,
                    layerTensorRoleNames: &layerTensorRoleNames)
                boundaryTensorCount += 1
            }
        case .recurrentTensor(let tensor):
            try validateBoundaryTensor(
                tensor, layerIndex: layerIndex, layerTensorRoleNames: &layerTensorRoleNames)
            boundaryTensorCount += 1
        case .composite(let components):
            if components.isEmpty {
                throw DecoderCacheLayoutError.emptyComposite(layerIndex: layerIndex)
            }
            for componentLayout in components {
                try validateLayerLayout(
                    componentLayout,
                    layerIndex: layerIndex,
                    layerTensorRoleNames: &layerTensorRoleNames,
                    sequenceTensorCount: &sequenceTensorCount,
                    boundaryTensorCount: &boundaryTensorCount)
            }
        }
    }

    private static func validateSequenceTensor(
        _ tensorLayout: DecoderCacheTensorLayout,
        layerIndex: Int,
        layerTensorRoleNames: inout Set<String>
    ) throws {
        guard let sequenceAxis = tensorLayout.sequenceAxis else {
            throw DecoderCacheLayoutError.sequenceTensorMissingAxis(
                layerIndex: layerIndex,
                tensorRoleName: tensorLayout.tensorRoleName)
        }
        try validateTensorRoleAndDimensions(
            tensorLayout, layerIndex: layerIndex, layerTensorRoleNames: &layerTensorRoleNames)
        if sequenceAxis >= tensorLayout.dimensions.count {
            throw DecoderCacheLayoutError.sequenceAxisOutsideTensorRank(
                layerIndex: layerIndex,
                tensorRoleName: tensorLayout.tensorRoleName,
                sequenceAxis: sequenceAxis,
                tensorRank: tensorLayout.dimensions.count)
        }
        if tensorLayout.dimensions[sequenceAxis] != 0 {
            throw DecoderCacheLayoutError.sequenceAxisMustUseDynamicDimension(
                layerIndex: layerIndex,
                tensorRoleName: tensorLayout.tensorRoleName)
        }
    }

    private static func validateBoundaryTensor(
        _ tensorLayout: DecoderCacheTensorLayout,
        layerIndex: Int,
        layerTensorRoleNames: inout Set<String>
    ) throws {
        if tensorLayout.sequenceAxis != nil {
            throw DecoderCacheLayoutError.boundaryTensorHasSequenceAxis(
                layerIndex: layerIndex,
                tensorRoleName: tensorLayout.tensorRoleName)
        }
        try validateTensorRoleAndDimensions(
            tensorLayout, layerIndex: layerIndex, layerTensorRoleNames: &layerTensorRoleNames)
        if tensorLayout.dimensions.contains(0) {
            throw DecoderCacheLayoutError.boundaryTensorHasDynamicDimension(
                layerIndex: layerIndex,
                tensorRoleName: tensorLayout.tensorRoleName)
        }
    }

    private static func validateTensorRoleAndDimensions(
        _ tensorLayout: DecoderCacheTensorLayout,
        layerIndex: Int,
        layerTensorRoleNames: inout Set<String>
    ) throws {
        if tensorLayout.tensorRoleName.isEmpty {
            throw DecoderCacheLayoutError.emptyTensorRole(layerIndex: layerIndex)
        }
        if tensorLayout.dimensions.isEmpty {
            throw DecoderCacheLayoutError.zeroTensorRank(
                layerIndex: layerIndex,
                tensorRoleName: tensorLayout.tensorRoleName)
        }
        if !layerTensorRoleNames.insert(tensorLayout.tensorRoleName).inserted {
            throw DecoderCacheLayoutError.duplicateTensorRole(
                layerIndex: layerIndex,
                tensorRoleName: tensorLayout.tensorRoleName)
        }
    }
}
