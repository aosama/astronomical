import Foundation

/// Checked payload and alignment arithmetic over one validated
/// decoder-cache layout, port of the Rust `storage_geometry` module.
extension DecoderCacheLayout {

    /// Returns whether this model persists append-only sequence state.
    public var hasSequenceState: Bool {
        return sequenceTensorCount > 0
    }

    /// Returns whether this model persists complete-boundary state.
    public var hasBoundaryState: Bool {
        return boundaryTensorCount > 0
    }

    /// Returns the total exact sequence-state payload bytes added by one token.
    public func sequenceStatePayloadByteCountPerToken() throws -> Int {
        var sequencePayloadBytesPerToken = 0
        for persistedTensorLayout in sequenceTensorLayouts() {
            let tensorPayloadBytesPerToken = try persistedTensorLayout.tensorLayout
                .sequencePayloadByteCountPerToken()
            let (summedPayloadBytesPerToken, additionOverflowed) =
                sequencePayloadBytesPerToken.addingReportingOverflow(tensorPayloadBytesPerToken)
            if additionOverflowed {
                throw DecoderCacheLayoutError.sequenceStatePayloadByteCountPerTokenOverflow
            }
            sequencePayloadBytesPerToken = summedPayloadBytesPerToken
        }
        return sequencePayloadBytesPerToken
    }

    /// Returns the largest exact payload owned by one sequence tensor at the
    /// requested length.
    public func maximumSequenceTensorPayloadByteCount(
        sequenceTokenCount: Int
    ) throws -> Int {
        var maximumTensorPayloadBytes = 0
        for persistedTensorLayout in sequenceTensorLayouts() {
            let payloadBytesPerToken = try persistedTensorLayout.tensorLayout
                .sequencePayloadByteCountPerToken()
            let (tensorPayloadBytes, multiplicationOverflowed) =
                payloadBytesPerToken.multipliedReportingOverflow(by: sequenceTokenCount)
            if multiplicationOverflowed {
                throw DecoderCacheLayoutError.sequenceTensorPayloadByteCountOverflow
            }
            maximumTensorPayloadBytes = max(maximumTensorPayloadBytes, tensorPayloadBytes)
        }
        return maximumTensorPayloadBytes
    }

    /// Returns the largest source payload live during one incremental cache
    /// restore step.
    ///
    /// A restore loads one sequence block (every sequence tensor at block
    /// length) and, separately, the complete boundary snapshot (every
    /// boundary tensor). Because the two load at separate times, admission
    /// needs the larger source rather than the sum of both.
    public func incrementalRestoreSourceWorkspaceByteCount(
        sequenceBlockTokenCount: Int
    ) throws -> Int {
        let payloadBytesPerToken = try sequenceStatePayloadByteCountPerToken()
        let (sequenceSourceBytes, multiplicationOverflowed) =
            payloadBytesPerToken.multipliedReportingOverflow(by: sequenceBlockTokenCount)
        if multiplicationOverflowed {
            throw DecoderCacheLayoutError.sequenceTensorPayloadByteCountOverflow
        }
        let boundarySourceBytes = try boundarySnapshotPayloadByteCount()
        return max(sequenceSourceBytes, boundarySourceBytes)
    }

    /// Returns payload bytes for one complete boundary snapshot.
    public func boundarySnapshotPayloadByteCount() throws -> Int {
        var boundaryPayloadBytes = 0
        for persistedTensorLayout in boundaryTensorLayouts() {
            let tensorPayloadBytes = try persistedTensorLayout.tensorLayout.fixedPayloadByteCount()
            let (summedPayloadBytes, additionOverflowed) =
                boundaryPayloadBytes.addingReportingOverflow(tensorPayloadBytes)
            if additionOverflowed {
                throw DecoderCacheLayoutError.boundarySnapshotPayloadByteCountOverflow
            }
            boundaryPayloadBytes = summedPayloadBytes
        }
        return boundaryPayloadBytes
    }

    /// Returns payload bytes for one complete persistent model-state capture.
    public func persistentPromptCacheBlockPayloadByteCount(
        blockTokenCount: Int
    ) throws -> Int {
        let payloadBytesPerToken = try sequenceStatePayloadByteCountPerToken()
        let (sequencePayloadBytes, blockMultiplicationOverflowed) =
            payloadBytesPerToken.multipliedReportingOverflow(by: blockTokenCount)
        if blockMultiplicationOverflowed {
            throw DecoderCacheLayoutError.persistentPromptCacheBlockPayloadByteCountOverflow
        }
        let (blockPayloadBytes, snapshotAdditionOverflowed) =
            sequencePayloadBytes.addingReportingOverflow(try boundarySnapshotPayloadByteCount())
        if snapshotAdditionOverflowed {
            throw DecoderCacheLayoutError.persistentPromptCacheBlockPayloadByteCountOverflow
        }
        return blockPayloadBytes
    }

    /// Returns the natural token alignment shared by every append-only
    /// state component.
    public func persistenceAlignmentTokenCount() throws -> Int {
        var persistenceAlignmentTokenCount = 1
        for decoderLayerIndex in 0..<layerCount {
            if let decoderLayerLayout = layer(decoderLayerIndex) {
                persistenceAlignmentTokenCount = try checkedLayerPersistenceAlignment(
                    decoderLayerLayout,
                    currentAlignmentTokenCount: persistenceAlignmentTokenCount)
            }
        }
        return persistenceAlignmentTokenCount
    }

    private func checkedLayerPersistenceAlignment(
        _ decoderLayerLayout: DecoderCacheLayerLayout,
        currentAlignmentTokenCount: Int
    ) throws -> Int {
        switch decoderLayerLayout {
        case .appendOnlyAttention(_, _, let capacityGrowthTokens):
            return try Self.checkedLeastCommonMultiple(
                firstTokenCount: currentAlignmentTokenCount,
                secondTokenCount: capacityGrowthTokens)
        case .rotatingWindowAttention:
            return currentAlignmentTokenCount
        case .recurrentTensor:
            return currentAlignmentTokenCount
        case .composite(let components):
            var componentAlignmentTokenCount = currentAlignmentTokenCount
            for componentLayout in components {
                componentAlignmentTokenCount = try checkedLayerPersistenceAlignment(
                    componentLayout,
                    currentAlignmentTokenCount: componentAlignmentTokenCount)
            }
            return componentAlignmentTokenCount
        }
    }

    private static func checkedLeastCommonMultiple(
        firstTokenCount: Int,
        secondTokenCount: Int
    ) throws -> Int {
        let greatestCommonDivisorTokenCount = greatestCommonDivisor(
            firstTokenCount,
            secondTokenCount)
        let reducedFirstTokenCount = firstTokenCount / greatestCommonDivisorTokenCount
        let (leastCommonMultipleTokenCount, multiplicationOverflowed) =
            reducedFirstTokenCount.multipliedReportingOverflow(by: secondTokenCount)
        if multiplicationOverflowed {
            throw DecoderCacheLayoutError.persistenceAlignmentTokenCountOverflow
        }
        return leastCommonMultipleTokenCount
    }

    private static func greatestCommonDivisor(
        _ firstTokenCount: Int,
        _ secondTokenCount: Int
    ) -> Int {
        var remainingFirstTokenCount = firstTokenCount
        var remainingSecondTokenCount = secondTokenCount
        while remainingSecondTokenCount != 0 {
            let remainderTokenCount = remainingFirstTokenCount % remainingSecondTokenCount
            remainingFirstTokenCount = remainingSecondTokenCount
            remainingSecondTokenCount = remainderTokenCount
        }
        return remainingFirstTokenCount
    }
}
