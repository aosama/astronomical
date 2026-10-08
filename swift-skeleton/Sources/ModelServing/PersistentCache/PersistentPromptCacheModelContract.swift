import Foundation;

import CryptoKit;
import ModelServing;

/// Immutable storage and memory geometry for one model revision, port of
/// the Rust `PersistentPromptCacheModelContract`. Resolution happens once
/// from validated decoder layout and live machine/user budgets; every later
/// lookup, write, quota decision, and memory admission uses this same
/// contract so no component invents its own block size or tensor shape.
public struct PersistentPromptCacheModelContract: Equatable, Sendable {

    private let modelIdValue: String;
    private let modelRevisionValue: String;
    private let decoderCacheLayoutValue: DecoderCacheLayout;
    private let maximumContextTokenCountValue: Int;
    private let effectiveMlxMemoryCeilingBytesValue: UInt64;
    private let blockTokenCountValue: Int;
    private let commonPrefixCheckpointStrideBlocksValue: UInt32;
    private let sequenceStatePayloadBytesPerTokenValue: Int;
    private let sequenceStatePayloadBytesPerBlockValue: Int;
    private let boundaryStatePayloadBytesValue: Int;
    private let capturePayloadBytesValue: Int;
    private let sequenceStateFileBytesValue: UInt64;
    private let boundaryStateFileBytesValue: UInt64;
    private let maximumBlockManifestFileBytesValue: UInt64;
    private let maximumCommittedBlockBytesValue: UInt64;
    private let directPublicationWorkspaceBytesValue: Int;
    private let storageContractFingerprintData: Data;

    /// Resolves one deterministic exact-state storage policy for a loaded
    /// model and its budgets.
    public static func resolve(
        modelId: String,
        modelRevision: String,
        decoderCacheLayout: DecoderCacheLayout,
        maximumContextTokenCount: Int,
        effectiveMlxMemoryCeilingBytes: UInt64,
        globalSsdQuotaBytes: UInt64,
        configuredBlockTokenCount: Int?,
        commonPrefixCheckpointStrideBlocks: UInt32
    ) throws -> PersistentPromptCacheModelContract {
        if modelId.isEmpty {
            throw PersistentPromptCacheModelContractError.emptyModelId;
        }
        if modelRevision.isEmpty {
            throw PersistentPromptCacheModelContractError.emptyModelRevision;
        }
        if maximumContextTokenCount == 0 {
            throw PersistentPromptCacheModelContractError.zeroMaximumContextTokenCount;
        }
        if commonPrefixCheckpointStrideBlocks == 0 {
            throw PersistentPromptCacheModelContractError
                .zeroCommonPrefixCheckpointStrideBlocks;
        }
        if decoderCacheLayout.hasSequenceState == false
            && decoderCacheLayout.hasBoundaryState == false {
            throw PersistentPromptCacheModelContractError.noPersistentState;
        }

        let sequenceStatePayloadBytesPerToken: Int;
        let boundaryStatePayloadBytes: Int;
        let persistenceAlignmentTokenCount: Int;
        do {
            sequenceStatePayloadBytesPerToken = try decoderCacheLayout
                .sequenceStatePayloadByteCountPerToken();
            boundaryStatePayloadBytes = try decoderCacheLayout
                .boundarySnapshotPayloadByteCount();
            persistenceAlignmentTokenCount = try decoderCacheLayout
                .persistenceAlignmentTokenCount();
        } catch let layoutError as DecoderCacheLayoutError {
            throw PersistentPromptCacheModelContractError.decoderCacheLayout(layoutError);
        }
        // The block length belongs to the immutable artifact contract, not a
        // deployment-wide tuning constant. Matching append-only allocation
        // growth avoids capture boundaries that force incompatible state
        // reshaping, while the quota-derived size prevents a laptop with less
        // SSD capacity from accepting a state block it can never retain.
        let blockTokenCountIsUserConfigured: Bool = configuredBlockTokenCount != nil;
        var blockTokenCount: Int;
        switch configuredBlockTokenCount {
        case .some(let userConfiguredBlockTokenCount)
            where userConfiguredBlockTokenCount > 0
                && userConfiguredBlockTokenCount <= maximumContextTokenCount
                && userConfiguredBlockTokenCount % persistenceAlignmentTokenCount == 0:
            blockTokenCount = userConfiguredBlockTokenCount;
        case .some:
            throw PersistentPromptCacheModelContractError.invalidConfiguredBlockTokenCount(
                requiredAlignmentTokens: persistenceAlignmentTokenCount,
                maximumContextTokens: maximumContextTokenCount);
        case .none:
            blockTokenCount = try PersistentPromptCacheModelContract.deriveBlockTokenCount(
                maximumContextTokenCount: maximumContextTokenCount,
                persistenceAlignmentTokenCount: persistenceAlignmentTokenCount,
                sequenceStatePayloadBytesPerToken: sequenceStatePayloadBytesPerToken,
                boundaryStatePayloadBytes: boundaryStatePayloadBytes,
                globalSsdQuotaBytes: globalSsdQuotaBytes);
        }
        let maximumBlockManifestFileBytes: UInt64 = try PersistentPromptCacheStorageGeometry
            .maximumBlockManifestFileBytes(maximumContextTokenCount: maximumContextTokenCount);
        // Automatic sizing may need a larger aligned block to amortize
        // repeated manifests and recurrent snapshots across a maximum-length
        // chain. An explicit user block length is different: it is part of
        // the requested storage topology and must either fit exactly or fail
        // clearly. Silently changing it would make status, fingerprints, and
        // observed boundaries disagree with configuration.
        var sequenceStateFileBytes: UInt64 = 0;
        var boundaryStateFileBytes: UInt64 = 0;
        var maximumCommittedBlockBytes: UInt64 = 0;
        while true {
            do {
                sequenceStateFileBytes = try PersistentPromptCacheStorageGeometry
                    .exactStateFileBytes(
                        blockTokenCount: blockTokenCount,
                        persistedTensorLayouts: decoderCacheLayout.sequenceTensorLayouts());
                boundaryStateFileBytes = try PersistentPromptCacheStorageGeometry
                    .exactStateFileBytes(
                        blockTokenCount: blockTokenCount,
                        persistedTensorLayouts: decoderCacheLayout.boundaryTensorLayouts());
            } catch let layoutError as DecoderCacheLayoutError {
                throw PersistentPromptCacheModelContractError.decoderCacheLayout(layoutError);
            }
            let (stateFileBytes, stateOverflow) = sequenceStateFileBytes
                .addingReportingOverflow(boundaryStateFileBytes);
            let (committedBlockBytes, committedOverflow) = stateFileBytes
                .addingReportingOverflow(maximumBlockManifestFileBytes);
            if stateOverflow || committedOverflow {
                throw PersistentPromptCacheModelContractError.capturePayloadByteCountOverflow;
            }
            maximumCommittedBlockBytes = committedBlockBytes;
            let maximumCommittedBlockCount: Int = try PersistentPromptCacheModelContract
                .checkedCeilingDivision(
                    dividend: maximumContextTokenCount, divisor: blockTokenCount);
            let (maximumChainBytes, chainOverflow) = UInt64(maximumCommittedBlockCount)
                .multipliedReportingOverflow(by: maximumCommittedBlockBytes);
            if chainOverflow {
                throw PersistentPromptCacheModelContractError.capturePayloadByteCountOverflow;
            }
            if maximumChainBytes <= globalSsdQuotaBytes {
                break;
            }
            if blockTokenCountIsUserConfigured {
                throw PersistentPromptCacheModelContractError
                    .configuredBlockChainExceedsSsdQuota(
                        configuredBlockTokens: blockTokenCount,
                        maximumChainBytes: maximumChainBytes,
                        globalSsdQuotaBytes: globalSsdQuotaBytes);
            }
            let (grownBlockTokenCount, growthOverflow) = blockTokenCount
                .addingReportingOverflow(persistenceAlignmentTokenCount);
            if growthOverflow {
                throw PersistentPromptCacheModelContractError.capturePayloadByteCountOverflow;
            }
            let nextBlockTokenCount: Int = min(grownBlockTokenCount, maximumContextTokenCount);
            if nextBlockTokenCount == blockTokenCount {
                throw PersistentPromptCacheModelContractError.blockFilesExceedSsdQuota(
                    blockFileBytes: maximumCommittedBlockBytes,
                    globalSsdQuotaBytes: globalSsdQuotaBytes);
            }
            blockTokenCount = nextBlockTokenCount;
        }
        let (sequenceStatePayloadBytesPerBlock, perBlockOverflow) = sequenceStatePayloadBytesPerToken
            .multipliedReportingOverflow(by: blockTokenCount);
        if perBlockOverflow {
            throw PersistentPromptCacheModelContractError
                .sequenceStateBlockPayloadByteCountOverflow;
        }
        let (capturePayloadBytes, captureOverflow) = sequenceStatePayloadBytesPerBlock
            .addingReportingOverflow(boundaryStatePayloadBytes);
        if captureOverflow {
            throw PersistentPromptCacheModelContractError.capturePayloadByteCountOverflow;
        }
        // Native safetensors publication evaluates and copies one tensor at a
        // time. Peak *additional* workspace is therefore the largest
        // individual tensor, not the sum of captured decoder state already
        // owned by the request.
        let maximumSequenceTensorPayloadBytes: Int;
        do {
            maximumSequenceTensorPayloadBytes = try decoderCacheLayout
                .maximumSequenceTensorPayloadByteCount(sequenceTokenCount: blockTokenCount);
        } catch let layoutError as DecoderCacheLayoutError {
            throw PersistentPromptCacheModelContractError.decoderCacheLayout(layoutError);
        }
        var maximumBoundaryTensorPayloadBytes: Int = 0;
        for persistedTensorLayout: DecoderCachePersistedTensorLayout
            in decoderCacheLayout.boundaryTensorLayouts() {
            let fixedPayloadBytes: Int;
            do {
                fixedPayloadBytes = try persistedTensorLayout.tensorLayout
                    .fixedPayloadByteCount();
            } catch let layoutError as DecoderCacheLayoutError {
                throw PersistentPromptCacheModelContractError.decoderCacheLayout(layoutError);
            }
            maximumBoundaryTensorPayloadBytes = max(maximumBoundaryTensorPayloadBytes, fixedPayloadBytes);
        }
        let directPublicationWorkspaceBytes: Int = max(
            maximumSequenceTensorPayloadBytes, maximumBoundaryTensorPayloadBytes);
        // Captured arrays are existing decoder-state ownership. Direct
        // publication adds only the largest one-at-a-time contiguous tensor
        // materialization proven by the writer.
        let directPublicationWorkspaceBytesUInt64: UInt64 = UInt64(
            max(directPublicationWorkspaceBytes, 0));
        if directPublicationWorkspaceBytesUInt64 > effectiveMlxMemoryCeilingBytes {
            throw PersistentPromptCacheModelContractError.captureExceedsMlxMemoryCeiling(
                captureMemoryBytes: directPublicationWorkspaceBytesUInt64,
                effectiveMlxMemoryCeilingBytes: effectiveMlxMemoryCeilingBytes);
        }
        if maximumCommittedBlockBytes > globalSsdQuotaBytes {
            throw PersistentPromptCacheModelContractError.blockFilesExceedSsdQuota(
                blockFileBytes: maximumCommittedBlockBytes,
                globalSsdQuotaBytes: globalSsdQuotaBytes);
        }
        // The fingerprint is compatibility identity, not merely model
        // identity. Any layout, dtype, block size, retained-checkpoint
        // stride, format, model, or revision change must prevent old files
        // from being joined to a chain whose capture and retention topology
        // differs.
        let storageContractFingerprint: Data = PersistentPromptCacheModelContract
            .storageContractFingerprint(
                modelId: modelId,
                modelRevision: modelRevision,
                decoderCacheLayout: decoderCacheLayout,
                blockTokenCount: blockTokenCount,
                commonPrefixCheckpointStrideBlocks: commonPrefixCheckpointStrideBlocks);

        return PersistentPromptCacheModelContract(
            modelIdValue: modelId,
            modelRevisionValue: modelRevision,
            decoderCacheLayoutValue: decoderCacheLayout,
            maximumContextTokenCountValue: maximumContextTokenCount,
            effectiveMlxMemoryCeilingBytesValue: effectiveMlxMemoryCeilingBytes,
            blockTokenCountValue: blockTokenCount,
            commonPrefixCheckpointStrideBlocksValue: commonPrefixCheckpointStrideBlocks,
            sequenceStatePayloadBytesPerTokenValue: sequenceStatePayloadBytesPerToken,
            sequenceStatePayloadBytesPerBlockValue: sequenceStatePayloadBytesPerBlock,
            boundaryStatePayloadBytesValue: boundaryStatePayloadBytes,
            capturePayloadBytesValue: capturePayloadBytes,
            sequenceStateFileBytesValue: sequenceStateFileBytes,
            boundaryStateFileBytesValue: boundaryStateFileBytes,
            maximumBlockManifestFileBytesValue: maximumBlockManifestFileBytes,
            maximumCommittedBlockBytesValue: maximumCommittedBlockBytes,
            directPublicationWorkspaceBytesValue: directPublicationWorkspaceBytes,
            storageContractFingerprintData: storageContractFingerprint);
    }

    public var modelId: String {
        return self.modelIdValue;
    }

    public var modelRevision: String {
        return self.modelRevisionValue;
    }

    public var decoderCacheLayout: DecoderCacheLayout {
        return self.decoderCacheLayoutValue;
    }

    public var maximumContextTokenCount: Int {
        return self.maximumContextTokenCountValue;
    }

    public var blockTokenCount: Int {
        return self.blockTokenCountValue;
    }

    public var commonPrefixCheckpointStrideBlocks: UInt32 {
        return self.commonPrefixCheckpointStrideBlocksValue;
    }

    public var hasSequenceState: Bool {
        return self.decoderCacheLayoutValue.hasSequenceState;
    }

    public var hasBoundaryState: Bool {
        return self.decoderCacheLayoutValue.hasBoundaryState;
    }

    public var sequenceStatePayloadBytesPerToken: Int {
        return self.sequenceStatePayloadBytesPerTokenValue;
    }

    public var sequenceStatePayloadBytesPerBlock: Int {
        return self.sequenceStatePayloadBytesPerBlockValue;
    }

    public var boundaryStatePayloadBytes: Int {
        return self.boundaryStatePayloadBytesValue;
    }

    public var capturePayloadBytes: Int {
        return self.capturePayloadBytesValue;
    }

    public var sequenceStateFileBytes: UInt64 {
        return self.sequenceStateFileBytesValue;
    }

    /// Exact on-disk size of a sequence-state file holding `blockTokenCount`
    /// tokens. Full blocks reuse the size cached at construction; partial
    /// prefix-cache tails are computed on demand from the same layout
    /// contract.
    public func sequenceStateFileBytesForBlockTokenCount(
        blockTokenCount: Int
    ) throws -> UInt64 {
        if blockTokenCount == self.blockTokenCountValue {
            return self.sequenceStateFileBytesValue;
        }
        do {
            return try PersistentPromptCacheStorageGeometry.exactStateFileBytes(
                blockTokenCount: blockTokenCount,
                persistedTensorLayouts: self.decoderCacheLayoutValue.sequenceTensorLayouts());
        } catch let layoutError as DecoderCacheLayoutError {
            throw PersistentPromptCacheModelContractError.decoderCacheLayout(layoutError);
        }
    }

    /// Exact on-disk size of a boundary-state file recorded for a block
    /// holding `blockTokenCount` tokens. Boundary tensors carry no sequence
    /// axis, but the header metadata embeds the token count, so the byte
    /// size still depends on it through the serialized header.
    public func boundaryStateFileBytesForBlockTokenCount(
        blockTokenCount: Int
    ) throws -> UInt64 {
        if blockTokenCount == self.blockTokenCountValue {
            return self.boundaryStateFileBytesValue;
        }
        do {
            return try PersistentPromptCacheStorageGeometry.exactStateFileBytes(
                blockTokenCount: blockTokenCount,
                persistedTensorLayouts: self.decoderCacheLayoutValue.boundaryTensorLayouts());
        } catch let layoutError as DecoderCacheLayoutError {
            throw PersistentPromptCacheModelContractError.decoderCacheLayout(layoutError);
        }
    }

    public var boundaryStateFileBytes: UInt64 {
        return self.boundaryStateFileBytesValue;
    }

    public var maximumBlockManifestFileBytes: UInt64 {
        return self.maximumBlockManifestFileBytesValue;
    }

    public var maximumCommittedBlockBytes: UInt64 {
        return self.maximumCommittedBlockBytesValue;
    }

    public var directPublicationWorkspaceBytes: Int {
        return self.directPublicationWorkspaceBytesValue;
    }

    public var effectiveMlxMemoryCeilingBytes: UInt64 {
        return self.effectiveMlxMemoryCeilingBytesValue;
    }

    /// The 32-byte storage-contract fingerprint bound into every block key.
    public func storageContractFingerprint() -> Data {
        return self.storageContractFingerprintData;
    }

    /// The hexadecimal encoding stamped into on-disk headers and manifests.
    public func storageContractFingerprintHex() -> String {
        return self.storageContractFingerprintData.map({ (fingerprintByte: UInt8) -> String in
            return String(format: "%02x", fingerprintByte);
        }).joined();
    }

    /// Recurrent state is copied at every restorable boundary. With
    /// append-only state, choose a block large enough to amortize that fixed
    /// capture; for recurrent-only models, distribute snapshots across the
    /// context according to the available global SSD quota instead.
    private static func deriveBlockTokenCount(
        maximumContextTokenCount: Int,
        persistenceAlignmentTokenCount: Int,
        sequenceStatePayloadBytesPerToken: Int,
        boundaryStatePayloadBytes: Int,
        globalSsdQuotaBytes: UInt64
    ) throws -> Int {
        var unalignedBlockTokenCount: Int;
        if sequenceStatePayloadBytesPerToken > 0 {
            if boundaryStatePayloadBytes == 0 {
                unalignedBlockTokenCount = persistenceAlignmentTokenCount;
            } else {
                unalignedBlockTokenCount = try PersistentPromptCacheModelContract
                    .checkedCeilingDivision(
                        dividend: boundaryStatePayloadBytes,
                        divisor: sequenceStatePayloadBytesPerToken);
            }
        } else {
            if boundaryStatePayloadBytes == 0 {
                throw PersistentPromptCacheModelContractError.noPersistentState;
            }
            let maximumBoundarySnapshotCount: UInt64 = boundaryStatePayloadBytes == 0
                ? 0
                : globalSsdQuotaBytes / UInt64(boundaryStatePayloadBytes);
            if maximumBoundarySnapshotCount == 0 {
                throw PersistentPromptCacheModelContractError.boundarySnapshotExceedsSsdQuota(
                    boundarySnapshotBytes: UInt64(boundaryStatePayloadBytes),
                    globalSsdQuotaBytes: globalSsdQuotaBytes);
            }
            unalignedBlockTokenCount = try PersistentPromptCacheModelContract
                .checkedCeilingDivision(
                    dividend: maximumContextTokenCount,
                    divisor: Int(maximumBoundarySnapshotCount));
        }
        // Alignment is applied after the quota calculation so every full
        // block can be restored into the tensor capacities described by the
        // layout. Context remains a hard upper bound.
        let alignedBlockTokenCount: Int = try PersistentPromptCacheModelContract
            .checkedAlignUp(
                tokenCount: max(unalignedBlockTokenCount, 1),
                alignmentTokenCount: persistenceAlignmentTokenCount);
        return min(alignedBlockTokenCount, maximumContextTokenCount);
    }

    private static func checkedCeilingDivision(
        dividend: Int, divisor: Int
    ) throws -> Int {
        if divisor == 0 {
            throw PersistentPromptCacheModelContractError.zeroStorageGeometryDivisor;
        }
        let quotient: Int = dividend / divisor;
        let (roundedQuotient, overflow) = quotient
            .addingReportingOverflow(dividend % divisor == 0 ? 0 : 1);
        if overflow {
            throw PersistentPromptCacheModelContractError.blockTokenCountOverflow;
        }
        return roundedQuotient;
    }

    private static func checkedAlignUp(
        tokenCount: Int, alignmentTokenCount: Int
    ) throws -> Int {
        let alignmentRemainder: Int = tokenCount % alignmentTokenCount;
        if alignmentRemainder == 0 {
            return tokenCount;
        }
        let (alignedTokenCount, overflow) = tokenCount
            .addingReportingOverflow(alignmentTokenCount - alignmentRemainder);
        if overflow {
            throw PersistentPromptCacheModelContractError.blockTokenCountOverflow;
        }
        return alignedTokenCount;
    }

    /// Every variable-length field is length-prefixed to prevent
    /// concatenation ambiguity (`ab` + `c` versus `a` + `bc`). Numeric
    /// fields use fixed-width big-endian bytes so the digest is independent
    /// of host architecture.
    private static func storageContractFingerprint(
        modelId: String,
        modelRevision: String,
        decoderCacheLayout: DecoderCacheLayout,
        blockTokenCount: Int,
        commonPrefixCheckpointStrideBlocks: UInt32
    ) -> Data {
        var hasher: SHA256 = SHA256();
        PersistentPromptCacheModelContract.updateLengthPrefixedBytes(
            hasher: &hasher,
            bytes: Data(PersistentPromptCacheBlockHeader.FORMAT_VERSION.utf8));
        PersistentPromptCacheModelContract.updateLengthPrefixedBytes(
            hasher: &hasher, bytes: Data(modelId.utf8));
        PersistentPromptCacheModelContract.updateLengthPrefixedBytes(
            hasher: &hasher, bytes: Data(modelRevision.utf8));
        var bigEndianBlockTokenCount: UInt64 = UInt64(blockTokenCount).bigEndian;
        withUnsafeBytes(of: &bigEndianBlockTokenCount) { hasher.update(data: $0); };
        var bigEndianStrideBlocks: UInt32 = commonPrefixCheckpointStrideBlocks.bigEndian;
        withUnsafeBytes(of: &bigEndianStrideBlocks) { hasher.update(data: $0); };
        PersistentPromptCacheModelContract.updateTensorLayoutFingerprint(
            hasher: &hasher,
            stateKindName: "sequence",
            persistedTensorLayouts: decoderCacheLayout.sequenceTensorLayouts());
        PersistentPromptCacheModelContract.updateTensorLayoutFingerprint(
            hasher: &hasher,
            stateKindName: "boundary",
            persistedTensorLayouts: decoderCacheLayout.boundaryTensorLayouts());
        return Data(hasher.finalize());
    }

    private static func updateTensorLayoutFingerprint(
        hasher: inout SHA256,
        stateKindName: String,
        persistedTensorLayouts: [DecoderCachePersistedTensorLayout]
    ) {
        PersistentPromptCacheModelContract.updateLengthPrefixedBytes(
            hasher: &hasher, bytes: Data(stateKindName.utf8));
        var bigEndianLayoutCount: UInt64 = UInt64(persistedTensorLayouts.count).bigEndian;
        withUnsafeBytes(of: &bigEndianLayoutCount) { hasher.update(data: $0); };
        for persistedTensorLayout: DecoderCachePersistedTensorLayout in persistedTensorLayouts {
            var bigEndianLayerIndex: UInt64 = UInt64(persistedTensorLayout.decoderLayerIndex)
                .bigEndian;
            withUnsafeBytes(of: &bigEndianLayerIndex) { hasher.update(data: $0); };
            let tensorLayout: DecoderCacheTensorLayout = persistedTensorLayout.tensorLayout;
            PersistentPromptCacheModelContract.updateLengthPrefixedBytes(
                hasher: &hasher, bytes: Data(tensorLayout.tensorRoleName.utf8));
            PersistentPromptCacheModelContract.updateLengthPrefixedBytes(
                hasher: &hasher, bytes: Data(tensorLayout.dtype.safetensorsDtypeName.utf8));
            var bigEndianSequenceAxis: UInt64 = UInt64(
                tensorLayout.sequenceAxis.map({ (sequenceAxis: Int) -> UInt64 in
                    return UInt64(sequenceAxis);
                }) ?? UInt64(Int.max)).bigEndian;
            withUnsafeBytes(of: &bigEndianSequenceAxis) { hasher.update(data: $0); };
            var bigEndianDimensionCount: UInt64 = UInt64(tensorLayout.dimensions.count).bigEndian;
            withUnsafeBytes(of: &bigEndianDimensionCount) { hasher.update(data: $0); };
            for tensorDimension: Int in tensorLayout.dimensions {
                var bigEndianDimension: UInt64 = UInt64(tensorDimension).bigEndian;
                withUnsafeBytes(of: &bigEndianDimension) { hasher.update(data: $0); };
            }
        }
    }

    private static func updateLengthPrefixedBytes(
        hasher: inout SHA256, bytes: Data
    ) {
        var bigEndianByteCount: UInt64 = UInt64(bytes.count).bigEndian;
        withUnsafeBytes(of: &bigEndianByteCount) { hasher.update(data: $0); };
        hasher.update(data: bytes);
    }
}
