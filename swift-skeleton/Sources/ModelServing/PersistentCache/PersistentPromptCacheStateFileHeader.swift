import Foundation;


/// The deterministic safetensors state-file layout shared by the exact-size
/// geometry projection and the direct MLX writer. Quota admission and
/// publication both build their header bytes from this one implementation,
/// so a written state file can never disagree with the byte projection that
/// admitted it, and the projection can never drift from the native writer's
/// largest-tensor-first materialization order.
enum PersistentPromptCacheStateFileHeader {

    /// One tensor's projected header entry: the on-disk name, safetensors
    /// dtype label, materialized dimensions, and exact payload byte count.
    struct TensorEntry: Equatable {

        var tensorName: String;
        var dtypeName: String;
        var dimensions: [Int];
        var payloadByteCount: Int;
    }

    /// Projects the contract tensor layouts at `blockTokenCount` and orders
    /// them largest-payload-first with a name tie-break. The writer
    /// materializes tensors in this order to bound peak workspace, so
    /// payload offsets must be assigned in the same sequence for the
    /// exact-size geometry to stay predictive.
    static func tensorEntries(
        persistedTensorLayouts: [DecoderCachePersistedTensorLayout],
        blockTokenCount: Int
    ) throws -> [TensorEntry] {
        if persistedTensorLayouts.isEmpty {
            return [];
        }
        var tensorEntries: [TensorEntry] = [];
        for persistedTensorLayout: DecoderCachePersistedTensorLayout in persistedTensorLayouts {
            let tensorLayout: DecoderCacheTensorLayout = persistedTensorLayout.tensorLayout;
            let dimensions: [Int] = tensorLayout.dimensions.enumerated().map(
                { (dimensionEntry: (offset: Int, element: Int)) -> Int in
                    if tensorLayout.sequenceAxis == dimensionEntry.offset {
                        return blockTokenCount;
                    }
                    return dimensionEntry.element;
                });
            let payloadByteCount: Int;
            do {
                if tensorLayout.sequenceAxis != nil {
                    payloadByteCount = try tensorLayout.sequencePayloadByteCountPerToken()
                        * blockTokenCount;
                } else {
                    payloadByteCount = try tensorLayout.fixedPayloadByteCount();
                }
            } catch let layoutError as DecoderCacheLayoutError {
                throw PersistentPromptCacheModelContractError.decoderCacheLayout(layoutError);
            }
            tensorEntries.append(TensorEntry(
                tensorName: persistedTensorLayout.persistentTensorName,
                dtypeName: tensorLayout.dtype.safetensorsDtypeName,
                dimensions: dimensions,
                payloadByteCount: payloadByteCount));
        }
        tensorEntries.sort(by: { (leftEntry: TensorEntry, rightEntry: TensorEntry) -> Bool in
            if leftEntry.payloadByteCount != rightEntry.payloadByteCount {
                return leftEntry.payloadByteCount > rightEntry.payloadByteCount;
            }
            return leftEntry.tensorName < rightEntry.tensorName;
        });
        return tensorEntries;
    }

    /// Builds the complete header JSON exactly as it must appear on disk:
    /// keys serialized in sorted byte order (the Rust serde_json BTreeMap
    /// projection, where "__metadata__" sorts before the "layer_" tensor
    /// names), tensor fields in data_offsets/dtype/shape order, and every
    /// metadata value as a JSON string so readers can distinguish a token
    /// count from a bare number.
    static func headerJson(
        tensorEntries: [TensorEntry],
        blockTokenCount: Int,
        storageContractFingerprint: String,
        formatVersion: String
    ) throws -> String {
        let metadataJson: String = "{"
            + "\"block_token_count\":\"\(blockTokenCount)\","
            + "\"format_version\":\"\(formatVersion)\","
            + "\"storage_contract_fingerprint\":\"\(storageContractFingerprint)\""
            + "}";
        var headerEntries: Array<(entryKey: String, entryJson: String)> =
            [("__metadata__", "\"__metadata__\":\(metadataJson)")];
        var payloadOffsetBytes: UInt64 = 0;
        for tensorEntry: TensorEntry in tensorEntries {
            let tensorPayloadBytes: UInt64 = UInt64(max(tensorEntry.payloadByteCount, 0));
            let (payloadEndBytes, endOverflow) = payloadOffsetBytes
                .addingReportingOverflow(tensorPayloadBytes);
            if endOverflow {
                throw PersistentPromptCacheModelContractError.capturePayloadByteCountOverflow;
            }
            let dimensionsJson: String = tensorEntry.dimensions
                .map(String.init)
                .joined(separator: ",");
            headerEntries.append((
                tensorEntry.tensorName,
                "\"\(tensorEntry.tensorName)\":{"
                    + "\"data_offsets\":[\(payloadOffsetBytes),\(payloadEndBytes)],"
                    + "\"dtype\":\"\(tensorEntry.dtypeName)\","
                    + "\"shape\":[\(dimensionsJson)]}"));
            payloadOffsetBytes = payloadEndBytes;
        }
        headerEntries.sort(by: { (leftEntry: (entryKey: String, entryJson: String),
                                   rightEntry: (entryKey: String, entryJson: String)) -> Bool in
            return leftEntry.entryKey.utf8.lexicographicallyPrecedes(rightEntry.entryKey.utf8);
        });
        return "{"
            + headerEntries.map({ (headerEntry: (entryKey: String, entryJson: String)) -> String in
                return headerEntry.entryJson;
            })
            .joined(separator: ",") + "}";
    }

    /// The complete on-disk byte count: the 8-byte little-endian header
    /// length prefix, the header JSON, and every tensor payload.
    static func totalFileBytes(
        headerJsonByteCount: Int,
        totalPayloadByteCount: UInt64
    ) throws -> UInt64 {
        let headerByteCount: UInt64 = UInt64(headerJsonByteCount);
        let (headerSectionBytes, sectionOverflow) = SafetensorsFraming
            .SAFETENSORS_HEADER_LENGTH_PREFIX_BYTES
            .addingReportingOverflow(headerByteCount);
        if sectionOverflow {
            throw PersistentPromptCacheModelContractError.capturePayloadByteCountOverflow;
        }
        let (totalBytes, totalOverflow) = headerSectionBytes
            .addingReportingOverflow(totalPayloadByteCount);
        if totalOverflow {
            throw PersistentPromptCacheModelContractError.capturePayloadByteCountOverflow;
        }
        return totalBytes;
    }
}
