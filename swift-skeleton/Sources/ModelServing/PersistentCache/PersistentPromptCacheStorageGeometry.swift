import Foundation;

import ModelServing;

/// Exact on-disk size projection for model-bound prompt-cache artifacts,
/// port of the Rust `model_contract_storage_geometry`. Quota admission
/// happens before publication, so estimates cannot be vague: these formulas
/// reproduce the safetensors header and payload layout using the same
/// deterministic largest-tensor-first order as the native writer.
enum PersistentPromptCacheStorageGeometry {

    /// The metadata content the exact-size projection models. Fingerprint
    /// contents do not affect length: every SHA-256 hexadecimal encoding
    /// occupies exactly 64 bytes, so zeroes model the final header.
    private static let FINGERPRINT_PLACEHOLDER: String = String(repeating: "0", count: 64);

    static func exactStateFileBytes(
        blockTokenCount: Int,
        persistedTensorLayouts: [DecoderCachePersistedTensorLayout]
    ) throws -> UInt64 {
        if persistedTensorLayouts.isEmpty {
            return 0;
        }
        // The writer materializes the largest tensor first to bound peak
        // workspace. Sorting here the same way also makes JSON insertion and
        // payload offsets deterministic, which is required for exact byte
        // prediction.
        var tensorGeometry: [(tensorName: String, dtypeName: String, dimensions: [Int], payloadBytes: Int)] = [];
        for persistedTensorLayout: DecoderCachePersistedTensorLayout in persistedTensorLayouts {
            let tensorLayout: DecoderCacheTensorLayout = persistedTensorLayout.tensorLayout;
            let dimensions: [Int] = tensorLayout.dimensions.enumerated().map(
                { (dimensionEntry: (offset: Int, element: Int)) -> Int in
                    if tensorLayout.sequenceAxis == dimensionEntry.offset {
                        return blockTokenCount;
                    }
                    return dimensionEntry.element;
                });
            let payloadBytes: Int;
            do {
                if tensorLayout.sequenceAxis != nil {
                    payloadBytes = try tensorLayout.sequencePayloadByteCountPerToken()
                        * blockTokenCount;
                } else {
                    payloadBytes = try tensorLayout.fixedPayloadByteCount();
                }
            } catch let layoutError as DecoderCacheLayoutError {
                throw PersistentPromptCacheModelContractError.decoderCacheLayout(layoutError);
            }
            tensorGeometry.append((
                persistedTensorLayout.persistentTensorName,
                tensorLayout.dtype.safetensorsDtypeName,
                dimensions,
                payloadBytes));
        }
        tensorGeometry.sort(by: { (leftTensor, rightTensor) -> Bool in
            if leftTensor.payloadBytes != rightTensor.payloadBytes {
                return leftTensor.payloadBytes > rightTensor.payloadBytes;
            }
            return leftTensor.tensorName < rightTensor.tensorName;
        });
        let metadataJson: String = "{"
            + "\"block_token_count\":\"\(blockTokenCount)\","
            + "\"format_version\":\"\(PersistentPromptCacheBlockHeader.FORMAT_VERSION)\","
            + "\"storage_contract_fingerprint\":\"\(PersistentPromptCacheStorageGeometry.FINGERPRINT_PLACEHOLDER)\""
            + "}";
        var headerEntries: Array<(entryKey: String, entryJson: String)> =
            [("__metadata__", "\"__metadata__\":\(metadataJson)")];
        var payloadOffsetBytes: UInt64 = 0;
        for tensorEntry in tensorGeometry {
            let tensorPayloadBytes: UInt64 = UInt64(max(tensorEntry.payloadBytes, 0));
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
        // Insertion order matches the Rust serde_json BTreeMap projection:
        // the JSON object is serialized in sorted key order ("__metadata__"
        // sorts before the "layer_" tensor names), so the serialized header
        // byte count matches the Rust formula exactly.
        headerEntries.sort(by: { (leftEntry, rightEntry) -> Bool in
            return leftEntry.entryKey.utf8.lexicographicallyPrecedes(rightEntry.entryKey.utf8);
        });
        let headerJson: String = "{"
            + headerEntries.map({ (headerEntry: (entryKey: String, entryJson: String)) -> String in
                return headerEntry.entryJson;
            })
            .joined(separator: ",") + "}";
        let headerByteCount: UInt64 = UInt64(headerJson.utf8.count);
        let (headerSectionBytes, sectionOverflow) = SafetensorsFraming
            .SAFETENSORS_HEADER_LENGTH_PREFIX_BYTES
            .addingReportingOverflow(headerByteCount);
        if sectionOverflow {
            throw PersistentPromptCacheModelContractError.capturePayloadByteCountOverflow;
        }
        let (totalBytes, totalOverflow) = headerSectionBytes
            .addingReportingOverflow(payloadOffsetBytes);
        if totalOverflow {
            throw PersistentPromptCacheModelContractError.capturePayloadByteCountOverflow;
        }
        return totalBytes;
    }

    static func maximumBlockManifestFileBytes(
        maximumContextTokenCount: Int
    ) throws -> UInt64 {
        // Use the widest possible values and all optional fields so the
        // transaction can reject an unexpectedly larger manifest before
        // consuming global quota.
        let maximumBlockIndex: UInt64 = UInt64(max(maximumContextTokenCount, 0));
        let manifestJson: String = "{"
            + "\"block_hash\":\"\(PersistentPromptCacheStorageGeometry.FINGERPRINT_PLACEHOLDER)\","
            + "\"block_index\":\(maximumBlockIndex),"
            + "\"format_version\":\"\(PersistentPromptCacheBlockHeader.FORMAT_VERSION)\","
            + "\"has_boundary_state\":true,"
            + "\"has_sequence_state\":true,"
            + "\"parent_block_hash\":\"\(PersistentPromptCacheStorageGeometry.FINGERPRINT_PLACEHOLDER)\","
            + "\"storage_contract_fingerprint\":\"\(PersistentPromptCacheStorageGeometry.FINGERPRINT_PLACEHOLDER)\""
            + "}";
        return UInt64(manifestJson.utf8.count);
    }
}
