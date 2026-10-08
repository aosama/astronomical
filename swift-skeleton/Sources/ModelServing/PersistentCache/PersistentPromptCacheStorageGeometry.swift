import Foundation;


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
        let tensorEntries: [PersistentPromptCacheStateFileHeader.TensorEntry] = try
            PersistentPromptCacheStateFileHeader.tensorEntries(
                persistedTensorLayouts: persistedTensorLayouts,
                blockTokenCount: blockTokenCount);
        if tensorEntries.isEmpty {
            return 0;
        }
        let headerJson: String = try PersistentPromptCacheStateFileHeader.headerJson(
            tensorEntries: tensorEntries,
            blockTokenCount: blockTokenCount,
            storageContractFingerprint: PersistentPromptCacheStorageGeometry
                .FINGERPRINT_PLACEHOLDER,
            formatVersion: PersistentPromptCacheBlockHeader.FORMAT_VERSION);
        var totalPayloadByteCount: UInt64 = 0;
        for tensorEntry: PersistentPromptCacheStateFileHeader.TensorEntry in tensorEntries {
            let (payloadTotalBytes, payloadOverflow) = totalPayloadByteCount
                .addingReportingOverflow(UInt64(max(tensorEntry.payloadByteCount, 0)));
            if payloadOverflow {
                throw PersistentPromptCacheModelContractError.capturePayloadByteCountOverflow;
            }
            totalPayloadByteCount = payloadTotalBytes;
        }
        return try PersistentPromptCacheStateFileHeader.totalFileBytes(
            headerJsonByteCount: headerJson.utf8.count,
            totalPayloadByteCount: totalPayloadByteCount);
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
