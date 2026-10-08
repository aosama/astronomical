import Foundation

/// The bounded-reader-facing view of one shard manifest: the synthetic
/// safetensors header. Port of the Rust
/// `QuantizedExpertShardManifest::rebased_safetensors_header`.
extension QuantizedExpertShardManifest {

    /**
     * Serializes the manifest's tensor placements with offsets relative to
     * this shard's virtual payload: the 8-byte little-endian length prefix
     * plus the JSON tensor map the stock safetensors decoder expects.
     *
     * - Returns: The complete synthetic header bytes.
     */
    public func rebasedSafetensorsHeader() throws -> Data {
        var headerEntriesByTensorName: [String: [String: Any]] = Dictionary(
            minimumCapacity: self.tensorRanges.count)
        for tensorRange: QuantizedExpertTensorRange in self.tensorRanges {
            let tensorStartBytes: UInt64 = tensorRange.virtualPayloadOffsetBytes
            let tensorEndBytes: UInt64 = tensorStartBytes + UInt64(tensorRange.byteCount)
            headerEntriesByTensorName[tensorRange.tensorName] = [
                "dtype": tensorRange.dtype.canonicalName,
                "shape": tensorRange.shape,
                "data_offsets": [tensorStartBytes, tensorEndBytes],
            ]
        }
        // Sorted keys keep the header byte-stable across runs; the stock
        // decoder accepts any key order, and determinism keeps bounded-read
        // metrics comparable.
        guard JSONSerialization.isValidJSONObject(headerEntriesByTensorName) else {
            throw ExpertPagingError.manifestValidationFailure(
                description: "the rebased safetensors header is not serializable JSON")
        }
        let encodedHeader: Data
        do {
            encodedHeader = try JSONSerialization.data(
                withJSONObject: headerEntriesByTensorName,
                options: [.sortedKeys])
        } catch {
            throw ExpertPagingError.manifestValidationFailure(
                description: "the rebased safetensors header could not be serialized")
        }
        var headerBytes: Data = Data(capacity: 8 + encodedHeader.count)
        var littleEndianHeaderLength: UInt64 = UInt64(encodedHeader.count).littleEndian
        withUnsafeBytes(of: &littleEndianHeaderLength) { (lengthBuffer: UnsafeRawBufferPointer) -> Void in
            headerBytes.append(contentsOf: lengthBuffer)
        }
        headerBytes.append(encodedHeader)
        return headerBytes
    }
}
