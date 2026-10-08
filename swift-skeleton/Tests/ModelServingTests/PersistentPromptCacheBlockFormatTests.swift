import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Hermetic journeys for the bounded persisted-state header validator:
/// contract-generated sequence and boundary headers validate, a wrong
/// execution dtype, a missing tensor, a foreign storage-contract
/// fingerprint, and a previous disposable format version all fail closed
/// before any payload use.
final class PersistentPromptCacheBlockFormatTests {

    @Test
    func should_accept_a_contract_generated_sequence_state_header() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let sequenceStateFileUrl: URL = try PersistentPromptCacheBlockFormatTests
            .writeContractGeneratedFile(
                directoryName: UUID().uuidString,
                fileName: "sequence.safetensors",
                modelContract: modelContract,
                shouldWriteBoundaryState: false);

        let sequenceStateHeader: PersistentPromptCacheBlockHeader = try #require(
            try? PersistentPromptCacheBlockHeader.readKvBlock(
                blockFileUrl: sequenceStateFileUrl, modelContract: modelContract),
            "the contract-generated sequence state should validate");

        #expect(sequenceStateHeader.blockTokenCount == modelContract.blockTokenCount);
        #expect(sequenceStateHeader.tensorCount
            == modelContract.decoderCacheLayout.sequenceTensorCount);
        #expect(sequenceStateHeader.storageContractFingerprint
            == modelContract.storageContractFingerprintHex());
    }

    @Test
    func should_accept_a_contract_generated_boundary_state_header() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let boundaryStateFileUrl: URL = try PersistentPromptCacheBlockFormatTests
            .writeContractGeneratedFile(
                directoryName: UUID().uuidString,
                fileName: "boundary.safetensors",
                modelContract: modelContract,
                shouldWriteBoundaryState: true);

        let boundaryStateHeader: PersistentPromptCacheBlockHeader = try #require(
            try? PersistentPromptCacheBlockHeader.readRecurrentSnapshot(
                snapshotFileUrl: boundaryStateFileUrl, modelContract: modelContract),
            "the contract-generated boundary state should validate");

        #expect(boundaryStateHeader.tensorCount
            == modelContract.decoderCacheLayout.boundaryTensorCount);
    }

    @Test
    func should_reject_a_declared_dtype_that_differs_from_the_model_contract() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let firstSequenceTensorName: String = try #require(
            modelContract.decoderCacheLayout.sequenceTensorLayouts().first,
            "the frozen model should have sequence state").persistentTensorName;
        let sequenceStateFileUrl: URL = try PersistentPromptCacheBlockFormatTests
            .writeContractGeneratedFile(
                directoryName: UUID().uuidString,
                fileName: "wrong-dtype.safetensors",
                modelContract: modelContract,
                shouldWriteBoundaryState: false,
                wrongDtypeTensorName: firstSequenceTensorName);

        #expect(throws: (any Error).self) {
            _ = try PersistentPromptCacheBlockHeader.readKvBlock(
                blockFileUrl: sequenceStateFileUrl, modelContract: modelContract);
        };
    }

    @Test
    func should_reject_a_missing_declared_tensor() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let firstSequenceTensorName: String = try #require(
            modelContract.decoderCacheLayout.sequenceTensorLayouts().first,
            "the frozen model should have sequence state").persistentTensorName;
        let sequenceStateFileUrl: URL = try PersistentPromptCacheBlockFormatTests
            .writeContractGeneratedFile(
                directoryName: UUID().uuidString,
                fileName: "missing-tensor.safetensors",
                modelContract: modelContract,
                shouldWriteBoundaryState: false,
                omittedTensorName: firstSequenceTensorName);

        #expect(throws: (any Error).self) {
            _ = try PersistentPromptCacheBlockHeader.readKvBlock(
                blockFileUrl: sequenceStateFileUrl, modelContract: modelContract);
        };
    }

    @Test
    func should_reject_a_foreign_storage_contract_fingerprint_before_payload_use() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let sequenceStateFileUrl: URL = try PersistentPromptCacheBlockFormatTests
            .writeContractGeneratedFile(
                directoryName: UUID().uuidString,
                fileName: "foreign-contract.safetensors",
                modelContract: modelContract,
                shouldWriteBoundaryState: false);
        try PersistentPromptCacheBlockFormatTests.replaceMetadataValue(
            fileUrl: sequenceStateFileUrl,
            metadataName: "storage_contract_fingerprint",
            metadataValue: String(repeating: "00", count: 32));

        do {
            _ = try PersistentPromptCacheBlockHeader.readKvBlock(
                blockFileUrl: sequenceStateFileUrl, modelContract: modelContract);
            Issue.record("a foreign storage-contract fingerprint must be rejected");
        } catch {
            // The rejection is the journey's success.
        }
    }

    @Test
    func should_reject_a_previous_disposable_format_version() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let sequenceStateFileUrl: URL = try PersistentPromptCacheBlockFormatTests
            .writeContractGeneratedFile(
                directoryName: UUID().uuidString,
                fileName: "old-format.safetensors",
                modelContract: modelContract,
                shouldWriteBoundaryState: false);
        try PersistentPromptCacheBlockFormatTests.replaceMetadataValue(
            fileUrl: sequenceStateFileUrl,
            metadataName: "format_version",
            metadataValue: "11");

        do {
            _ = try PersistentPromptCacheBlockHeader.readKvBlock(
                blockFileUrl: sequenceStateFileUrl, modelContract: modelContract);
            Issue.record("the old disposable format must be rejected");
        } catch let rejection as PersistentPromptCacheBlockError {
            guard case let .unsupportedFormatVersion(actualFormatVersion, expectedFormatVersion) =
                rejection
            else {
                Issue.record("expected the format-version rejection, got \(rejection)");
                return;
            }
            #expect(actualFormatVersion == "11");
            #expect(expectedFormatVersion == "12");
        } catch {
            Issue.record("expected a typed block rejection, got \(error)");
        }
    }

    /// Writes one contract-shaped synthetic safetensors state file whose
    /// header mirrors the storage geometry; optional overrides corrupt the
    /// dtype or omit a tensor for the rejection journeys.
    private static func writeContractGeneratedFile(
        directoryName: String,
        fileName: String,
        modelContract: PersistentPromptCacheModelContract,
        shouldWriteBoundaryState: Bool,
        wrongDtypeTensorName: String? = nil,
        omittedTensorName: String? = nil
    ) throws -> URL {
        let persistedTensorLayouts: [DecoderCachePersistedTensorLayout] = shouldWriteBoundaryState
            ? modelContract.decoderCacheLayout.boundaryTensorLayouts()
            : modelContract.decoderCacheLayout.sequenceTensorLayouts();
        var headerObject: [String: Any] = [:];
        var payloadOffsetBytes: UInt64 = 0;
        for persistedTensorLayout: DecoderCachePersistedTensorLayout in persistedTensorLayouts {
            let tensorName: String = persistedTensorLayout.persistentTensorName;
            if omittedTensorName == tensorName {
                continue;
            }
            let tensorLayout: DecoderCacheTensorLayout = persistedTensorLayout.tensorLayout;
            let tensorShape: [Int] = tensorLayout.dimensions.enumerated().map(
                { (dimensionEntry: (offset: Int, element: Int)) -> Int in
                    if dimensionEntry.offset == tensorLayout.sequenceAxis {
                        return modelContract.blockTokenCount;
                    }
                    return dimensionEntry.element;
                });
            let tensorElementSizeBytes: UInt64 = UInt64(tensorLayout.dtype.scalarByteCount);
            var tensorPayloadByteCount: UInt64 = tensorElementSizeBytes;
            for tensorDimension: Int in tensorShape {
                tensorPayloadByteCount = tensorPayloadByteCount
                    &* UInt64(max(tensorDimension, 0));
            }
            let payloadEndBytes: UInt64 = payloadOffsetBytes &+ tensorPayloadByteCount;
            // Pick a dtype different from the contract rather than assuming
            // the frozen sequence tensors use one particular precision. The
            // header validator must reject the mismatch before it considers
            // the synthetic payload contents.
            let declaredDtype: String;
            if wrongDtypeTensorName == tensorName {
                declaredDtype = tensorLayout.dtype.safetensorsDtypeName == "BF16"
                    ? "F32" : "BF16";
            } else {
                declaredDtype = tensorLayout.dtype.safetensorsDtypeName;
            }
            headerObject[tensorName] = [
                "dtype": declaredDtype,
                "shape": tensorShape,
                "data_offsets": [payloadOffsetBytes, payloadEndBytes],
            ];
            payloadOffsetBytes = payloadEndBytes;
        }
        headerObject["__metadata__"] = [
            "format_version": "12",
            "block_token_count": String(modelContract.blockTokenCount),
            "storage_contract_fingerprint": modelContract.storageContractFingerprintHex(),
        ];
        let headerData: Data = try JSONSerialization.data(
            withJSONObject: headerObject, options: []);
        let temporaryDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent(directoryName, isDirectory: true);
        try FileManager.default.createDirectory(
            at: temporaryDirectoryUrl, withIntermediateDirectories: true);
        let fileUrl: URL = temporaryDirectoryUrl.appendingPathComponent(fileName);
        var fileBytes: Data = Data();
        var littleEndianHeaderLength: UInt64 = UInt64(headerData.count);
        withUnsafeBytes(of: &littleEndianHeaderLength) { (valueBuffer: UnsafeRawBufferPointer) in
            fileBytes.append(contentsOf: valueBuffer);
        };
        fileBytes.append(headerData);
        fileBytes.append(Data(count: Int(payloadOffsetBytes)));
        try fileBytes.write(to: fileUrl);
        return fileUrl;
    }

    /// Rewrites one `__metadata__` field inside the framed header and
    /// re-frames the file around the modified header bytes.
    private static func replaceMetadataValue(
        fileUrl: URL, metadataName: String, metadataValue: String
    ) throws {
        let fileBytes: Data = try Data(contentsOf: fileUrl);
        let headerLengthBytes: Int = fileBytes.prefix(8).withUnsafeBytes(
            { (valueBuffer: UnsafeRawBufferPointer) -> Int in
                return Int(UInt64(littleEndian: valueBuffer.loadUnaligned(fromByteOffset: 0, as: UInt64.self)));
            });
        let headerObject: [String: Any] = try JSONSerialization.jsonObject(
            with: fileBytes.subdata(in: 8..<(8 + headerLengthBytes)), options: [])
            as! [String: Any];
        var metadataObject: [String: Any] = headerObject["__metadata__"] as! [String: Any];
        metadataObject[metadataName] = metadataValue;
        var replacementObject: [String: Any] = headerObject;
        replacementObject["__metadata__"] = metadataObject;
        let headerData: Data = try JSONSerialization.data(
            withJSONObject: replacementObject, options: []);
        let payloadBytes: Data = fileBytes.subdata(in: (8 + headerLengthBytes)..<fileBytes.count);
        var replacementBytes: Data = Data();
        var littleEndianHeaderLength: UInt64 = UInt64(headerData.count);
        withUnsafeBytes(of: &littleEndianHeaderLength) { (valueBuffer: UnsafeRawBufferPointer) in
            replacementBytes.append(contentsOf: valueBuffer);
        };
        replacementBytes.append(headerData);
        replacementBytes.append(payloadBytes);
        try replacementBytes.write(to: fileUrl);
    }
}
