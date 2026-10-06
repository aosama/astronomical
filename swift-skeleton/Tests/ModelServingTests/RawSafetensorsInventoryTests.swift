import Foundation;
import ModelServing;
import Testing;
import JourneyCategories;

/**
 * Behavioral journeys for the strict family-neutral raw inventory and the
 * bounded partial-profile validator, twin-porting
 * crates/model-serving/tests/hermetic/raw_safetensors_inventory.rs.
 */
@Suite(.tags(.hermeticJourney))
final class RawSafetensorsInventoryTests {

    private static let WEIGHTS_FILE_NAME: String = "model.safetensors";

    @Test
    func should_inventory_the_retained_descriptor_after_the_artifact_path_is_replaced() throws {
        let header: String = "{\"z.tensor\":{\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[0,4]},"
            + "\"__metadata__\":{\"format\":\"mlx\"},"
            + "\"a.tensor\":{\"dtype\":\"I32\",\"shape\":[1],\"data_offsets\":[4,8]}}";
        let framedBytes: Data = Self.safetensorsBytes(headerText: header, payloadBytes: Data([1, 2, 3, 4, 5, 6, 7, 8]));
        let modelDirectory: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: modelDirectory); }
        let weightsUrl: URL = modelDirectory.appendingPathComponent(Self.WEIGHTS_FILE_NAME);
        try framedBytes.write(to: weightsUrl);
        let validatedWeightsFile: ValidatedWeightsFile = try Self.validateWeightsFile(
            modelDirectory: modelDirectory, sizeBytes: UInt64(framedBytes.count));

        try FileManager.default.moveItem(
            at: weightsUrl, to: modelDirectory.appendingPathComponent("retained-model.safetensors"));
        try Data("replacement".utf8).write(to: weightsUrl);

        let inventory: RawSafetensorsInventory = try validatedWeightsFile
            .readRawSafetensorsInventoryForTests();
        let payloadStart: UInt64 = 8 + UInt64(header.utf8.count);

        #expect(inventory.shardPayloadBytes == 8);
        #expect(inventory.tensorDescriptors.count == 2);
        #expect(inventory.tensorDescriptors[0].tensorName == "a.tensor");
        #expect(inventory.tensorDescriptors[0].dtype.canonicalName == "I32");
        #expect(inventory.tensorDescriptors[0].shape == [1]);
        #expect(inventory.tensorDescriptors[0].dataStartOffsetBytes == payloadStart + 4);
        #expect(inventory.tensorDescriptors[0].dataEndOffsetBytes == payloadStart + 8);
        #expect(inventory.tensorDescriptors[1].tensorName == "z.tensor");
        let metadataExcluded: Bool = inventory.tensorDescriptors.allSatisfy({ (descriptor: RawSafetensorsTensorDescriptor) -> Bool in
            return descriptor.tensorName != "__metadata__";
        });
        #expect(metadataExcluded);
    }

    @Test
    func should_accept_every_dtype_supported_by_the_safetensors_format() throws {
        let dtypeCases: Array<(dtypeName: String, payloadLengthBytes: Int)> = [
            ("BOOL", 8), ("F4", 4), ("F6_E2M3", 6), ("F6_E3M2", 6),
            ("U8", 8), ("I8", 8), ("F8_E5M2", 8), ("F8_E4M3", 8), ("F8_E8M0", 8),
            ("F8_E4M3FNUZ", 8), ("F8_E5M2FNUZ", 8),
            ("I16", 16), ("U16", 16), ("F16", 16), ("BF16", 16),
            ("I32", 32), ("U32", 32), ("F32", 32),
            ("C64", 64), ("F64", 64), ("I64", 64), ("U64", 64),
        ];

        for dtypeCase: (dtypeName: String, payloadLengthBytes: Int) in dtypeCases {
            // Eight elements make all sub-byte dtypes byte-aligned.
            let header: String = "{\"tensor\":{\"dtype\":\"\(dtypeCase.dtypeName)\",\"shape\":[8],\"data_offsets\":[0,\(dtypeCase.payloadLengthBytes)]}}";
            let (modelDirectory, validatedWeightsFile): (URL, ValidatedWeightsFile) = try Self
                .validatedWeightsFileForHeader(
                    headerText: header, payloadBytes: Data(repeating: 0, count: dtypeCase.payloadLengthBytes));
            defer { try? FileManager.default.removeItem(at: modelDirectory); }
            let inventory: RawSafetensorsInventory = try validatedWeightsFile
                .readRawSafetensorsInventoryForTests();
            #expect(inventory.tensorDescriptors[0].dtype.canonicalName == dtypeCase.dtypeName);
            #expect(
                inventory.tensorDescriptors[0].tensorPayloadBytes
                    == UInt64(dtypeCase.payloadLengthBytes));
        }
    }

    @Test
    func should_reject_duplicate_tensor_and_nested_object_keys() throws {
        let duplicateCases: Array<String> = [
            "{\"tensor\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},"
                + "\"tensor\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}",
            "{\"tensor\":{\"dtype\":\"U8\",\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,1]}}",
            "{\"__metadata__\":{\"format\":\"mlx\",\"format\":\"pt\"},"
                + "\"tensor\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}",
        ];

        for duplicateHeader: String in duplicateCases {
            let (modelDirectory, validatedWeightsFile): (URL, ValidatedWeightsFile) = try Self
                .validatedWeightsFileForHeader(
                    headerText: duplicateHeader, payloadBytes: Data([0]));
            defer { try? FileManager.default.removeItem(at: modelDirectory); }
            do {
                _ = try validatedWeightsFile.readRawSafetensorsInventoryForTests();
                Issue.record("duplicate object keys must fail before replacement");
            } catch let validationError as ArtifactValidationError {
                guard case .invalidSafetensorsHeader(_, let problem) = validationError else {
                    Issue.record("expected InvalidSafetensorsHeader, got \(validationError)");
                    return;
                }
                #expect(
                    problem.lowercased().contains("duplicate"),
                    "the duplicate-key rejection must name the duplication, got: \(problem)");
            }
        }
    }

    @Test
    func should_reject_invalid_dtypes_shapes_and_intervals() throws {
        let malformedHeaderCases: Array<(header: String, payload: Data, caseName: String)> = [
            ("{\"tensor\":{\"dtype\":\"UNKNOWN\",\"shape\":[1],\"data_offsets\":[0,1]}}",
             Data([0]), "dtype"),
            ("{\"tensor\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}",
             Data(repeating: 0, count: 4), "width"),
            ("{\"tensor\":{\"dtype\":\"F4\",\"shape\":[1],\"data_offsets\":[0,1]}}",
             Data([0]), "alignment"),
            ("{\"tensor\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,0]}}",
             Data([0]), "reversed"),
        ];

        for malformedCase: (header: String, payload: Data, caseName: String) in malformedHeaderCases {
            let (modelDirectory, validatedWeightsFile): (URL, ValidatedWeightsFile) = try Self
                .validatedWeightsFileForHeader(
                    headerText: malformedCase.header, payloadBytes: malformedCase.payload);
            defer { try? FileManager.default.removeItem(at: modelDirectory); }
            do {
                _ = try validatedWeightsFile.readRawSafetensorsInventoryForTests();
                Issue.record("\(malformedCase.caseName) must fail closed");
            } catch is ArtifactValidationError {
                // Any typed artifact rejection proves the malformed input fails closed.
            }
        }
    }

    @Test
    func should_reject_overflow_gaps_overlaps_out_of_bounds_and_trailing_bytes() throws {
        let maximumDimension: Int = Int.max;
        let aggregateOverflowHeader: String =
            "{\"first\":{\"dtype\":\"U8\",\"shape\":[\(maximumDimension)],\"data_offsets\":[0,\(maximumDimension)]},"
            + "\"second\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}";
        let malformedAggregateCases: Array<(header: String, payload: Data)> = [
            ("{\"tensor\":{\"dtype\":\"U64\",\"shape\":[18446744073709551615,2],\"data_offsets\":[0,0]}}", Data()),
            (aggregateOverflowHeader, Data()),
            ("{\"first\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},\"second\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[2,3]}}",
             Data(repeating: 0, count: 3)),
            ("{\"first\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]},\"second\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}",
             Data(repeating: 0, count: 2)),
            ("{\"tensor\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}", Data([0])),
            ("{\"tensor\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}", Data(repeating: 0, count: 2)),
        ];

        for malformedCase: (header: String, payload: Data) in malformedAggregateCases {
            let (modelDirectory, validatedWeightsFile): (URL, ValidatedWeightsFile) = try Self
                .validatedWeightsFileForHeader(
                    headerText: malformedCase.header, payloadBytes: malformedCase.payload);
            defer { try? FileManager.default.removeItem(at: modelDirectory); }
            do {
                _ = try validatedWeightsFile.readRawSafetensorsInventoryForTests();
                Issue.record("invalid aggregate or interval accounting must fail closed");
            } catch is ArtifactValidationError {
                // Any typed artifact rejection proves the accounting fails closed.
            }
        }
    }

    @Test
    func should_reject_an_empty_tensor_name_and_an_over_bounded_header() throws {
        let (emptyNameDirectory, emptyNameWeightsFile): (URL, ValidatedWeightsFile) = try Self
            .validatedWeightsFileForHeader(
                headerText: "{\"\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}",
                payloadBytes: Data([0]));
        defer { try? FileManager.default.removeItem(at: emptyNameDirectory); }
        do {
            _ = try emptyNameWeightsFile.readRawSafetensorsInventoryForTests();
            Issue.record("an empty tensor name must be rejected");
        } catch let validationError as ArtifactValidationError {
            guard case .invalidSafetensorsTensorName(_, let tensorNameLengthBytes, _) = validationError else {
                Issue.record("expected InvalidSafetensorsTensorName, got \(validationError)");
                return;
            }
            #expect(tensorNameLengthBytes == 0);
        }

        let declaredHeaderLengthBytes: UInt64 = 16 * 1024 * 1024 + 1;
        let modelDirectory: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: modelDirectory); }
        var lengthPrefixBytes: Array<UInt8> = Array();
        var littleEndianValue: UInt64 = declaredHeaderLengthBytes;
        withUnsafeBytes(of: &littleEndianValue) { (valueBuffer: UnsafeRawBufferPointer) -> Void in
            lengthPrefixBytes.append(contentsOf: Array(valueBuffer));
        };
        try Data(lengthPrefixBytes).write(
            to: modelDirectory.appendingPathComponent(Self.WEIGHTS_FILE_NAME));
        let validatedWeightsFile: ValidatedWeightsFile = try Self.validateWeightsFile(
            modelDirectory: modelDirectory, sizeBytes: 8);
        do {
            _ = try validatedWeightsFile.readRawSafetensorsInventoryForTests();
            Issue.record("a header beyond the format bound must be rejected");
        } catch let validationError as ArtifactValidationError {
            guard case .safetensorsHeaderLengthTooLarge = validationError else {
                Issue.record("expected SafetensorsHeaderLengthTooLarge, got \(validationError)");
                return;
            }
        }
    }

    @Test
    func should_continue_excluding_metadata_from_existing_profile_validation() throws {
        let header: String = "{\"__metadata__\":{\"format\":\"mlx\"},"
            + "\"tensor\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
        let framedBytes: Data = Self.safetensorsBytes(
            headerText: header, payloadBytes: Data(repeating: 0, count: 4));
        let framedFileUrl: URL = try Self.writeTemporaryFile(framedFileBytes: framedBytes);
        defer { try? FileManager.default.removeItem(at: framedFileUrl); }
        let fileHandle: FileHandle = try FileHandle(forReadingFrom: framedFileUrl);
        defer { fileHandle.closeFile(); }
        let metadata: BoundedSafetensors.PartialProfileMetadata = try BoundedSafetensors
            .validateBoundedSafetensorsWithPartialProfiles(
                fileHandle: fileHandle, fileSizeBytes: UInt64(framedBytes.count),
                weightsFileName: Self.WEIGHTS_FILE_NAME,
                profiledTensorProfiles: [TensorProfile(
                    name: "tensor", dtype: .float32, shape: [1],
                    equivalentPublishedShapes: [])],
                acceptedExtraTensorNames: Set());
        #expect(metadata.totalPayloadBytes == 4);
    }

    // MARK: - Fixture helpers

    private static func safetensorsBytes(headerText: String, payloadBytes: Data) -> Data {
        var framedBytes: Array<UInt8> = Array();
        var littleEndianHeaderLength: UInt64 = UInt64(headerText.utf8.count);
        withUnsafeBytes(of: &littleEndianHeaderLength) { (valueBuffer: UnsafeRawBufferPointer) -> Void in
            framedBytes.append(contentsOf: Array(valueBuffer));
        };
        framedBytes.append(contentsOf: Array(headerText.utf8));
        framedBytes.append(contentsOf: Array(payloadBytes));
        return Data(framedBytes);
    }

    private static func validatedWeightsFileForHeader(
        headerText: String, payloadBytes: Data) throws -> (URL, ValidatedWeightsFile) {
        let modelDirectory: URL = try makeTemporaryDirectory();
        let framedBytes: Data = safetensorsBytes(headerText: headerText, payloadBytes: payloadBytes);
        try framedBytes.write(to: modelDirectory.appendingPathComponent(WEIGHTS_FILE_NAME));
        let validatedWeightsFile: ValidatedWeightsFile = try validateWeightsFile(
            modelDirectory: modelDirectory, sizeBytes: UInt64(framedBytes.count));
        return (modelDirectory, validatedWeightsFile);
    }

    private static func validateWeightsFile(
        modelDirectory: URL, sizeBytes: UInt64) throws -> ValidatedWeightsFile {
        return try RequiredFiles.validateRequiredFileForTests(
            modelDirectory: modelDirectory.path,
            requiredFileProfile: RequiredFileProfile(
                fileName: WEIGHTS_FILE_NAME, sizeBytes: sizeBytes));
    }

    private static func makeTemporaryDirectory() throws -> URL {
        let directoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("raw-inventory-\(UUID().uuidString)");
        try FileManager.default.createDirectory(at: directoryUrl, withIntermediateDirectories: true);
        return directoryUrl;
    }

    private static func writeTemporaryFile(framedFileBytes: Data) throws -> URL {
        let framedFileUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("raw-inventory-\(UUID().uuidString).safetensors");
        try framedFileBytes.write(to: framedFileUrl);
        return framedFileUrl;
    }
}
