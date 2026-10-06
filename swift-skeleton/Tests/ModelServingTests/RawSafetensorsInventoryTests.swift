import XCTest;
import ModelServing;

/// Behavioral journeys for the strict family-neutral raw inventory and the
/// bounded partial-profile validator, twin-porting
/// crates/model-serving/tests/hermetic/raw_safetensors_inventory.rs.
final class RawSafetensorsInventoryTests: XCTestCase {

    private static let WEIGHTS_FILE_NAME: String = "model.safetensors";

    func testShouldInventoryTheRetainedDescriptorAfterTheArtifactPathIsReplaced() throws {
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

        XCTAssertEqual(inventory.shardPayloadBytes, 8);
        XCTAssertEqual(inventory.tensorDescriptors.count, 2);
        XCTAssertEqual(inventory.tensorDescriptors[0].tensorName, "a.tensor");
        XCTAssertEqual(inventory.tensorDescriptors[0].dtype.canonicalName, "I32");
        XCTAssertEqual(inventory.tensorDescriptors[0].shape, [1]);
        XCTAssertEqual(inventory.tensorDescriptors[0].dataStartOffsetBytes, payloadStart + 4);
        XCTAssertEqual(inventory.tensorDescriptors[0].dataEndOffsetBytes, payloadStart + 8);
        XCTAssertEqual(inventory.tensorDescriptors[1].tensorName, "z.tensor");
        let metadataExcluded: Bool = inventory.tensorDescriptors.allSatisfy { (descriptor: RawSafetensorsTensorDescriptor) -> Bool in
            return descriptor.tensorName != "__metadata__";
        };
        XCTAssertTrue(metadataExcluded);
    }

    func testShouldAcceptEveryDtypeSupportedByTheSafetensorsFormat() throws {
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
            XCTAssertEqual(inventory.tensorDescriptors[0].dtype.canonicalName, dtypeCase.dtypeName);
            XCTAssertEqual(
                inventory.tensorDescriptors[0].tensorPayloadBytes,
                UInt64(dtypeCase.payloadLengthBytes));
        }
    }

    func testShouldRejectDuplicateTensorAndNestedObjectKeys() throws {
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
            XCTAssertThrowsError(
                try validatedWeightsFile.readRawSafetensorsInventoryForTests(),
                "duplicate object keys must fail before replacement") { thrownError in
                guard case .invalidSafetensorsHeader(_, let problem) = thrownError as! ArtifactValidationError else {
                    XCTFail("expected InvalidSafetensorsHeader, got \(thrownError)");
                    return;
                }
                XCTAssertTrue(
                    problem.lowercased().contains("duplicate"),
                    "the duplicate-key rejection must name the duplication, got: \(problem)");
            };
        }
    }

    func testShouldRejectInvalidDtypesShapesAndIntervals() throws {
        let cases: Array<(header: String, payload: Data, caseName: String)> = [
            ("{\"tensor\":{\"dtype\":\"UNKNOWN\",\"shape\":[1],\"data_offsets\":[0,1]}}",
             Data([0]), "dtype"),
            ("{\"tensor\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}",
             Data(repeating: 0, count: 4), "width"),
            ("{\"tensor\":{\"dtype\":\"F4\",\"shape\":[1],\"data_offsets\":[0,1]}}",
             Data([0]), "alignment"),
            ("{\"tensor\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,0]}}",
             Data([0]), "reversed"),
        ];

        for invalidCase: (header: String, payload: Data, caseName: String) in cases {
            let (modelDirectory, validatedWeightsFile): (URL, ValidatedWeightsFile) = try Self
                .validatedWeightsFileForHeader(
                    headerText: invalidCase.header, payloadBytes: invalidCase.payload);
            defer { try? FileManager.default.removeItem(at: modelDirectory); }
            XCTAssertThrowsError(
                try validatedWeightsFile.readRawSafetensorsInventoryForTests(),
                "\(invalidCase.caseName) must fail closed");
        }
    }

    func testShouldRejectOverflowGapsOverlapsOutOfBoundsAndTrailingBytes() throws {
        let maximumDimension: Int = Int.max;
        let aggregateOverflowHeader: String =
            "{\"first\":{\"dtype\":\"U8\",\"shape\":[\(maximumDimension)],\"data_offsets\":[0,\(maximumDimension)]},"
            + "\"second\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}";
        let cases: Array<(header: String, payload: Data)> = [
            ("{\"tensor\":{\"dtype\":\"U64\",\"shape\":[18446744073709551615,2],\"data_offsets\":[0,0]}}", Data()),
            (aggregateOverflowHeader, Data()),
            ("{\"first\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},\"second\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[2,3]}}",
             Data(repeating: 0, count: 3)),
            ("{\"first\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]},\"second\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}",
             Data(repeating: 0, count: 2)),
            ("{\"tensor\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}", Data([0])),
            ("{\"tensor\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}", Data(repeating: 0, count: 2)),
        ];

        for invalidCase: (header: String, payload: Data) in cases {
            let (modelDirectory, validatedWeightsFile): (URL, ValidatedWeightsFile) = try Self
                .validatedWeightsFileForHeader(
                    headerText: invalidCase.header, payloadBytes: invalidCase.payload);
            defer { try? FileManager.default.removeItem(at: modelDirectory); }
            XCTAssertThrowsError(
                try validatedWeightsFile.readRawSafetensorsInventoryForTests(),
                "invalid aggregate or interval accounting must fail closed");
        }
    }

    func testShouldRejectAnEmptyTensorNameAndAnOverBoundedHeader() throws {
        let (emptyNameDirectory, emptyNameWeightsFile): (URL, ValidatedWeightsFile) = try Self
            .validatedWeightsFileForHeader(
                headerText: "{\"\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}",
                payloadBytes: Data([0]));
        defer { try? FileManager.default.removeItem(at: emptyNameDirectory); }
        XCTAssertThrowsError(try emptyNameWeightsFile.readRawSafetensorsInventoryForTests()) { thrownError in
            guard case .invalidSafetensorsTensorName(_, let tensorNameLengthBytes, _) = thrownError as! ArtifactValidationError else {
                XCTFail("expected InvalidSafetensorsTensorName, got \(thrownError)");
                return;
            }
            XCTAssertEqual(tensorNameLengthBytes, 0);
        };

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
        XCTAssertThrowsError(try validatedWeightsFile.readRawSafetensorsInventoryForTests()) { thrownError in
            guard case .safetensorsHeaderLengthTooLarge = thrownError as! ArtifactValidationError else {
                XCTFail("expected SafetensorsHeaderLengthTooLarge, got \(thrownError)");
                return;
            }
        };
    }

    func testShouldContinueExcludingMetadataFromExistingProfileValidation() throws {
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
        XCTAssertEqual(metadata.totalPayloadBytes, 4);
    }

    // MARK: - Fixture helpers

    fileprivate static func safetensorsBytes(headerText: String, payloadBytes: Data) -> Data {
        var framedBytes: Array<UInt8> = Array();
        var littleEndianHeaderLength: UInt64 = UInt64(headerText.utf8.count);
        withUnsafeBytes(of: &littleEndianHeaderLength) { (valueBuffer: UnsafeRawBufferPointer) -> Void in
            framedBytes.append(contentsOf: Array(valueBuffer));
        };
        framedBytes.append(contentsOf: Array(headerText.utf8));
        framedBytes.append(contentsOf: Array(payloadBytes));
        return Data(framedBytes);
    }

    fileprivate static func validatedWeightsFileForHeader(
        headerText: String, payloadBytes: Data) throws -> (URL, ValidatedWeightsFile) {
        let modelDirectory: URL = try makeTemporaryDirectory();
        let framedBytes: Data = safetensorsBytes(headerText: headerText, payloadBytes: payloadBytes);
        try framedBytes.write(to: modelDirectory.appendingPathComponent(WEIGHTS_FILE_NAME));
        let validatedWeightsFile: ValidatedWeightsFile = try validateWeightsFile(
            modelDirectory: modelDirectory, sizeBytes: UInt64(framedBytes.count));
        return (modelDirectory, validatedWeightsFile);
    }

    fileprivate static func validateWeightsFile(
        modelDirectory: URL, sizeBytes: UInt64) throws -> ValidatedWeightsFile {
        return try RequiredFiles.validateRequiredFileForTests(
            modelDirectory: modelDirectory.path,
            requiredFileProfile: RequiredFileProfile(
                fileName: WEIGHTS_FILE_NAME, sizeBytes: sizeBytes));
    }

    fileprivate static func makeTemporaryDirectory() throws -> URL {
        let directoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("raw-inventory-\(UUID().uuidString)");
        try FileManager.default.createDirectory(at: directoryUrl, withIntermediateDirectories: true);
        return directoryUrl;
    }

    fileprivate static func writeTemporaryFile(framedFileBytes: Data) throws -> URL {
        let framedFileUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("raw-inventory-\(UUID().uuidString).safetensors");
        try framedFileBytes.write(to: framedFileUrl);
        return framedFileUrl;
    }
}
