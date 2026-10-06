import XCTest;
import ModelServing;

/// Behavioral journeys for the optional Qwen MTP sidecar, twin-porting
/// crates/model-serving/tests/qwen3_5_hermetic/mtp_sidecar.rs: fail-open
/// target-only degradation, structured diagnostics, and declaration path
/// safety.
final class Qwen35MtpSidecarTests: XCTestCase {

    private func mtpProfiles() -> Array<TensorProfile> {
        return [
            TensorProfile(
                name: "language_model.mtp.proj.weight", dtype: .uint32, shape: [1],
                equivalentPublishedShapes: []),
            TensorProfile(
                name: "language_model.mtp.proj.scales", dtype: .affineQuantizationFloat,
                shape: [1], equivalentPublishedShapes: []),
            TensorProfile(
                name: "language_model.mtp.proj.biases", dtype: .affineQuantizationFloat,
                shape: [1], equivalentPublishedShapes: []),
        ];
    }

    private func completeSidecarTensors() -> Array<(storedName: String, dtype: String, payload: Data)> {
        return [
            ("mtp.proj.weight", "U32", Data([0, 0, 0, 0])),
            ("mtp.proj.scales", "BF16", Data([0, 0])),
            ("mtp.proj.biases", "BF16", Data([0, 0])),
        ];
    }

    private func writeSidecar(
        modelDirectory: URL, relativePath: String,
        tensors: Array<(storedName: String, dtype: String, payload: Data)>) throws {
        let sidecarUrl: URL = modelDirectory.appendingPathComponent(relativePath);
        let parentDirectory: URL = sidecarUrl.deletingLastPathComponent();
        try FileManager.default.createDirectory(
            at: parentDirectory, withIntermediateDirectories: true);
        var payloadBytes: Array<UInt8> = Array();
        var headerEntries: Array<String> = Array();
        for sidecarTensor: (storedName: String, dtype: String, payload: Data) in tensors {
            let startOffset: Int = payloadBytes.count;
            payloadBytes.append(contentsOf: Array(sidecarTensor.payload));
            headerEntries.append(
                "\"\(sidecarTensor.storedName)\":{\"dtype\":\"\(sidecarTensor.dtype)\","
                    + "\"shape\":[1],\"data_offsets\":[\(startOffset),\(payloadBytes.count)]}");
        }
        let header: String = "{" + headerEntries.joined(separator: ",") + "}";
        try Data(Self.framedBytes(
            headerText: header, payloadByteValues: payloadBytes)).write(to: sidecarUrl);
    }

    private func writeRawSidecar(
        modelDirectory: URL, headerText: String, payloadValues: Array<UInt8>) throws {
        try Data(Self.framedBytes(
            headerText: headerText, payloadByteValues: payloadValues)).write(
            to: modelDirectory.appendingPathComponent("mtp.safetensors"));
    }

    func testShouldAcceptRootAndNestedQwenMtpSidecarsWithExactAccounting() throws {
        for relativePath: String in ["mtp.safetensors", "optiq/mtp.safetensors"] {
            let modelDirectory: URL = try Self.makeTemporaryDirectory();
            defer { try? FileManager.default.removeItem(at: modelDirectory); }
            try self.writeSidecar(
                modelDirectory: modelDirectory, relativePath: relativePath,
                tensors: self.completeSidecarTensors());
            let declaration: Qwen35MtpSidecarDeclaration = try Qwen35MtpSidecarDeclaration
                .parse(relativePath: relativePath);

            let outcome: Qwen35MtpSidecarValidationOutcome = Qwen35MtpSidecar.validateForTests(
                modelDirectory: modelDirectory.path, declaration: declaration,
                canonicalProfiles: self.mtpProfiles(), existingCanonicalNames: []);

            XCTAssertTrue(outcome.isAvailable());
            XCTAssertEqual(outcome.sourceCount(), 1);
            XCTAssertEqual(outcome.tensorCount(), 3);
            XCTAssertEqual(outcome.payloadBytes(), 8);
            XCTAssertEqual(
                outcome.storedName(canonicalName: "language_model.mtp.proj.weight"),
                "mtp.proj.weight");
        }
    }

    func testShouldPreserveTargetOnlyForMissingMalformedPartialOrWrongDtypeSidecars() throws {
        let scenarios: Array<(scenarioName: String, tensors: Array<(storedName: String, dtype: String, payload: Data)>)> = [
            ("malformed", []),
            ("partial", [("mtp.proj.weight", "U32", Data([0, 0, 0, 0]))]),
            ("wrong-dtype", [
                ("mtp.proj.weight", "BF16", Data([0, 0])),
                ("mtp.proj.scales", "BF16", Data([0, 0])),
                ("mtp.proj.biases", "BF16", Data([0, 0])),
            ]),
        ];
        for scenario: (scenarioName: String, tensors: Array<(storedName: String, dtype: String, payload: Data)>) in scenarios {
            let modelDirectory: URL = try Self.makeTemporaryDirectory();
            defer { try? FileManager.default.removeItem(at: modelDirectory); }
            if scenario.scenarioName == "malformed" {
                try Data("invalid".utf8).write(
                    to: modelDirectory.appendingPathComponent("mtp.safetensors"));
            } else {
                try self.writeSidecar(
                    modelDirectory: modelDirectory, relativePath: "mtp.safetensors",
                    tensors: scenario.tensors);
            }
            let declaration: Qwen35MtpSidecarDeclaration = try Qwen35MtpSidecarDeclaration
                .parse(relativePath: "mtp.safetensors");
            let outcome: Qwen35MtpSidecarValidationOutcome = Qwen35MtpSidecar.validateForTests(
                modelDirectory: modelDirectory.path, declaration: declaration,
                canonicalProfiles: self.mtpProfiles(), existingCanonicalNames: []);
            XCTAssertFalse(outcome.isAvailable(), "scenario \(scenario.scenarioName)");
            XCTAssertEqual(outcome.sourceCount(), 0, "scenario \(scenario.scenarioName)");
            XCTAssertEqual(outcome.payloadBytes(), 0, "scenario \(scenario.scenarioName)");
        }

        let missingDirectory: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: missingDirectory); }
        let missingDeclaration: Qwen35MtpSidecarDeclaration = try Qwen35MtpSidecarDeclaration
            .parse(relativePath: "missing.safetensors");
        let missingOutcome: Qwen35MtpSidecarValidationOutcome = Qwen35MtpSidecar.validateForTests(
            modelDirectory: missingDirectory.path, declaration: missingDeclaration,
            canonicalProfiles: self.mtpProfiles(), existingCanonicalNames: []);
        XCTAssertFalse(missingOutcome.isAvailable());
    }

    func testShouldAcceptSidecarWithExtraUndeclaredTensorsAsFutureExtensibility() throws {
        let modelDirectory: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: modelDirectory); }
        var extraTensors: Array<(storedName: String, dtype: String, payload: Data)> =
            self.completeSidecarTensors();
        extraTensors.append(("mtp.unexpected", "BF16", Data([0, 0])));
        try self.writeSidecar(
            modelDirectory: modelDirectory, relativePath: "mtp.safetensors",
            tensors: extraTensors);
        let declaration: Qwen35MtpSidecarDeclaration = try Qwen35MtpSidecarDeclaration
            .parse(relativePath: "mtp.safetensors");
        let outcome: Qwen35MtpSidecarValidationOutcome = Qwen35MtpSidecar.validateForTests(
            modelDirectory: modelDirectory.path, declaration: declaration,
            canonicalProfiles: self.mtpProfiles(), existingCanonicalNames: []);
        XCTAssertTrue(
            outcome.isAvailable(),
            "an extra undeclared tensor must not reject the sidecar");
        XCTAssertEqual(outcome.tensorCount(), 3, "only profiled tensors enter the inventory");
    }

    func testShouldReportStructuredSidecarValidationErrors() throws {
        // A dtype mismatch yields a ProfileValidationFailed diagnostic with a
        // human-readable cause.
        let wrongDtypeDirectory: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: wrongDtypeDirectory); }
        try self.writeSidecar(
            modelDirectory: wrongDtypeDirectory, relativePath: "mtp.safetensors",
            tensors: [
                ("mtp.proj.weight", "BF16", Data([0, 0])),
                ("mtp.proj.scales", "BF16", Data([0, 0])),
                ("mtp.proj.biases", "BF16", Data([0, 0])),
            ]);
        let declaration: Qwen35MtpSidecarDeclaration = try Qwen35MtpSidecarDeclaration
            .parse(relativePath: "mtp.safetensors");
        XCTAssertThrowsError(
            try Qwen35MtpSidecar.validateResultForTests(
                modelDirectory: wrongDtypeDirectory.path, declaration: declaration,
                canonicalProfiles: self.mtpProfiles(), existingCanonicalNames: []),
            "a dtype mismatch must fail validation") { thrownError in
            guard let sidecarError: Qwen35MtpSidecarValidationError =
                thrownError as? Qwen35MtpSidecarValidationError else {
                XCTFail("expected a sidecar validation error, got \(thrownError)");
                return;
            }
            guard case .profileValidationFailed(let tensorName, _) = sidecarError else {
                XCTFail("expected ProfileValidationFailed, got \(sidecarError)");
                return;
            }
            XCTAssertEqual(tensorName, "language_model.mtp.proj.weight");
            XCTAssertTrue(
                sidecarError.errorDescription?.contains("dtype mismatch") ?? false,
                "diagnostic was: \(sidecarError.errorDescription ?? "")");
        };

        // A partial sidecar (missing profile tensors) yields a
        // MissingProfileTensor diagnostic.
        let partialDirectory: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: partialDirectory); }
        try self.writeSidecar(
            modelDirectory: partialDirectory, relativePath: "mtp.safetensors",
            tensors: [("mtp.proj.weight", "U32", Data([0, 0, 0, 0]))]);
        XCTAssertThrowsError(
            try Qwen35MtpSidecar.validateResultForTests(
                modelDirectory: partialDirectory.path, declaration: declaration,
                canonicalProfiles: self.mtpProfiles(), existingCanonicalNames: []),
            "a partial sidecar must fail validation") { thrownError in
            XCTAssertEqual(
                thrownError as? Qwen35MtpSidecarValidationError,
                .missingProfileTensor(tensorName: "language_model.mtp.proj.scales"));
            XCTAssertTrue(
                (thrownError as? Qwen35MtpSidecarValidationError)?.errorDescription?
                    .contains("not found in sidecar") ?? false,
                "diagnostic was: \(thrownError)");
        };
    }

    func testShouldRejectUnsafeQwenSidecarPathsBeforeFilesystemAccess() throws {
        for unsafePath: String in [
            "", "/mtp.safetensors", "../mtp.safetensors", "weights/../mtp.safetensors",
            "./mtp.safetensors", "weights//mtp.safetensors", "C:\\mtp.safetensors", "mtp.bin",
        ] {
            XCTAssertThrowsError(
                try Qwen35MtpSidecarDeclaration.parse(relativePath: unsafePath),
                "unsafe path '\(unsafePath)'") { thrownError in
                XCTAssertEqual(
                    thrownError as? Qwen35MtpSidecarDeclarationError,
                    .unsafeRelativePath);
            };
        }
    }

    func testShouldDisableOptionalMtpWhenEmbeddedAndSidecarCanonicalNamesCollide() throws {
        let modelDirectory: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: modelDirectory); }
        try self.writeSidecar(
            modelDirectory: modelDirectory, relativePath: "mtp.safetensors",
            tensors: self.completeSidecarTensors());
        let declaration: Qwen35MtpSidecarDeclaration = try Qwen35MtpSidecarDeclaration
            .parse(relativePath: "mtp.safetensors");
        let outcome: Qwen35MtpSidecarValidationOutcome = Qwen35MtpSidecar.validateForTests(
            modelDirectory: modelDirectory.path, declaration: declaration,
            canonicalProfiles: self.mtpProfiles(),
            existingCanonicalNames: ["language_model.mtp.proj.weight"]);

        XCTAssertFalse(outcome.isAvailable());
        XCTAssertEqual(outcome.sourceCount(), 0);
        XCTAssertEqual(outcome.tensorCount(), 0);
    }

    func testShouldPreserveTargetOnlyForWrongShapeAndInvalidOffsets() throws {
        let wrongShapeDirectory: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: wrongShapeDirectory); }
        try self.writeRawSidecar(
            modelDirectory: wrongShapeDirectory,
            headerText: "{\"mtp.proj.weight\":{\"dtype\":\"U32\",\"shape\":[2],\"data_offsets\":[0,8]},"
                + "\"mtp.proj.scales\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[8,10]},"
                + "\"mtp.proj.biases\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[10,12]}}",
            payloadValues: Array(repeating: 0, count: 12));
        let invalidOffsetDirectory: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: invalidOffsetDirectory); }
        try self.writeRawSidecar(
            modelDirectory: invalidOffsetDirectory,
            headerText: "{\"mtp.proj.weight\":{\"dtype\":\"U32\",\"shape\":[1],\"data_offsets\":[1,5]},"
                + "\"mtp.proj.scales\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[5,7]},"
                + "\"mtp.proj.biases\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[7,9]}}",
            payloadValues: Array(repeating: 0, count: 9));
        let declaration: Qwen35MtpSidecarDeclaration = try Qwen35MtpSidecarDeclaration
            .parse(relativePath: "mtp.safetensors");

        for modelDirectory: URL in [wrongShapeDirectory, invalidOffsetDirectory] {
            let outcome: Qwen35MtpSidecarValidationOutcome = Qwen35MtpSidecar.validateForTests(
                modelDirectory: modelDirectory.path, declaration: declaration,
                canonicalProfiles: self.mtpProfiles(), existingCanonicalNames: []);
            XCTAssertFalse(outcome.isAvailable());
            XCTAssertEqual(outcome.sourceCount(), 0);
        }
    }

    func testShouldPreserveTargetOnlyWhenAnOrdinarySidecarSymlinkEscapesTheModelDirectory() throws {
        let modelDirectory: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: modelDirectory); }
        let outsideDirectory: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: outsideDirectory); }
        try self.writeSidecar(
            modelDirectory: outsideDirectory, relativePath: "outside.safetensors",
            tensors: self.completeSidecarTensors());
        try FileManager.default.createSymbolicLink(
            atPath: modelDirectory.appendingPathComponent("mtp.safetensors").path,
            withDestinationPath: outsideDirectory.appendingPathComponent("outside.safetensors").path);
        let declaration: Qwen35MtpSidecarDeclaration = try Qwen35MtpSidecarDeclaration
            .parse(relativePath: "mtp.safetensors");

        let outcome: Qwen35MtpSidecarValidationOutcome = Qwen35MtpSidecar.validateForTests(
            modelDirectory: modelDirectory.path, declaration: declaration,
            canonicalProfiles: self.mtpProfiles(), existingCanonicalNames: []);

        XCTAssertFalse(outcome.isAvailable());
        XCTAssertEqual(outcome.sourceCount(), 0);
    }

    // MARK: - Fixture helpers

    fileprivate static func framedBytes(
        headerText: String, payloadByteValues: Array<UInt8>) -> Data {
        var framedFileBytes: Array<UInt8> = Array();
        var littleEndianHeaderLength: UInt64 = UInt64(headerText.utf8.count);
        withUnsafeBytes(of: &littleEndianHeaderLength) { (valueBuffer: UnsafeRawBufferPointer) -> Void in
            framedFileBytes.append(contentsOf: Array(valueBuffer));
        };
        framedFileBytes.append(contentsOf: Array(headerText.utf8));
        framedFileBytes.append(contentsOf: payloadByteValues);
        return Data(framedFileBytes);
    }

    fileprivate static func makeTemporaryDirectory() throws -> URL {
        let directoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtp-sidecar-\(UUID().uuidString)");
        try FileManager.default.createDirectory(at: directoryUrl, withIntermediateDirectories: true);
        return directoryUrl;
    }
}
