import XCTest;
import ModelServing;

/// Behavioral journeys for required-file validation, twin-porting
/// crates/model-serving/tests/hermetic/required_files.rs: Hugging Face
/// snapshot symlink confinement, shared-blob provenance, path-safety, and
/// retained-descriptor bounded reads.
final class RequiredFilesTests: XCTestCase {

    fileprivate static let CONFIG_BYTES: Data = Data(#"{"model_type":"qwen3_5_moe"}"#.utf8);
    /// Digest the shared store names its object after in the shared-blob fixtures.
    fileprivate static let SHARED_BLOB_CONTENT_DIGEST: String =
        "ab" + String(repeating: "0", count: 62);
    /// Digest the shared store does not name its object after; a snapshot tree
    /// record carrying it must not authenticate the fixture's shared blob.
    fileprivate static let UNRELATED_CONTENT_DIGEST: String =
        "cd" + String(repeating: "0", count: 62);

    fileprivate static func makeTemporaryDirectory() throws -> URL {
        let directoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("required-files-\(UUID().uuidString)");
        try FileManager.default.createDirectory(at: directoryUrl, withIntermediateDirectories: true);
        return directoryUrl;
    }

    func testShouldAcceptAHuggingFaceSnapshotSymlinkToItsOwnBlobDirectory() throws {
        let temporaryRoot: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: temporaryRoot); }
        let modelCacheDirectory: URL = temporaryRoot.appendingPathComponent("models--example--model");
        let blobDirectory: URL = modelCacheDirectory.appendingPathComponent("blobs");
        let snapshotDirectory: URL = modelCacheDirectory
            .appendingPathComponent("snapshots/commit-hash");
        try FileManager.default.createDirectory(at: blobDirectory, withIntermediateDirectories: true);
        try FileManager.default.createDirectory(at: snapshotDirectory, withIntermediateDirectories: true);
        try Self.CONFIG_BYTES.write(to: blobDirectory.appendingPathComponent("config-blob"));
        try FileManager.default.createSymbolicLink(
            atPath: snapshotDirectory.appendingPathComponent("config.json").path,
            withDestinationPath: "../../blobs/config-blob");

        let validatedWeightsFile: ValidatedWeightsFile = try RequiredFiles.validateRequiredFileForTests(
            modelDirectory: snapshotDirectory.path,
            requiredFileProfile: RequiredFileProfile(
                fileName: "config.json", sizeBytes: UInt64(Self.CONFIG_BYTES.count)));
        let validatedFileHandle: FileHandle = validatedWeightsFile.intoFile();
        let actualConfigBytes: Data = validatedFileHandle.readDataToEndOfFile();
        XCTAssertEqual(actualConfigBytes, Self.CONFIG_BYTES);
    }

    func testShouldAcceptAHuggingFaceSnapshotSymlinkToAVerifiedSharedBlob() throws {
        let sharedBlobLayout: SharedBlobLayout = try SharedBlobLayout.create();
        defer { try? FileManager.default.removeItem(at: sharedBlobLayout.hubDirectory); }
        try sharedBlobLayout.writeTreeRecord(
            recordedDigestText: Self.SHARED_BLOB_CONTENT_DIGEST,
            recordedSizeBytes: UInt64(Self.CONFIG_BYTES.count));

        XCTAssertNoThrow(try sharedBlobLayout.validate());
    }

    func testShouldRejectASharedBlobThatIsNotTheSnapshotRecordedContentAddress() throws {
        let sharedBlobLayout: SharedBlobLayout = try SharedBlobLayout.create();
        defer { try? FileManager.default.removeItem(at: sharedBlobLayout.hubDirectory); }
        try sharedBlobLayout.writeTreeRecord(
            recordedDigestText: Self.UNRELATED_CONTENT_DIGEST,
            recordedSizeBytes: UInt64(Self.CONFIG_BYTES.count));

        XCTAssertThrowsError(try sharedBlobLayout.validate(), "must fail closed") { thrownError in
            guard let validationError: ArtifactValidationError = thrownError as? ArtifactValidationError else {
                XCTFail("expected an ArtifactValidationError, got \(thrownError)");
                return;
            }
            XCTAssertEqual(
                validationError, .huggingFaceSharedBlobIdentityMismatch(
                    fileName: "config.json",
                    recordedDigestText: Self.UNRELATED_CONTENT_DIGEST));
        };
    }

    func testShouldRejectASharedBlobWhoseSizeDisagreesWithItsSnapshotTreeRecord() throws {
        let sharedBlobLayout: SharedBlobLayout = try SharedBlobLayout.create();
        defer { try? FileManager.default.removeItem(at: sharedBlobLayout.hubDirectory); }
        try sharedBlobLayout.writeTreeRecord(
            recordedDigestText: Self.SHARED_BLOB_CONTENT_DIGEST,
            recordedSizeBytes: UInt64(Self.CONFIG_BYTES.count + 1));

        XCTAssertThrowsError(try sharedBlobLayout.validate(), "must fail closed") { thrownError in
            guard let validationError: ArtifactValidationError = thrownError as? ArtifactValidationError else {
                XCTFail("expected an ArtifactValidationError, got \(thrownError)");
                return;
            }
            XCTAssertEqual(
                validationError, .huggingFaceSharedBlobSizeMismatch(
                    fileName: "config.json",
                    recordedSizeBytes: UInt64(Self.CONFIG_BYTES.count + 1),
                    actualSizeBytes: UInt64(Self.CONFIG_BYTES.count)));
        };
    }

    func testShouldRejectASharedBlobWithoutASnapshotTreeRecord() throws {
        let sharedBlobLayout: SharedBlobLayout = try SharedBlobLayout.create();
        defer { try? FileManager.default.removeItem(at: sharedBlobLayout.hubDirectory); }

        XCTAssertThrowsError(try sharedBlobLayout.validate(), "must fail closed") { thrownError in
            guard let validationError: ArtifactValidationError = thrownError as? ArtifactValidationError else {
                XCTFail("expected an ArtifactValidationError, got \(thrownError)");
                return;
            }
            guard case .huggingFaceSharedBlobMetadataUnavailable(let fileName, _) = validationError else {
                XCTFail("expected HuggingFaceSharedBlobMetadataUnavailable, got \(validationError)");
                return;
            }
            XCTAssertEqual(fileName, "config.json");
        };
    }

    func testShouldRejectAHuggingFaceSnapshotSymlinkThatEscapesItsBlobDirectory() throws {
        let temporaryRoot: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: temporaryRoot); }
        let modelCacheDirectory: URL = temporaryRoot.appendingPathComponent("models--example--model");
        let blobDirectory: URL = modelCacheDirectory.appendingPathComponent("blobs");
        let snapshotDirectory: URL = modelCacheDirectory
            .appendingPathComponent("snapshots/commit-hash");
        try FileManager.default.createDirectory(at: blobDirectory, withIntermediateDirectories: true);
        try FileManager.default.createDirectory(at: snapshotDirectory, withIntermediateDirectories: true);
        try Data("outside".utf8).write(to: temporaryRoot.appendingPathComponent("outside-config.json"));
        try FileManager.default.createSymbolicLink(
            atPath: snapshotDirectory.appendingPathComponent("config.json").path,
            withDestinationPath: "../../../outside-config.json");

        XCTAssertThrowsError(
            try RequiredFiles.validateRequiredFileForTests(
                modelDirectory: snapshotDirectory.path,
                requiredFileProfile: RequiredFileProfile(fileName: "config.json", sizeBytes: 0)),
            "a snapshot symlink outside its own blob directory must fail closed") { thrownError in
            guard let validationError: ArtifactValidationError = thrownError as? ArtifactValidationError else {
                XCTFail("expected an ArtifactValidationError, got \(thrownError)");
                return;
            }
            guard case .huggingFaceSnapshotSymlinkEscapesBlobDirectory(let fileName, _, _) = validationError else {
                XCTFail("expected HuggingFaceSnapshotSymlinkEscapesBlobDirectory, got \(validationError)");
                return;
            }
            XCTAssertEqual(fileName, "config.json");
        };
    }

    func testShouldContinueRejectingSymlinksInRegularModelDirectories() throws {
        let modelDirectory: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: modelDirectory); }
        try Data("contents".utf8).write(
            to: modelDirectory.appendingPathComponent("config-contents.json"));
        try FileManager.default.createSymbolicLink(
            atPath: modelDirectory.appendingPathComponent("config.json").path,
            withDestinationPath: "config-contents.json");

        XCTAssertThrowsError(
            try RequiredFiles.validateRequiredFileForTests(
                modelDirectory: modelDirectory.path,
                requiredFileProfile: RequiredFileProfile(fileName: "config.json", sizeBytes: 0)),
            "regular model directories must continue rejecting symlinks") { thrownError in
            XCTAssertEqual(
                thrownError as? ArtifactValidationError,
                .requiredFileIsSymlink(fileName: "config.json"));
        };
    }

    func testShouldRejectARequiredFileNameWithParentDirectoryComponents() throws {
        let modelDirectory: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: modelDirectory); }
        try Data("contents".utf8).write(to: modelDirectory.appendingPathComponent("outside.json"));

        XCTAssertThrowsError(
            try RequiredFiles.validateRequiredFileForTests(
                modelDirectory: modelDirectory.path,
                requiredFileProfile: RequiredFileProfile(fileName: "../outside.json", sizeBytes: 0)),
            "required file names must not escape the model directory") { thrownError in
            XCTAssertEqual(
                thrownError as? ArtifactValidationError,
                .invalidProfileFileName(fileName: "../outside.json"));
        };
    }

    func testShouldReadAnOrdinaryJsonSidecarThroughItsRetainedDescriptor() throws {
        let modelDirectory: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: modelDirectory); }
        let sidecarFileName: String = "model.safetensors.index.json";
        let sidecarUrl: URL = modelDirectory.appendingPathComponent(sidecarFileName);
        let retainedSidecarBytes: Data = Data(
            #"{"weight_map":{"model.weight":"model.safetensors"}}"#.utf8);
        try retainedSidecarBytes.write(to: sidecarUrl);
        let validatedRequiredFile: ValidatedWeightsFile = try RequiredFiles.validateRequiredFileForTests(
            modelDirectory: modelDirectory.path,
            requiredFileProfile: RequiredFileProfile(
                fileName: sidecarFileName, sizeBytes: UInt64(retainedSidecarBytes.count)));

        // Replacing the pathname proves that the read stays on the retained inode.
        try FileManager.default.moveItem(
            at: sidecarUrl, to: modelDirectory.appendingPathComponent("validated-index.json"));
        try Data(#"{"replacement":true}"#.utf8).write(to: sidecarUrl);

        let actualSidecarBytes: Data = try validatedRequiredFile.readBoundedBytesForTests(
            maximumSizeBytes: UInt64(retainedSidecarBytes.count));
        XCTAssertEqual(actualSidecarBytes, retainedSidecarBytes);
    }

    func testShouldRejectABoundedRequiredFileReadAboveItsExplicitLimit() throws {
        let modelDirectory: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: modelDirectory); }
        let sidecarFileName: String = "model.safetensors.index.json";
        let sidecarBytes: Data = Data(#"{"weight_map":{}}"#.utf8);
        try sidecarBytes.write(to: modelDirectory.appendingPathComponent(sidecarFileName));
        let validatedRequiredFile: ValidatedWeightsFile = try RequiredFiles.validateRequiredFileForTests(
            modelDirectory: modelDirectory.path,
            requiredFileProfile: RequiredFileProfile(
                fileName: sidecarFileName, sizeBytes: UInt64(sidecarBytes.count)));

        XCTAssertThrowsError(
            try validatedRequiredFile.readBoundedBytesForTests(
                maximumSizeBytes: UInt64(sidecarBytes.count - 1)),
            "a sidecar above the caller's explicit limit must fail closed") { thrownError in
            XCTAssertEqual(
                thrownError as? ArtifactValidationError, .boundedRequiredFileTooLarge(
                    fileName: sidecarFileName,
                    actualSizeBytes: UInt64(sidecarBytes.count),
                    maximumSizeBytes: UInt64(sidecarBytes.count - 1)));
        };
    }

    func testShouldPreserveTheSourceWhenARetainedDescriptorBecomesShort() throws {
        let modelDirectory: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: modelDirectory); }
        let sidecarFileName: String = "model.safetensors.index.json";
        let sidecarUrl: URL = modelDirectory.appendingPathComponent(sidecarFileName);
        let sidecarBytes: Data = Data(
            #"{"weight_map":{"model.weight":"model.safetensors"}}"#.utf8);
        try sidecarBytes.write(to: sidecarUrl);
        let validatedRequiredFile: ValidatedWeightsFile = try RequiredFiles.validateRequiredFileForTests(
            modelDirectory: modelDirectory.path,
            requiredFileProfile: RequiredFileProfile(
                fileName: sidecarFileName, sizeBytes: UInt64(sidecarBytes.count)));
        try Data().write(to: sidecarUrl);

        XCTAssertThrowsError(
            try validatedRequiredFile.readBoundedBytesForTests(
                maximumSizeBytes: UInt64(sidecarBytes.count)),
            "a short retained descriptor must fail with its read source") { thrownError in
            XCTAssertEqual(
                thrownError as? ArtifactValidationError, .readBoundedRequiredFile(
                    fileName: sidecarFileName, problem: "failed to fill whole buffer"));
        };
    }

    func testShouldRejectADuplicateRequiredProfileBeforeReplacingTheFirstFile() throws {
        let modelDirectory: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: modelDirectory); }
        try Self.CONFIG_BYTES.write(to: modelDirectory.appendingPathComponent("config.json"));
        let duplicateProfiles: Array<RequiredFileProfile> = [
            RequiredFileProfile(
                fileName: "config.json", sizeBytes: UInt64(Self.CONFIG_BYTES.count)),
            RequiredFileProfile(
                fileName: "config.json", sizeBytes: UInt64(Self.CONFIG_BYTES.count + 1)),
        ];

        XCTAssertThrowsError(
            try duplicateProfiles[0].validateAllForTests(
                modelDirectory: modelDirectory.path, requiredFileProfiles: duplicateProfiles),
            "a repeated profile name must fail instead of replacing its first descriptor") { thrownError in
            XCTAssertEqual(
                thrownError as? ArtifactValidationError,
                .duplicateProfileFileName(fileName: "config.json"));
        };
    }
}

/// One Hugging Face hub whose model entry reaches an object in the hub-level
/// shared blob store, so each test can vary exactly one verification input.
/// Twin-port of the Rust SharedBlobLayout fixture.
private final class SharedBlobLayout {
    let hubDirectory: URL;
    let snapshotDirectory: URL;

    private init(hubDirectory: URL, snapshotDirectory: URL) {
        self.hubDirectory = hubDirectory;
        self.snapshotDirectory = snapshotDirectory;
    }

    /// Creates the shared store object, the entry's local blob link, the
    /// snapshot symlink, and the entry's tree metadata directory.
    static func create() throws -> SharedBlobLayout {
        let hubDirectory: URL = try RequiredFilesTests.makeTemporaryDirectory();
        let modelCacheDirectory: URL = hubDirectory.appendingPathComponent("models--example--model");
        let localBlobDirectory: URL = modelCacheDirectory.appendingPathComponent("blobs");
        let snapshotDirectory: URL = modelCacheDirectory
            .appendingPathComponent("snapshots/commit-hash");
        let sharedBlobDirectory: URL = hubDirectory
            .appendingPathComponent("blobs/ab");
        try FileManager.default.createDirectory(at: localBlobDirectory, withIntermediateDirectories: true);
        try FileManager.default.createDirectory(at: sharedBlobDirectory, withIntermediateDirectories: true);
        try FileManager.default.createDirectory(at: snapshotDirectory, withIntermediateDirectories: true);
        try FileManager.default.createDirectory(
            at: modelCacheDirectory.appendingPathComponent("trees"), withIntermediateDirectories: true);
        try RequiredFilesTests.CONFIG_BYTES.write(
            to: sharedBlobDirectory
                .appendingPathComponent(RequiredFilesTests.SHARED_BLOB_CONTENT_DIGEST));
        try FileManager.default.createSymbolicLink(
            atPath: localBlobDirectory.appendingPathComponent("local-config-blob").path,
            withDestinationPath: "../../blobs/ab/"
                + RequiredFilesTests.SHARED_BLOB_CONTENT_DIGEST);
        try FileManager.default.createSymbolicLink(
            atPath: snapshotDirectory.appendingPathComponent("config.json").path,
            withDestinationPath: "../../blobs/local-config-blob");

        return SharedBlobLayout(hubDirectory: hubDirectory, snapshotDirectory: snapshotDirectory);
    }

    /// Writes the snapshot tree record that anchors trust for the shared blob.
    func writeTreeRecord(recordedDigestText: String, recordedSizeBytes: UInt64) throws {
        let treeRecordJsonText: String = """
        {"format_version":1,"files":{"config.json":{"size":\(recordedSizeBytes),"blob_id":"git-blob-id","lfs_sha256":"\(recordedDigestText)","lfs_size":\(recordedSizeBytes)}}}
        """;
        try Data(treeRecordJsonText.utf8).write(
            to: self.hubDirectory
                .appendingPathComponent("models--example--model/trees/commit-hash.json"));
    }

    func validate() throws -> Void {
        _ = try RequiredFiles.validateRequiredFileForTests(
            modelDirectory: self.snapshotDirectory.path,
            requiredFileProfile: RequiredFileProfile(
                fileName: "config.json", sizeBytes: UInt64(RequiredFilesTests.CONFIG_BYTES.count)));
    }
}
