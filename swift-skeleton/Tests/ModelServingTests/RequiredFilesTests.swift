import Foundation;
import ModelServing;
import Testing;
import JourneyCategories;

/**
 * Behavioral journeys for required-file validation, twin-porting
 * crates/model-serving/tests/hermetic/required_files.rs: Hugging Face
 * snapshot symlink confinement, shared-blob provenance, path-safety, and
 * retained-descriptor bounded reads.
 */
@Suite(.tags(.hermeticJourney))
final class RequiredFilesTests {

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

    @Test
    func should_accept_a_hugging_face_snapshot_symlink_to_its_own_blob_directory() throws {
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
        #expect(actualConfigBytes == Self.CONFIG_BYTES);
    }

    @Test
    func should_accept_a_hugging_face_snapshot_symlink_to_a_verified_shared_blob() throws {
        let sharedBlobLayout: SharedBlobLayout = try SharedBlobLayout.create();
        defer { try? FileManager.default.removeItem(at: sharedBlobLayout.hubDirectory); }
        try sharedBlobLayout.writeTreeRecord(
            recordedDigestText: Self.SHARED_BLOB_CONTENT_DIGEST,
            recordedSizeBytes: UInt64(Self.CONFIG_BYTES.count));

        try sharedBlobLayout.validate();
    }

    @Test
    func should_reject_a_shared_blob_that_is_not_the_snapshot_recorded_content_address() throws {
        let sharedBlobLayout: SharedBlobLayout = try SharedBlobLayout.create();
        defer { try? FileManager.default.removeItem(at: sharedBlobLayout.hubDirectory); }
        try sharedBlobLayout.writeTreeRecord(
            recordedDigestText: Self.UNRELATED_CONTENT_DIGEST,
            recordedSizeBytes: UInt64(Self.CONFIG_BYTES.count));

        do {
            try sharedBlobLayout.validate();
            Issue.record("a shared blob without the recorded content address must fail closed");
        } catch let validationError as ArtifactValidationError {
            #expect(
                validationError == ArtifactValidationError.huggingFaceSharedBlobIdentityMismatch(
                    fileName: "config.json",
                    recordedDigestText: Self.UNRELATED_CONTENT_DIGEST));
        }
    }

    @Test
    func should_reject_a_shared_blob_whose_size_disagrees_with_its_snapshot_tree_record() throws {
        let sharedBlobLayout: SharedBlobLayout = try SharedBlobLayout.create();
        defer { try? FileManager.default.removeItem(at: sharedBlobLayout.hubDirectory); }
        try sharedBlobLayout.writeTreeRecord(
            recordedDigestText: Self.SHARED_BLOB_CONTENT_DIGEST,
            recordedSizeBytes: UInt64(Self.CONFIG_BYTES.count + 1));

        do {
            try sharedBlobLayout.validate();
            Issue.record("a shared blob whose size disagrees with the tree record must fail closed");
        } catch let validationError as ArtifactValidationError {
            #expect(
                validationError == ArtifactValidationError.huggingFaceSharedBlobSizeMismatch(
                    fileName: "config.json",
                    recordedSizeBytes: UInt64(Self.CONFIG_BYTES.count + 1),
                    actualSizeBytes: UInt64(Self.CONFIG_BYTES.count)));
        }
    }

    @Test
    func should_reject_a_shared_blob_without_a_snapshot_tree_record() throws {
        let sharedBlobLayout: SharedBlobLayout = try SharedBlobLayout.create();
        defer { try? FileManager.default.removeItem(at: sharedBlobLayout.hubDirectory); }

        do {
            try sharedBlobLayout.validate();
            Issue.record("a shared blob without a snapshot tree record must fail closed");
        } catch let validationError as ArtifactValidationError {
            guard case .huggingFaceSharedBlobMetadataUnavailable(let fileName, _) = validationError else {
                Issue.record("expected HuggingFaceSharedBlobMetadataUnavailable, got \(validationError)");
                return;
            }
            #expect(fileName == "config.json");
        }
    }

    @Test
    func should_reject_a_hugging_face_snapshot_symlink_that_escapes_its_blob_directory() throws {
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

        do {
            _ = try RequiredFiles.validateRequiredFileForTests(
                modelDirectory: snapshotDirectory.path,
                requiredFileProfile: RequiredFileProfile(fileName: "config.json", sizeBytes: 0));
            Issue.record("a snapshot symlink outside its own blob directory must fail closed");
        } catch let validationError as ArtifactValidationError {
            guard case .huggingFaceSnapshotSymlinkEscapesBlobDirectory(let fileName, _, _) = validationError else {
                Issue.record("expected HuggingFaceSnapshotSymlinkEscapesBlobDirectory, got \(validationError)");
                return;
            }
            #expect(fileName == "config.json");
        }
    }

    @Test
    func should_continue_rejecting_symlinks_in_regular_model_directories() throws {
        let modelDirectory: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: modelDirectory); }
        try Data("contents".utf8).write(
            to: modelDirectory.appendingPathComponent("config-contents.json"));
        try FileManager.default.createSymbolicLink(
            atPath: modelDirectory.appendingPathComponent("config.json").path,
            withDestinationPath: "config-contents.json");

        do {
            _ = try RequiredFiles.validateRequiredFileForTests(
                modelDirectory: modelDirectory.path,
                requiredFileProfile: RequiredFileProfile(fileName: "config.json", sizeBytes: 0));
            Issue.record("regular model directories must continue rejecting symlinks");
        } catch let validationError as ArtifactValidationError {
            #expect(
                validationError == ArtifactValidationError.requiredFileIsSymlink(
                    fileName: "config.json"));
        }
    }

    @Test
    func should_reject_a_required_file_name_with_parent_directory_components() throws {
        let modelDirectory: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: modelDirectory); }
        try Data("contents".utf8).write(to: modelDirectory.appendingPathComponent("outside.json"));

        do {
            _ = try RequiredFiles.validateRequiredFileForTests(
                modelDirectory: modelDirectory.path,
                requiredFileProfile: RequiredFileProfile(fileName: "../outside.json", sizeBytes: 0));
            Issue.record("required file names must not escape the model directory");
        } catch let validationError as ArtifactValidationError {
            #expect(
                validationError == ArtifactValidationError.invalidProfileFileName(
                    fileName: "../outside.json"));
        }
    }

    @Test
    func should_read_an_ordinary_json_sidecar_through_its_retained_descriptor() throws {
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
        #expect(actualSidecarBytes == retainedSidecarBytes);
    }

    @Test
    func should_reject_a_bounded_required_file_read_above_its_explicit_limit() throws {
        let modelDirectory: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: modelDirectory); }
        let sidecarFileName: String = "model.safetensors.index.json";
        let sidecarBytes: Data = Data(#"{"weight_map":{}}"#.utf8);
        try sidecarBytes.write(to: modelDirectory.appendingPathComponent(sidecarFileName));
        let validatedRequiredFile: ValidatedWeightsFile = try RequiredFiles.validateRequiredFileForTests(
            modelDirectory: modelDirectory.path,
            requiredFileProfile: RequiredFileProfile(
                fileName: sidecarFileName, sizeBytes: UInt64(sidecarBytes.count)));

        do {
            _ = try validatedRequiredFile.readBoundedBytesForTests(
                maximumSizeBytes: UInt64(sidecarBytes.count - 1));
            Issue.record("a sidecar above the caller's explicit limit must fail closed");
        } catch let validationError as ArtifactValidationError {
            #expect(
                validationError == ArtifactValidationError.boundedRequiredFileTooLarge(
                    fileName: sidecarFileName,
                    actualSizeBytes: UInt64(sidecarBytes.count),
                    maximumSizeBytes: UInt64(sidecarBytes.count - 1)));
        }
    }

    @Test
    func should_preserve_the_source_when_a_retained_descriptor_becomes_short() throws {
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

        do {
            _ = try validatedRequiredFile.readBoundedBytesForTests(
                maximumSizeBytes: UInt64(sidecarBytes.count));
            Issue.record("a short retained descriptor must fail with its read source");
        } catch let validationError as ArtifactValidationError {
            #expect(
                validationError == ArtifactValidationError.readBoundedRequiredFile(
                    fileName: sidecarFileName, problem: "failed to fill whole buffer"));
        }
    }

    @Test
    func should_reject_a_duplicate_required_profile_before_replacing_the_first_file() throws {
        let modelDirectory: URL = try Self.makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: modelDirectory); }
        try Self.CONFIG_BYTES.write(to: modelDirectory.appendingPathComponent("config.json"));
        let duplicateProfiles: Array<RequiredFileProfile> = [
            RequiredFileProfile(
                fileName: "config.json", sizeBytes: UInt64(Self.CONFIG_BYTES.count)),
            RequiredFileProfile(
                fileName: "config.json", sizeBytes: UInt64(Self.CONFIG_BYTES.count + 1)),
        ];

        do {
            try duplicateProfiles[0].validateAllForTests(
                modelDirectory: modelDirectory.path, requiredFileProfiles: duplicateProfiles);
            Issue.record("a repeated profile name must fail instead of replacing its first descriptor");
        } catch let validationError as ArtifactValidationError {
            #expect(
                validationError == ArtifactValidationError.duplicateProfileFileName(
                    fileName: "config.json"));
        }
    }
}

/**
 * One Hugging Face hub whose model entry reaches an object in the hub-level
 * shared blob store, so each journey can vary exactly one verification input.
 * Twin-port of the Rust SharedBlobLayout fixture.
 */
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
